# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test for greedy_reward_prediction_policy."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import List, Dict, Text, Tuple

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.multi_objective import multi_objective_scalarizer
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import greedy_multi_objective_neural_policy as greedy_multi_objective_policy
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import test_utils

from tensorflow.python.framework import test_util  # type: ignore pylint: disable=g-direct-tensorflow-import  # TF internal


class DummyNet(network.Network):

  def __init__(self, observation_spec: types.Nested[tf.TypeSpec],
               kernel_weights: np.ndarray, bias: np.ndarray):
    """A simple linear network.

    Args:
      observation_spec: The observation specification.
      kernel_weights: A 2-d numpy array of shape [input_size, output_size].
      bias: A 1-d numpy array of shape [output_size].
    """
    super(DummyNet, self).__init__(observation_spec, (), 'DummyNet')
    assert len(kernel_weights.shape) == 2
    assert len(bias.shape) == 1
    assert kernel_weights.shape[1] == bias.shape[0]

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            kernel_weights.shape[1],
            kernel_initializer=tf.compat.v1.initializers.constant(
                kernel_weights),
            bias_initializer=tf.compat.v1.initializers.constant(bias))
    ]

  def call(self, inputs: tf.Tensor, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class HeteroscedasticDummyNet(
    heteroscedastic_q_network.HeteroscedasticQNetwork):

  def __init__(self, kernel_weights: np.ndarray, bias: np.ndarray):
    """A simple linear heteroscedastic network.

    Args:
      kernel_weights: A 2-d numpy array of shape [input_size, output_size].
      bias: A 1-d numpy array of shape [output_size].
    """
    assert len(kernel_weights.shape) == 2
    assert len(bias.shape) == 1
    assert kernel_weights.shape[1] == bias.shape[0]

    input_spec = array_spec.ArraySpec([kernel_weights.shape[0]], np.float32)
    action_spec = array_spec.BoundedArraySpec([1], np.float32, 1,
                                              kernel_weights.shape[1])

    input_tensor_spec = tensor_spec.from_spec(input_spec)
    action_tensor_spec = tensor_spec.from_spec(action_spec)

    super(HeteroscedasticDummyNet, self).__init__(input_tensor_spec,
                                                  action_tensor_spec)
    self._value_layer = tf.keras.layers.Dense(
        kernel_weights.shape[1],
        kernel_initializer=tf.compat.v1.initializers.constant(kernel_weights),
        bias_initializer=tf.compat.v1.initializers.constant(bias))

    self._log_variance_layer = tf.keras.layers.Dense(
        kernel_weights.shape[1],
        kernel_initializer=tf.compat.v1.initializers.constant(kernel_weights),
        bias_initializer=tf.compat.v1.initializers.constant(bias))

  def call(self, inputs: tf.Tensor, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    value = self._value_layer(inputs)
    log_variance = self._log_variance_layer(inputs)
    predictions = collections.namedtuple('QBanditNetworkResult',
                                         ('q_value_logits', 'log_variance'))
    predictions = predictions(value, log_variance)

    return predictions, network_state


@test_util.run_all_in_graph_and_eager_modes
class ScalarizeObjectivesTest(test_utils.TestCase):

  def setUp(self):
    super(ScalarizeObjectivesTest, self).setUp()
    hv_params = [
        multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
            slope=1, offset=0)
    ] * 3
    self._scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0, 1, 0.01], hv_params)

  def testScalarizeObjectivesWrongInputRankRaisesError(self):
    objectives_tensor = tf.constant([1], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, 'The objectives_tensor should be rank-3, but is rank-1'):
      greedy_multi_objective_policy.scalarize_objectives(
          objectives_tensor, self._scalarizer)

  def testScalarizeObjectivesWrongNumberOfObjectiveRaisesError(self):
    objectives_tensor = tf.constant([[[1, 2, 3]], [[4, 5, 6]]],
                                    dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, 'The number of input objectives should be 3, but is 1'):
      self.evaluate(
          greedy_multi_objective_policy.scalarize_objectives(
              objectives_tensor, self._scalarizer))

  def testScalarizeObjectives(self):
    objectives_tensor = tf.constant(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
        dtype=tf.float32)
    scalarized_reward = greedy_multi_objective_policy.scalarize_objectives(
        objectives_tensor, self._scalarizer)
    self.assertAllClose(
        self.evaluate(scalarized_reward), [[3, 4], [9, 10]],
        rtol=1e-4,
        atol=1e-3)


@test_util.run_all_in_graph_and_eager_modes
class GreedyRewardPredictionPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(GreedyRewardPredictionPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    hv_params = [
        multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
            slope=1, offset=0)
    ] * 3
    self._scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0, 1, 0.01], hv_params)
    self._bias = np.asarray([-1, -1, -1])
    self._kernel_weights = [
        np.asarray([[1, 2, 3], [4, 5, 6]]),
        np.asarray([[3, 1, 2], [5, 4, 6]]),
        np.asarray([[2, 3, 1], [5, 6, 4]])
    ]

  def _create_heteroscedastic_networks(self) -> List[HeteroscedasticDummyNet]:
    return [
        HeteroscedasticDummyNet(weights, self._bias)
        for weights in self._kernel_weights
    ]

  def _create_objective_networks(self) -> List[DummyNet]:
    return [
        DummyNet(self._obs_spec, weights, self._bias)
        for weights in self._kernel_weights
    ]

  def _create_arm_policy_and_observations(
      self
  ) -> Tuple[greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy,
             Dict[Text, tf.Tensor]]:
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 4)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    objective_networks = [
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)) for _ in range(3)
    ]
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        time_step_spec,
        action_spec,
        self._scalarizer,
        objective_networks,
        accepts_per_arm_features=True,
        emit_policy_info=('predicted_rewards_mean',))
    action_feature = tf.cast(
        tf.reshape(tf.random.shuffle(tf.range(24)), shape=[2, 4, 3]),
        dtype=tf.float32)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [2, 1]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            action_feature
    }
    return policy, observations

  def testBuild(self):
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, self._action_spec, self._scalarizer,
        self._create_objective_networks())

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  def testMultipleActionsRaiseError(self):
    action_spec = [tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)] * 2
    with self.assertRaisesRegexp(
        NotImplementedError,
        'action_spec can only contain a single BoundedTensorSpec'):
      greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
          self._time_step_spec, action_spec, self._scalarizer,
          self._create_objective_networks())

  def testTooFewNetworksRaiseError(self):
    with self.assertRaisesRegexp(
        ValueError,
        'Number of objectives should be at least two, but found to be 1'):
      greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
          self._time_step_spec, self._action_spec, self._scalarizer,
          [self._create_objective_networks()[0]])

  def testWrongActionsRaiseError(self):
    action_spec = tensor_spec.BoundedTensorSpec((5, 6, 7), tf.float32, 0, 2)
    with self.assertRaisesRegexp(
        NotImplementedError,
        'action_spec must be a BoundedTensorSpec of type int32.*'):
      greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
          self._time_step_spec, action_spec, self._scalarizer,
          self._create_objective_networks())

  def testWrongOutputLayerRaiseError(self):
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 20)
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, action_spec, self._scalarizer,
        self._create_objective_networks())
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    with self.assertRaisesRegexp(
        ValueError,
        r'The number of actions \(11\) does not match objective network 0'
        r' output size \(3\)\.'):
      policy.action(time_step)

  def testUnmatchingPolicyState(self):
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, self._action_spec, self._scalarizer,
        self._create_objective_networks())
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    with self.assertRaisesRegexp(
        ValueError,
        'policy_state and policy_state_spec structures do not match:'):
      policy.action(time_step, policy_state=[()])

  def testAction(self):
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, self._action_spec, self._scalarizer,
        self._create_objective_networks())
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [2, 0])

  def testActionHeteroscedastic(self):
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, self._action_spec, self._scalarizer,
        self._create_heteroscedastic_networks())
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [2, 0])

  def testActionScalarSpecWithShift(self):
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 12)
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, action_spec, self._scalarizer,
        self._create_objective_networks())

    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [12, 10])

  def testMaskedAction(self):
    observation_spec = (tensor_spec.TensorSpec([2], tf.float32),
                        tensor_spec.TensorSpec([3], tf.int32))
    time_step_spec = ts.time_step_spec(observation_spec)

    def split_fn(obs):
      return obs[0], obs[1]

    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        time_step_spec,
        self._action_spec,
        self._scalarizer,
        self._create_objective_networks(),
        observation_and_action_constraint_splitter=split_fn)

    observations = (tf.constant([[1, 2], [2, 1]], dtype=tf.float32),
                    tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32))
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [2, 1])

  def testUpdate(self):
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, self._action_spec, self._scalarizer,
        self._create_objective_networks())
    new_policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec, self._action_spec, self._scalarizer,
        self._create_objective_networks())

    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)

    action_step = policy.action(time_step)
    new_action_step = new_policy.action(time_step)

    self.assertEqual(len(policy.variables()), 6)
    self.assertEqual(len(new_policy.variables()), 6)
    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertIsNone(self.evaluate(new_policy.update(policy)))

    self.assertAllEqual(self.evaluate(action_step.action), [2, 0])
    self.assertAllEqual(self.evaluate(new_action_step.action), [2, 0])

  def testPredictedRewards(self):
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        self._create_objective_networks(),
        emit_policy_info=('predicted_rewards_mean',))
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [2, 0])
    # The expected values are obtained by passing the observation through the
    # Keras dense layer of the DummyNet (defined above).
    predicted_rewards_expected_array = np.array([[[8, 11, 14], [12, 8, 13],
                                                  [11, 14, 8]],
                                                 [[5, 8, 11], [10, 5, 9],
                                                  [8, 11, 5]]])
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)

  def testNoneTimeStepSpecForPerArmFeaturesRaisesError(self):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 4)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    objective_networks = [
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)) for _ in range(3)
    ]
    with self.assertRaisesRegexp(
        ValueError,
        'time_step_spec should not be None for per-arm-features policies'):
      greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
          None,
          action_spec,
          self._scalarizer,
          objective_networks,
          accepts_per_arm_features=True,
          emit_policy_info=('predicted_rewards_mean',))

  def testPerArmRewards(self):
    policy, observations = self._create_arm_policy_and_observations()
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action, p_info, first_arm_features = self.evaluate([
        action_step.action, action_step.info,
        observations[bandit_spec_utils.PER_ARM_FEATURE_KEY][0]
    ])
    self.assertAllEqual(action.shape, [2])
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 3, 4])
    self.assertAllEqual(p_info.chosen_arm_features.shape, [2, 3])
    first_action = action[0]
    self.assertAllEqual(p_info.chosen_arm_features[0],
                        first_arm_features[first_action])

    # Check that zeroing out some of the actions does not affect the predicted
    # rewards for unchanged actions. This is to make sure that action feature
    # padding does not influence the behavior.

    if not tf.executing_eagerly():
      # The below comparison will only work in tf2 because of the random per-arm
      # observations get re-drawn in tf1.
      return
    action_feature = observations[bandit_spec_utils.PER_ARM_FEATURE_KEY]
    padded_action_feature = tf.concat(
        [action_feature[:, 0:1, :],
         tf.zeros(shape=[2, 3, 3], dtype=tf.float32)],
        axis=1)
    observations[bandit_spec_utils.PER_ARM_FEATURE_KEY] = padded_action_feature
    time_step = ts.restart(observations, batch_size=2)
    padded_action_step = policy.action(time_step)
    padded_p_info = self.evaluate(padded_action_step.info)
    self.assertAllEqual(p_info.predicted_rewards_mean[:, :, 0],
                        padded_p_info.predicted_rewards_mean[:, :, 0])

  def testPerArmRewardsVariableNumActions(self):
    policy, observations = self._create_arm_policy_and_observations()
    observations[bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY] = tf.constant(
        [1, 1], dtype=tf.int32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action, p_info, first_arm_features = self.evaluate([
        action_step.action, action_step.info,
        observations[bandit_spec_utils.PER_ARM_FEATURE_KEY][0]
    ])
    self.assertAllEqual(action.shape, [2])
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 3, 4])
    self.assertAllEqual(p_info.chosen_arm_features.shape, [2, 3])
    first_action = action[0]
    self.assertEqual(first_action, 0)
    self.assertAllEqual(p_info.chosen_arm_features[0],
                        first_arm_features[first_action])
    self.assertEqual(action[1], 0)

  def testPerArmRewardsSparseObs(self):
    obs_spec = {
        'global': {'sport': tensor_spec.TensorSpec((), tf.string)},
        'per_arm': {
            'name': tensor_spec.TensorSpec((3,), tf.string),
            'fruit': tensor_spec.TensorSpec((3,), tf.string)
        }
    }
    columns_a = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'name', ['bob', 'george', 'wanda']))
    columns_b = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'fruit', ['banana', 'kiwi', 'pear']))
    columns_c = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'sport', ['bridge', 'chess', 'snooker']))

    objective_networks = []
    for _ in range(3):
      objective_networks.append(
          global_and_arm_feature_network
          .create_feed_forward_common_tower_network(
              observation_spec=obs_spec,
              global_layers=(4, 3, 2),
              arm_layers=(6, 5, 4),
              common_layers=(7, 6, 5),
              global_preprocessing_combiner=(
                  tf.compat.v2.keras.layers.DenseFeatures([columns_c])),
              arm_preprocessing_combiner=tf.compat.v2.keras.layers
              .DenseFeatures([columns_a, columns_b])))
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        time_step_spec,
        action_spec,
        self._scalarizer,
        objective_networks,
        accepts_per_arm_features=True,
        emit_policy_info=('predicted_rewards_mean',))
    observations = {
        'global': {
            'sport': tf.constant(['snooker', 'chess'])
        },
        'per_arm': {
            'name':
                tf.constant([['george', 'george', 'george'],
                             ['bob', 'bob', 'bob']]),
            'fruit':
                tf.constant([['banana', 'banana', 'banana'],
                             ['kiwi', 'kiwi', 'kiwi']])
        }
    }

    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.tables_initializer()
    ])
    action, p_info, first_arm_name_feature = self.evaluate([
        action_step.action, action_step.info,
        observations[bandit_spec_utils.PER_ARM_FEATURE_KEY]['name'][0]
    ])
    self.assertAllEqual(action.shape, [2])
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 3, 3])
    self.assertAllEqual(p_info.chosen_arm_features['name'].shape, [2])
    self.assertAllEqual(p_info.chosen_arm_features['fruit'].shape, [2])
    first_action = action[0]
    self.assertAllEqual(p_info.chosen_arm_features['name'][0],
                        first_arm_name_feature[first_action])


if __name__ == '__main__':
  tf.test.main()
