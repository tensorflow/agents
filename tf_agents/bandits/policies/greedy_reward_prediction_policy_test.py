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
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.policies import greedy_reward_prediction_policy as greedy_reward_policy
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal


class DummyNet(network.Network):

  def __init__(self, observation_spec, num_actions=3):
    super(DummyNet, self).__init__(observation_spec, (), 'DummyNet')

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer([[1, 1.5, 2],
                                                        [1, 1.5, 4]]),
            bias_initializer=tf.constant_initializer([[1], [1], [-10]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class HeteroscedasticDummyNet(
    heteroscedastic_q_network.HeteroscedasticQNetwork):

  def __init__(self, name=None, num_actions=3):
    input_spec = array_spec.ArraySpec([2], np.float32)
    action_spec = array_spec.BoundedArraySpec([1], np.float32, 1, num_actions)

    input_tensor_spec = tensor_spec.from_spec(input_spec)
    action_tensor_spec = tensor_spec.from_spec(action_spec)

    super(HeteroscedasticDummyNet, self).__init__(input_tensor_spec,
                                                  action_tensor_spec)
    self._value_layer = tf.keras.layers.Dense(
        num_actions,
        kernel_initializer=tf.constant_initializer([[1, 1.5, 2], [1, 1.5, 4]]),
        bias_initializer=tf.constant_initializer([[1], [1], [-10]]))

    self._log_variance_layer = tf.keras.layers.Dense(
        num_actions,
        kernel_initializer=tf.constant_initializer([[1, 1.5, 2], [1, 1.5, 4]]),
        bias_initializer=tf.constant_initializer([[1], [1], [-10]]))

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    value = self._value_layer(inputs)
    log_variance = self._log_variance_layer(inputs)
    predictions = collections.namedtuple('QBanditNetworkResult',
                                         ('q_value_logits', 'log_variance'))
    predictions = predictions(value, log_variance)

    return predictions, network_state


@test_util.run_all_in_graph_and_eager_modes
class GreedyRewardPredictionPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(GreedyRewardPredictionPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)

  def testBuild(self):
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec))

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  def testMultipleActionsRaiseError(self):
    action_spec = [tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)] * 2
    with self.assertRaisesRegexp(
        NotImplementedError,
        'action_spec can only contain a single BoundedTensorSpec'):
      greedy_reward_policy.GreedyRewardPredictionPolicy(
          self._time_step_spec,
          action_spec,
          reward_network=DummyNet(self._obs_spec))

  def testWrongActionsRaiseError(self):
    action_spec = tensor_spec.BoundedTensorSpec((5, 6, 7), tf.float32, 0, 2)
    with self.assertRaisesRegexp(
        NotImplementedError,
        'action_spec must be a BoundedTensorSpec of type int32.*'):
      greedy_reward_policy.GreedyRewardPredictionPolicy(
          self._time_step_spec,
          action_spec,
          reward_network=DummyNet(self._obs_spec))

  def testWrongOutputLayerRaiseError(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 20)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        action_spec,
        reward_network=DummyNet(self._obs_spec))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    with self.assertRaisesRegexp(
        ValueError,
        r'The number of actions \(11\) does not match the reward_network output'
        r' size \(3\)\.'):
      policy.action(time_step, seed=1)

  def testAction(self):
    tf.compat.v1.set_random_seed(1)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])

  def testActionHeteroscedastic(self):
    tf.compat.v1.set_random_seed(1)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec, self._action_spec,
        reward_network=HeteroscedasticDummyNet())
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])

  def testActionScalarSpec(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        action_spec,
        reward_network=DummyNet(self._obs_spec))

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])

  def testActionScalarSpecWithShift(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 12)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        action_spec,
        reward_network=DummyNet(self._obs_spec))

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [11, 12])

  def testMaskedAction(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    observation_spec = (tensor_spec.TensorSpec([2], tf.float32),
                        tensor_spec.TensorSpec([3], tf.int32))
    time_step_spec = ts.time_step_spec(observation_spec)

    def split_fn(obs):
      return obs[0], obs[1]

    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        time_step_spec,
        action_spec,
        reward_network=DummyNet(observation_spec[0]),
        observation_and_action_constraint_splitter=split_fn)

    observations = (tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
                    tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32))
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [2, 1])

  def testUpdate(self):
    tf.compat.v1.set_random_seed(1)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec))
    new_policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec))

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)

    action_step = policy.action(time_step, seed=1)
    new_action_step = new_policy.action(time_step, seed=1)

    self.assertEqual(len(policy.variables()), 2)
    self.assertEqual(len(new_policy.variables()), 2)
    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertIsNone(self.evaluate(new_policy.update(policy)))

    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])
    self.assertAllEqual(self.evaluate(new_action_step.action), [1, 2])

  def testPredictedRewards(self):
    tf.compat.v1.set_random_seed(1)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        emit_policy_info=('predicted_rewards_mean',))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])
    # The expected values are obtained by passing the observation through the
    # Keras dense layer of the DummyNet (defined above).
    predicted_rewards_expected_array = np.array([[4.0, 5.5, 0.0],
                                                 [8.0, 11.5, 12.0]])
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)

  def testPerArmRewards(self):
    tf.compat.v1.set_random_seed(3000)
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 4)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    reward_network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))

    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        time_step_spec,
        action_spec,
        reward_network=reward_network,
        accepts_per_arm_features=True,
        emit_policy_info=('predicted_rewards_mean',))
    action_feature = tf.cast(
        tf.reshape(tf.random.shuffle(tf.range(24)), shape=[2, 4, 3]),
        dtype=tf.float32)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY: action_feature
    }
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action, p_info, first_arm_features = self.evaluate([
        action_step.action, action_step.info,
        observations[bandit_spec_utils.PER_ARM_FEATURE_KEY][0]
    ])
    self.assertAllEqual(action.shape, [2])
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 4])
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
    padded_action_feature = tf.concat(
        [action_feature[:, 0:1, :],
         tf.zeros(shape=[2, 3, 3], dtype=tf.float32)],
        axis=1)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY: padded_action_feature
    }
    time_step = ts.restart(observations, batch_size=2)
    padded_action_step = policy.action(time_step, seed=1)
    padded_p_info = self.evaluate(padded_action_step.info)
    self.assertAllEqual(p_info.predicted_rewards_mean[:, 0],
                        padded_p_info.predicted_rewards_mean[:, 0])

  def testPerArmRewardsVariableNumActions(self):
    tf.compat.v1.set_random_seed(3000)
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        2, 3, 4, add_num_actions_feature=True)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    reward_network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))

    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        time_step_spec,
        action_spec,
        reward_network=reward_network,
        accepts_per_arm_features=True,
        emit_policy_info=('predicted_rewards_mean',))
    action_feature = tf.cast(
        tf.reshape(tf.random.shuffle(tf.range(24)), shape=[2, 4, 3]),
        dtype=tf.float32)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            action_feature,
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
            tf.constant([2, 3], dtype=tf.int32)
    }
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action, p_info, first_arm_features = self.evaluate([
        action_step.action, action_step.info,
        observations[bandit_spec_utils.PER_ARM_FEATURE_KEY][0]
    ])
    self.assertAllEqual(action.shape, [2])
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 4])
    self.assertAllEqual(p_info.chosen_arm_features.shape, [2, 3])
    first_action = action[0]
    self.assertAllEqual(p_info.chosen_arm_features[0],
                        first_arm_features[first_action])

  def testPerArmRewardsSparseObs(self):
    tf.compat.v1.set_random_seed(3000)
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

    reward_network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            observation_spec=obs_spec,
            global_layers=(4, 3, 2),
            arm_layers=(6, 5, 4),
            common_layers=(7, 6, 5),
            global_preprocessing_combiner=(
                tf.compat.v2.keras.layers.DenseFeatures([columns_c])),
            arm_preprocessing_combiner=tf.compat.v2.keras.layers.DenseFeatures(
                [columns_a, columns_b])))

    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        time_step_spec,
        action_spec,
        reward_network=reward_network,
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
    action_step = policy.action(time_step, seed=1)
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
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 3])
    self.assertAllEqual(p_info.chosen_arm_features['name'].shape, [2])
    self.assertAllEqual(p_info.chosen_arm_features['fruit'].shape, [2])
    first_action = action[0]
    self.assertAllEqual(p_info.chosen_arm_features['name'][0],
                        first_arm_name_feature[first_action])

  def testPolicyWithConstraints(self):
    constraint_net = DummyNet(self._obs_spec)
    neural_constraint = constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

    tf.compat.v1.set_random_seed(1)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        constraints=[neural_constraint],
        emit_policy_info=('predicted_rewards_mean',))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])
    # The expected values are obtained by passing the observation through the
    # Keras dense layer of the DummyNet (defined above).
    predicted_rewards_expected_array = np.array([[4.0, 5.5, 0.0],
                                                 [8.0, 11.5, 12.0]])
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)


if __name__ == '__main__':
  tf.test.main()
