# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for policies sub-classing RewardPredictionBasePolicy.

These tests cover expected common behavior of policies sub-classing
`RewardPredictionBasePolicy`. A new sub-class should be accompanied with a
corresponding new test case added to the `test_cases` method below.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import boltzmann_reward_prediction_policy
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.policies import falcon_reward_prediction_policy
from tf_agents.bandits.policies import greedy_reward_prediction_policy
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


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name':
              'Greedy',
          'policy_class':
              greedy_reward_prediction_policy.GreedyRewardPredictionPolicy
      }, {
          'testcase_name':
              'Boltzmann',
          'policy_class':
              boltzmann_reward_prediction_policy.BoltzmannRewardPredictionPolicy
      }, {
          'testcase_name':
              'Falcon',
          'policy_class':
              falcon_reward_prediction_policy.FalconRewardPredictionPolicy
      })


@test_util.run_all_in_graph_and_eager_modes
class RewardPredictionPoliciesTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RewardPredictionPoliciesTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)

  @test_cases()
  def testBuild(self, policy_class):
    policy = policy_class(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec))

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  @test_cases()
  def testMultipleActionsRaiseError(self, policy_class):
    action_spec = [tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)] * 2
    with self.assertRaisesRegex(
        NotImplementedError,
        'action_spec can only contain a single BoundedTensorSpec'):
      policy_class(
          self._time_step_spec,
          action_spec,
          reward_network=DummyNet(self._obs_spec))

  @test_cases()
  def testWrongActionsRaiseError(self, policy_class):
    action_spec = tensor_spec.BoundedTensorSpec((5, 6, 7), tf.float32, 0, 2)
    with self.assertRaisesRegex(
        NotImplementedError,
        'action_spec must be a BoundedTensorSpec of type int32.*'):
      policy_class(
          self._time_step_spec,
          action_spec,
          reward_network=DummyNet(self._obs_spec))

  @test_cases()
  def testWrongOutputLayerRaiseError(self, policy_class):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 20)
    policy = policy_class(
        self._time_step_spec,
        action_spec,
        reward_network=DummyNet(self._obs_spec))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    with self.assertRaisesRegex(
        ValueError,
        r'The number of actions \(11\) does not match the reward_network output'
        r' size \(3\)\.'):
      policy.action(time_step, seed=1)

  @test_cases()
  def testAction(self, policy_class):
    tf.compat.v1.set_random_seed(1)
    policy = policy_class(
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
    self.assertAllInSet(self.evaluate(action_step.action), [0, 1, 2])

  @test_cases()
  def testActionHeteroscedastic(self, policy_class):
    tf.compat.v1.set_random_seed(1)
    policy = policy_class(
        self._time_step_spec,
        self._action_spec,
        reward_network=HeteroscedasticDummyNet())
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllInSet(self.evaluate(action_step.action), [0, 1, 2])

  @test_cases()
  def testActionScalarSpecWithShift(self, policy_class):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 12)
    policy = policy_class(
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
    self.assertAllInSet(self.evaluate(action_step.action), [10, 11, 12])

  @test_cases()
  def testMaskedAction(self, policy_class):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    observation_spec = (tensor_spec.TensorSpec([2], tf.float32),
                        tensor_spec.TensorSpec([3], tf.int32))
    time_step_spec = ts.time_step_spec(observation_spec)

    def split_fn(obs):
      return obs[0], obs[1]

    policy = policy_class(
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
    self.assertAllInSet(self.evaluate(action_step.action), [2, 1])

  @test_cases()
  def testPredictedRewards(self, policy_class):
    tf.compat.v1.set_random_seed(1)
    policy = policy_class(
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
    # The expected values are obtained by passing the observation through the
    # Keras dense layer of the DummyNet (defined above).
    predicted_rewards_expected_array = np.array([[4.0, 5.5, 0.0],
                                                 [8.0, 11.5, 12.0]])
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)

  @test_cases()
  def testPerArmRewards(self, policy_class):
    tf.compat.v1.set_random_seed(3000)
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 4)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    reward_network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))

    policy = policy_class(
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

  @test_cases()
  def testPerArmPolicyDistribution(self, policy_class):
    tf.compat.v1.set_random_seed(3000)
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 4)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    reward_network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))

    policy = policy_class(
        time_step_spec,
        action_spec,
        reward_network=reward_network,
        accepts_per_arm_features=True)
    action_feature = tf.cast(
        tf.reshape(tf.random.shuffle(tf.range(24)), shape=[2, 4, 3]),
        dtype=tf.float32)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            action_feature
    }
    time_step = ts.restart(observations, batch_size=2)
    distribution_step = policy.distribution(time_step)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    info = self.evaluate(distribution_step.info)
    self.assertAllEqual(info.chosen_arm_features.shape, [2, 3])

  @test_cases()
  def testPerArmRewardsVariableNumActions(self, policy_class):
    tf.compat.v1.set_random_seed(3000)
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        2, 3, 4, add_num_actions_feature=True)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    reward_network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))

    policy = policy_class(
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

  @test_cases()
  def testPerArmRewardsSparseObs(self, policy_class):
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
    policy = policy_class(
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

  @test_cases()
  def testPolicyWithConstraints(self, policy_class):
    constraint_net = DummyNet(self._obs_spec)
    # Create an `AbsoluteConstraint` where feasible actions must have predicted
    # rewards at most 0.0.
    absolute_constraint = constraints.AbsoluteConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net,
        comparator_fn=tf.less_equal,
        absolute_value=0.0)

    tf.compat.v1.set_random_seed(1)
    policy = policy_class(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        constraints=[absolute_constraint],
        emit_policy_info=('predicted_rewards_mean',))
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # The expected values are obtained by passing the observation through the
    # Keras dense layer of the DummyNet (defined above).
    predicted_rewards_expected_array = np.array([[4.0, 5.5, 0.0],
                                                 [4.0, 5.5, -2.0]])
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)
    # Under the `absolute_constraint`, only actions with predicted rewards
    # at most 0.0 are feasible.
    self.assertAllEqual(self.evaluate(action_step.action), [2, 2])


if __name__ == '__main__':
  tf.test.main()
