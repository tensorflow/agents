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

"""Test for boltzmann_reward_prediction_policy."""

import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import boltzmann_reward_prediction_policy as boltzmann_reward_policy
from tf_agents.networks import network
from tf_agents.policies import utils
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


@test_util.run_all_in_graph_and_eager_modes
class BoltzmannRewardPredictionPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(BoltzmannRewardPredictionPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)

  def testBoltzmannGumbelPredictedRewards(self):
    tf.compat.v1.set_random_seed(1)
    num_samples_list = []
    for k in range(3):
      num_samples_list.append(
          tf.compat.v2.Variable(
              tf.zeros([], dtype=tf.int32), name='num_samples_{}'.format(k)))
    num_samples_list[0].assign_add(2)
    num_samples_list[1].assign_add(4)
    num_samples_list[2].assign_add(1)
    policy = boltzmann_reward_policy.BoltzmannRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        boltzmann_gumbel_exploration_constant=10.0,
        emit_policy_info=(utils.InfoFields.PREDICTED_REWARDS_MEAN,),
        num_samples_list=num_samples_list)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    self.assertAllEqual(p_info.predicted_rewards_mean.shape, [2, 3])

  def testLargeTemperature(self):
    # With a very large temperature, the sampling probability will be uniform.
    policy = boltzmann_reward_policy.BoltzmannRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        temperature=10e8,
        emit_policy_info=(utils.InfoFields.LOG_PROBABILITY,))
    batch_size = 3000
    observations = tf.constant([[1, 2]] * batch_size, dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=batch_size)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are uniform.
    self.assertAllEqual(p_info.log_probability,
                        tf.math.log([1.0 / 3] * batch_size))
    # Check the empirical distribution of the chosen arms is uniform.
    actions = self.evaluate(action_step.action)
    self.assertAllInSet(actions, [0, 1, 2])
    # Set tolerance in the chosen count to be 4 std.
    tol = 4.0 * np.sqrt(batch_size * 1.0 / 3 * 2.0 / 3)
    for action in range(3):
      action_chosen_count = np.sum(actions == action)
      self.assertNear(
          action_chosen_count,
          1000,
          tol,
          msg=f'action: {action} is expected to be chosen between {1000 - tol} '
          f'and {1000 + tol} times, but was actually chosen '
          f'{action_chosen_count} times.')

  def testZeroTemperature(self):
    # With zero temperature, the chosen actions should be greedy.
    policy = boltzmann_reward_policy.BoltzmannRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        temperature=0.0,
        emit_policy_info=(utils.InfoFields.LOG_PROBABILITY,))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  def testZeroGumbelExploration(self):
    # When the Boltzmann-Gumbel exploration constant is almost 0, the chosen
    # actions should be greedy actions.
    num_samples_list = []
    for k in range(3):
      num_samples_list.append(
          tf.compat.v2.Variable(
              tf.zeros([], dtype=tf.int32), name='num_samples_{}'.format(k)))
    num_samples_list[0].assign_add(2)
    num_samples_list[1].assign_add(4)
    num_samples_list[2].assign_add(1)
    policy = boltzmann_reward_policy.BoltzmannRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        boltzmann_gumbel_exploration_constant=1e-12,
        num_samples_list=num_samples_list,
        emit_policy_info=(utils.InfoFields.PREDICTED_REWARDS_MEAN,))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  def testAllLargeNumSamples(self):
    # When every action has a very large number of samples, the chosen actions
    # should be greedy actions.
    num_samples_list = []
    for k in range(3):
      num_samples_list.append(
          tf.compat.v2.Variable(
              tf.zeros([], dtype=tf.int32), name='num_samples_{}'.format(k)))
    num_samples_list[0].assign_add(tf.int32.max - 10)
    num_samples_list[1].assign_add(tf.int32.max - 10)
    num_samples_list[2].assign_add(tf.int32.max - 10)
    policy = boltzmann_reward_policy.BoltzmannRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        boltzmann_gumbel_exploration_constant=100.0,
        num_samples_list=num_samples_list,
        emit_policy_info=(utils.InfoFields.PREDICTED_REWARDS_MEAN,))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  def testSomeSmallNumSamples(self):
    # When some action has a much smaller number of samples, it should be chosen
    # more frequently than other actions.
    num_samples_list = []
    for k in range(3):
      num_samples_list.append(
          tf.compat.v2.Variable(
              tf.zeros([], dtype=tf.int32), name='num_samples_{}'.format(k)))
    num_samples_list[0].assign_add(tf.int32.max - 10)
    num_samples_list[1].assign_add(1)
    num_samples_list[2].assign_add(tf.int32.max - 10)
    policy = boltzmann_reward_policy.BoltzmannRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        boltzmann_gumbel_exploration_constant=10.0,
        num_samples_list=num_samples_list,
        emit_policy_info=(utils.InfoFields.PREDICTED_REWARDS_MEAN,))
    batch_size = 3000
    observations = tf.constant([[1, 2]] * batch_size, dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=batch_size)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertAllInSet(actions, [0, 1, 2])
    action_counts = {action: np.sum(actions == action) for action in range(3)}
    self.assertAllLess([action_counts[0], action_counts[2]], action_counts[1])

if __name__ == '__main__':
  tf.test.main()
