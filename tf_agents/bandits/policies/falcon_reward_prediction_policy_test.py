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

"""Test for falcon_reward_prediction_policy."""

import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import falcon_reward_prediction_policy
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


class GetNumberOfTrainableElementsTest(test_utils.TestCase):

  def testNumberOfTrainableElements(self):
    dummy_net = DummyNet(tensor_spec.TensorSpec([2], tf.float32))
    dummy_net.create_variables()
    self.assertEqual(
        falcon_reward_prediction_policy.get_number_of_trainable_elements(
            dummy_net), 9)


@test_util.run_all_in_graph_and_eager_modes
class FalconRewardPredictionPolicyTest(test_utils.TestCase):

  def _check_uniform_actions(self, actions: np.ndarray,
                             batch_size: int) -> None:
    self.assertAllInSet(actions, [0, 1, 2])
    # Set tolerance in the chosen count to be 4 std.
    tol = 4.0 * np.sqrt(batch_size * 1.0 / 3 * 2.0 / 3)
    expected_count = batch_size / 3
    for action in range(3):
      action_chosen_count = np.sum(actions == action)
      self.assertNear(
          action_chosen_count,
          expected_count,
          tol,
          msg=f'action: {action} is expected to be chosen between '
          f'{expected_count - tol} and {expected_count + tol} times, but was '
          f'actually chosen {action_chosen_count} times.')

  def setUp(self):
    super(FalconRewardPredictionPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)

  def testBanditPolicyType(self):
    policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        emit_policy_info=(utils.InfoFields.BANDIT_POLICY_TYPE,))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(
        p_info.bandit_policy_type,
        [[utils.BanditPolicyType.FALCON], [utils.BanditPolicyType.FALCON]])

  def testZeroExploitationCoefficient(self):
    # With a zero exploitation coefficient, the sampling probability will be
    # uniform.
    num_samples_list = [
        tf.compat.v2.Variable(2, dtype=tf.int32, name='num_samples_0'),
        tf.compat.v2.Variable(4, dtype=tf.int32, name='num_samples_1'),
        tf.compat.v2.Variable(1, dtype=tf.int32, name='num_samples_2')
    ]
    policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        exploitation_coefficient=0.0,
        num_samples_list=num_samples_list,
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
    self._check_uniform_actions(actions, batch_size)

  def testLargeExploitationCoefficient(self):
    # With a very large exploitation coefficient and a positive number of
    # samples, the sampled actions will be greedy.
    num_samples_list = [
        tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_0'),
        tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_1'),
        tf.compat.v2.Variable(10, dtype=tf.int32, name='num_samples_2')
    ]
    policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        exploitation_coefficient=10.0**20,
        num_samples_list=num_samples_list,
        emit_policy_info=(utils.InfoFields.LOG_PROBABILITY,))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are greedy.
    self.assertAllClose(p_info.log_probability, [0.0, 0.0])
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  def testZeroNumSamples(self):
    # With number of samples being 0, the sampling probabiity will be uniform.
    num_samples_list = [
        tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_0'),
        tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_1'),
        tf.compat.v2.Variable(0, dtype=tf.int32, name='num_samples_2')
    ]
    policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        exploitation_coefficient=10.0,
        num_samples_list=num_samples_list,
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
    self._check_uniform_actions(actions, batch_size)

  def testLargeNumSamples(self):
    # With very large numbers of samples and a positive learning rate, the
    # sampled actions will be greedy.
    num_samples_list = [
        tf.compat.v2.Variable(int(1e10), dtype=tf.int32, name='num_samples_0'),
        tf.compat.v2.Variable(int(1e10), dtype=tf.int32, name='num_samples_1'),
        tf.compat.v2.Variable(int(1e10), dtype=tf.int32, name='num_samples_2')
    ]
    policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec),
        exploitation_coefficient=0.1,
        num_samples_list=num_samples_list,
        emit_policy_info=(utils.InfoFields.LOG_PROBABILITY,))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are greedy.
    self.assertAllClose(p_info.log_probability, [0.0, 0.0], atol=1e-3)
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

if __name__ == '__main__':
  tf.test.main()
