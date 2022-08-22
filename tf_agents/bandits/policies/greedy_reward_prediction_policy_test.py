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

"""Test for greedy_reward_prediction_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import greedy_reward_prediction_policy as greedy_reward_policy
from tf_agents.networks import network
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
class GreedyRewardPredictionPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(GreedyRewardPredictionPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)

  def testGreedyAction(self):
    tf.compat.v1.set_random_seed(1)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        self._action_spec,
        reward_network=DummyNet(self._obs_spec))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [1, 2])

  def testGreedyActionScalarSpecWithShift(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 12)
    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        self._time_step_spec,
        action_spec,
        reward_network=DummyNet(self._obs_spec))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [11, 12])

  def testMaskedGreedyAction(self):
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


if __name__ == '__main__':
  tf.test.main()
