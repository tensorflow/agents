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

"""Tests for neural_epsilon_greedy_agent.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


class DummyNet(network.Network):

  def __init__(self, observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

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


class AgentTest(tf.test.TestCase):

  def setUp(self):
    super(AgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation

  def testPolicyWithEpsilonGreedy(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=None,
        epsilon=0.1)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual([2], action_step.action.shape)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertIn(actions[0], [0, 1, 2])
    self.assertIn(actions[1], [0, 1, 2])

  def testPolicyWithEpsilonGreedyAndActionMask(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    obs_spec = (tensor_spec.TensorSpec([2], tf.float32),
                tensor_spec.TensorSpec([3], tf.int32))
    agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        ts.time_step_spec(obs_spec),
        self._action_spec,
        reward_network=reward_net,
        optimizer=None,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]),
        epsilon=0.1)
    observations = (tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
                    tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32))
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual([2], action_step.action.shape)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [2, 1])

  def testTrainPerArmAgent(self):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 3)
    time_step_spec = ts.time_step_spec(obs_spec)
    reward_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=optimizer,
        epsilon=0.1,
        accepts_per_arm_features=True)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tf.cast(
                tf.reshape(tf.range(18), shape=[2, 3, 3]), dtype=tf.float32)
    }
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions.shape, (2,))


if __name__ == '__main__':
  tf.test.main()
