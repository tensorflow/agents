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

import tensorflow as tf

from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


class DummyNet(network.Network):

  def __init__(self, observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self._layers.append(
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant(
                [[1, 1.5, 2],
                 [1, 1.5, 4]]),
            bias_initializer=tf.compat.v1.initializers.constant(
                [[1], [1], [-10]])))

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs, network_state


@test_util.run_all_in_graph_and_eager_modes
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


if __name__ == '__main__':
  tf.test.main()
