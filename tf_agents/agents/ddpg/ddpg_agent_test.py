# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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


"""Tests for tf_agents.agents.ddpg.ddpg_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.environments import time_step as ts
from tf_agents.specs import tensor_spec

slim = tf.contrib.slim
nest = tf.contrib.framework.nest


def get_dummy_actor_net(unbounded_actions=False):

  # When unbounded_actions=True, we skip the final tanh activation and the
  # action shift and scale. This allows us to compute the actor and critic
  # losses by hand more easily.

  def actor_net(time_steps, action_spec):
    with slim.arg_scope(
        [slim.fully_connected],
        activation_fn=None if unbounded_actions else tf.nn.tanh):

      single_action_spec = nest.flatten(action_spec)[0]
      states = tf.cast(nest.flatten(time_steps.observation)[0], tf.float32)
      actions = slim.fully_connected(
          states,
          single_action_spec.shape.num_elements(),
          scope='actions',
          weights_initializer=tf.constant_initializer([2, 1]),
          biases_initializer=tf.constant_initializer([5]),
          normalizer_fn=None)
      actions = tf.reshape(actions, [-1] + single_action_spec.shape.as_list())
      if not unbounded_actions:
        action_means = (
            single_action_spec.maximum + single_action_spec.minimum) / 2.0
        action_magnitudes = (
            single_action_spec.maximum - single_action_spec.minimum) / 2.0
        actions = action_means + action_magnitudes * actions
      actions = nest.pack_sequence_as(action_spec, [actions])
    return actions

  return actor_net


def _dummy_critic_net(time_steps, actions):
  states = slim.flatten(nest.flatten(time_steps.observation)[0])
  actions = slim.flatten(nest.flatten(actions)[0])

  joint = tf.concat([states, actions], 1)
  q_value = slim.fully_connected(
      joint,
      1,
      activation_fn=None,
      normalizer_fn=None,
      weights_initializer=tf.constant_initializer([1, 3, 2]),
      biases_initializer=tf.constant_initializer([4]),
      scope='q_value')
  return tf.reshape(q_value, [-1])


class DdpgAgentTest(tf.test.TestCase):

  def setUp(self):
    super(DdpgAgentTest, self).setUp()
    self._obs_spec = [tensor_spec.TensorSpec([2], tf.float32)]
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)]

  def testCreateAgent(self):
    ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        critic_net=_dummy_critic_net,
        actor_net=get_dummy_actor_net(unbounded_actions=False)
    )

  def testCreateAgentDefaultNetwork(self):
    ddpg_agent.DdpgAgent(self._time_step_spec, self._action_spec)

  def testCriticLoss(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        critic_net=_dummy_critic_net,
        actor_net=get_dummy_actor_net(unbounded_actions=True)
    )

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations)
    actions = [tf.constant([[5], [6]], dtype=tf.float32)]

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    expected_loss = 59.6
    loss = agent.critic_loss(time_steps, actions, next_time_steps)

    self.evaluate(tf.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testActorLoss(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        critic_net=_dummy_critic_net,
        actor_net=get_dummy_actor_net(unbounded_actions=True)
    )

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)

    expected_loss = 4.0
    loss = agent.actor_loss(time_steps)

    self.evaluate(tf.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testPolicy(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        critic_net=_dummy_critic_net,
        actor_net=get_dummy_actor_net(unbounded_actions=False)
    )

    observations = [tf.constant([1, 2], dtype=tf.float32)]
    time_steps = ts.restart(observations)
    action_step = agent.policy().action(time_steps)
    self.assertEqual(action_step.action[0].shape.as_list(), [1])

    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(all(actions_[0] <= self._action_spec[0].maximum))
    self.assertTrue(all(actions_[0] >= self._action_spec[0].minimum))


if __name__ == '__main__':
  tf.test.main()
