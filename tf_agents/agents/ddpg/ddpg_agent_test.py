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

"""Tests for tf_agents.agents.ddpg.ddpg_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import test_utils


class DummyActorNetwork(network.Network):
  """Creates an actor network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               unbounded_actions=False,
               name=None):
    super(DummyActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._output_tensor_spec = output_tensor_spec
    self._unbounded_actions = unbounded_actions
    activation = None if unbounded_actions else tf.keras.activations.tanh

    self._single_action_spec = tf.nest.flatten(output_tensor_spec)[0]
    self._layer = tf.keras.layers.Dense(
        self._single_action_spec.shape.num_elements(),
        activation=activation,
        kernel_initializer=tf.constant_initializer([2, 1]),
        bias_initializer=tf.constant_initializer([5]),
        name='action')

  def call(self, observations, step_type=(), network_state=()):
    del step_type  # unused.
    observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    output = self._layer(observations)
    actions = tf.reshape(output,
                         [-1] + self._single_action_spec.shape.as_list())

    if not self._unbounded_actions:
      actions = common.scale_to_spec(actions, self._single_action_spec)

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                              [actions])
    return output_actions, network_state


class DummyCriticNetwork(network.Network):

  def __init__(self, input_tensor_spec, name=None):
    super(DummyCriticNetwork, self).__init__(
        input_tensor_spec, state_spec=(), name=name)

    self._obs_layer = tf.keras.layers.Flatten()
    self._action_layer = tf.keras.layers.Flatten()
    self._joint_layer = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.constant_initializer([1, 3, 2]),
        bias_initializer=tf.constant_initializer([4]))

  def call(self, inputs, step_type=None, network_state=()):
    observations, actions = inputs
    del step_type
    observations = self._obs_layer(tf.nest.flatten(observations)[0])
    actions = self._action_layer(tf.nest.flatten(actions)[0])
    joint = tf.concat([observations, actions], 1)
    q_value = self._joint_layer(joint)
    q_value = tf.reshape(q_value, [-1])
    return q_value, network_state


class DdpgAgentTest(test_utils.TestCase):

  def setUp(self):
    super(DdpgAgentTest, self).setUp()
    self._obs_spec = [tensor_spec.TensorSpec([2], tf.float32)]
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)]

    network_input_spec = (self._obs_spec, self._action_spec)
    self._critic_net = DummyCriticNetwork(network_input_spec)
    self._bounded_actor_net = DummyActorNetwork(
        self._obs_spec, self._action_spec, unbounded_actions=False)
    self._unbounded_actor_net = DummyActorNetwork(
        self._obs_spec, self._action_spec, unbounded_actions=True)

  def testCreateAgent(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=self._bounded_actor_net,
        critic_network=self._critic_net,
        actor_optimizer=None,
        critic_optimizer=None,
    )
    self.assertIsNotNone(agent.policy)
    self.assertIsNotNone(agent.collect_policy)

  def testCriticLoss(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=self._unbounded_actor_net,
        critic_network=self._critic_net,
        actor_optimizer=None,
        critic_optimizer=None,
    )

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)

    actions = [tf.constant([[5], [6]], dtype=tf.float32)]

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    expected_loss = 59.6
    loss = agent.critic_loss(time_steps, actions, next_time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testActorLoss(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=self._unbounded_actor_net,
        critic_network=self._critic_net,
        actor_optimizer=None,
        critic_optimizer=None,
    )

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)

    expected_loss = 4.0
    loss = agent.actor_loss(time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testPolicy(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=self._unbounded_actor_net,
        critic_network=self._critic_net,
        actor_optimizer=None,
        critic_optimizer=None,
    )

    observations = [tf.constant([[1, 2]], dtype=tf.float32)]
    time_steps = ts.restart(observations)
    action_step = agent.policy.action(time_steps)
    self.assertEqual(action_step.action[0].shape.as_list(), [1, 1])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(all(actions_[0] <= self._action_spec[0].maximum))
    self.assertTrue(all(actions_[0] >= self._action_spec[0].minimum))

  def testAgentTrajectoryTrain(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=self._bounded_actor_net,
        critic_network=self._critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
    )

    trajectory_spec = trajectory.Trajectory(
        step_type=self._time_step_spec.step_type,
        observation=self._time_step_spec.observation,
        action=self._action_spec,
        policy_info=(),
        next_step_type=self._time_step_spec.step_type,
        reward=tensor_spec.BoundedTensorSpec(
            [], tf.float32, minimum=0.0, maximum=1.0, name='reward'),
        discount=self._time_step_spec.discount)

    sample_trajectory_experience = tensor_spec.sample_spec_nest(
        trajectory_spec, outer_dims=(3, 2))
    agent.train(sample_trajectory_experience)

  def testAgentTransitionTrain(self):
    agent = ddpg_agent.DdpgAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=self._bounded_actor_net,
        critic_network=self._critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
    )

    time_step_spec = self._time_step_spec._replace(
        reward=tensor_spec.BoundedTensorSpec(
            [], tf.float32, minimum=0.0, maximum=1.0, name='reward'))

    transition_spec = trajectory.Transition(
        time_step=time_step_spec,
        action_step=policy_step.PolicyStep(action=self._action_spec,
                                           state=(),
                                           info=()),
        next_time_step=time_step_spec)

    sample_trajectory_experience = tensor_spec.sample_spec_nest(
        transition_spec, outer_dims=(3,))
    agent.train(sample_trajectory_experience)


if __name__ == '__main__':
  tf.test.main()
