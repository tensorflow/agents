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

"""Tests for tf_agents.agents.td3.td3_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.td3 import td3_agent
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
               shared_layer=None,
               name=None):
    super(DummyActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    self._unbounded_actions = unbounded_actions
    activation = None if unbounded_actions else tf.keras.activations.tanh

    self._output_tensor_spec = output_tensor_spec
    self._single_action_spec = tf.nest.flatten(output_tensor_spec)[0]
    self._layer = tf.keras.layers.Dense(
        self._single_action_spec.shape.num_elements(),
        activation=activation,
        kernel_initializer=tf.constant_initializer([2, 1]),
        bias_initializer=tf.constant_initializer([5]),
        name='action')
    self._shared_layer = shared_layer

  def call(self, observations, step_type=(), network_state=()):
    del step_type  # unused.
    observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    if self._shared_layer:
      observations = self._shared_layer(observations)
    output = self._layer(observations)
    actions = tf.reshape(output,
                         [-1] + self._single_action_spec.shape.as_list())

    if not self._unbounded_actions:
      actions = common.scale_to_spec(actions, self._single_action_spec)

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec,
                                              [actions])
    return output_actions, network_state


class DummyCriticNetwork(network.Network):

  def __init__(self, input_tensor_spec, shared_layer=None, name=None):
    super(DummyCriticNetwork, self).__init__(
        input_tensor_spec, state_spec=(), name=name)

    self._obs_layer = tf.keras.layers.Flatten()
    self._shared_layer = shared_layer
    self._action_layer = tf.keras.layers.Flatten()
    self._joint_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.constant_initializer([1, 3, 2]),
        bias_initializer=tf.constant_initializer([4]))

  def call(self, inputs, step_type=None, network_state=()):
    observations, actions = inputs
    del step_type
    observations = self._obs_layer(tf.nest.flatten(observations)[0])
    if self._shared_layer:
      observations = self._shared_layer(observations)
    actions = self._action_layer(tf.nest.flatten(actions)[0])
    joint = tf.concat([observations, actions], 1)
    q_value = self._joint_layer(joint)
    q_value = tf.reshape(q_value, [-1])
    return q_value, network_state


class TD3AgentTest(test_utils.TestCase):

  def setUp(self):
    super(TD3AgentTest, self).setUp()
    self._obs_spec = [
        tensor_spec.BoundedTensorSpec([2], tf.float32, minimum=0, maximum=1)
    ]
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)]

    input_spec = (self._obs_spec, self._action_spec)
    self._critic_net = DummyCriticNetwork(input_spec)
    self._bounded_actor_net = DummyActorNetwork(
        self._obs_spec, self._action_spec, unbounded_actions=False)
    self._unbounded_actor_net = DummyActorNetwork(
        self._obs_spec, self._action_spec, unbounded_actions=True)

  def testCreateAgent(self):
    td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._bounded_actor_net,
        actor_optimizer=None,
        critic_optimizer=None,
        )

  def testAgentTrajectoryTrain(self):
    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._bounded_actor_net,
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
    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._bounded_actor_net,
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

  def testCriticLoss(self):
    # The loss is now 119.3098526. Investigate this.
    self.skipTest('b/123772477')
    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._unbounded_actor_net,
        actor_optimizer=None,
        critic_optimizer=None)

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)
    actions = [tf.constant([[5], [6]], dtype=tf.float32)]

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    # TODO(b/123772477): The loss changed from 119.054 to 118.910903931.
    expected_loss = 118.9109
    loss = agent.critic_loss(time_steps, actions, next_time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testActorLoss(self):
    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._unbounded_actor_net,
        actor_optimizer=None,
        critic_optimizer=None)

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)

    actions = [2 * 1 + 1 * 2 + 5, 2 * 3 + 1 * 4 + 5]
    negative_q_values = [
        -(1 * 1 + 3 * 2 + 2 * actions[0] + 4),
        -(1 * 3 + 3 * 4 + 2 * actions[1] + 4)
    ]
    expected_loss = np.mean(negative_q_values)
    loss = agent.actor_loss(time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testBatchedActorLoss(self):

    class FakeCriticNetwork(network.Network):
      """Fake critic network with random output Q value of a specified shape."""

      def __init__(self, input_tensor_spec, output_shape, name=None):
        self._output_shape = output_shape
        super(FakeCriticNetwork, self).__init__(
            input_tensor_spec, state_spec=(), name=name)

      def call(self, inputs, step_type=None, network_state=()):
        q_value = tf.random.uniform(
            self._output_shape, minval=0, maxval=1, dtype=tf.dtypes.float32,
        )
        return q_value, network_state

    critic_input_spec = (self._obs_spec, self._action_spec)
    critic_net = FakeCriticNetwork(critic_input_spec, output_shape=(2, 1),)
    obs_spec = [
        tensor_spec.BoundedTensorSpec([1], tf.float32, minimum=0, maximum=1)
    ]
    time_step_spec = ts.time_step_spec(obs_spec)
    agent = td3_agent.Td3Agent(
        time_step_spec,
        self._action_spec,
        critic_network=critic_net,
        actor_network=self._unbounded_actor_net,
        actor_optimizer=None,
        critic_optimizer=None)

    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=2)

    loss = agent.actor_loss(time_steps)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # Ensures that the actor loss calculation does not err out when the critic
    # output has rank > 1.
    self.evaluate(loss)

  def testPolicyProducesBoundedAction(self):
    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._bounded_actor_net,
        actor_optimizer=None,
        critic_optimizer=None)

    observations = [tf.constant([[1, 2]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=1)
    action = agent.policy.action(time_steps).action[0]
    self.assertEqual(action.shape.as_list(), [1, 1])

    self.evaluate([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()
    ])
    py_action = self.evaluate(action)
    self.assertTrue(all(py_action <= self._action_spec[0].maximum))
    self.assertTrue(all(py_action >= self._action_spec[0].minimum))

  def testPolicyAndCollectPolicyProducesDifferentActions(self):
    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        critic_network=self._critic_net,
        actor_network=self._bounded_actor_net,
        actor_optimizer=None,
        critic_optimizer=None)

    observations = [tf.constant([[1, 2]], dtype=tf.float32)]
    time_steps = ts.restart(observations, batch_size=1)
    action = agent.policy.action(time_steps).action[0]
    collect_policy_action = agent.collect_policy.action(time_steps).action[0]
    self.assertEqual(action.shape, collect_policy_action.shape)

    self.evaluate([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()
    ])
    py_action, py_collect_policy_action = self.evaluate(
        [action, collect_policy_action])
    self.assertNotEqual(py_action, py_collect_policy_action)

  def testSharedLayer(self):
    input_spec = (self._obs_spec, self._action_spec)

    shared_layer = tf.keras.layers.Dense(
        2,
        kernel_initializer=tf.constant_initializer([0]),
        bias_initializer=tf.constant_initializer([0]),
        name='shared')

    critic_net_1 = DummyCriticNetwork(input_spec, shared_layer=shared_layer)
    critic_net_2 = DummyCriticNetwork(input_spec, shared_layer=shared_layer)

    bounded_actor_net = DummyActorNetwork(
        self._obs_spec,
        self._action_spec,
        shared_layer=shared_layer,
        unbounded_actions=False)

    target_shared_layer = tf.keras.layers.Dense(
        2,
        kernel_initializer=tf.constant_initializer([0]),
        bias_initializer=tf.constant_initializer([0]),
        name='shared')

    target_critic_net_1 = DummyCriticNetwork(
        input_spec, shared_layer=target_shared_layer)
    target_critic_net_2 = DummyCriticNetwork(
        input_spec, shared_layer=target_shared_layer)
    target_bounded_actor_net = DummyActorNetwork(
        self._obs_spec,
        self._action_spec,
        shared_layer=target_shared_layer,
        unbounded_actions=False)

    agent = td3_agent.Td3Agent(
        self._time_step_spec,
        self._action_spec,
        actor_network=bounded_actor_net,
        critic_network=critic_net_1,
        critic_network_2=critic_net_2,
        target_actor_network=target_bounded_actor_net,
        target_critic_network=target_critic_net_1,
        target_critic_network_2=target_critic_net_2,
        actor_optimizer=None,
        critic_optimizer=None,
        target_update_tau=0.5)

    self.evaluate([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()
    ])

    self.evaluate(agent.initialize())

    for v in shared_layer.variables:
      self.evaluate(v.assign(v * 0 + 1))

    self.evaluate(agent._update_target())

    self.assertEqual(1.0, self.evaluate(shared_layer.variables[0][0][0]))
    self.assertEqual(0.5, self.evaluate(target_shared_layer.variables[0][0][0]))


if __name__ == '__main__':
  tf.test.main()
