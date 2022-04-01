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

"""Tests for tf_agents.agents.qtopt.qtopt_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.qtopt import qtopt_agent
from tf_agents.networks import network
from tf_agents.policies.samplers import qtopt_cem_actions_sampler_continuous
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import test_utils as trajectories_test_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


class DummyNet(network.Network):

  def __init__(self, input_spec, name=None, bias=2):
    super(DummyNet, self).__init__(
        input_spec, state_spec=(), name=name)

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.constant_initializer([2, 1]),
            bias_initializer=tf.constant_initializer([bias]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    for layer in self._dummy_layers:
      inputs = inputs[0]
      inputs = layer(inputs)
    return tf.reshape(inputs, [-1]), network_state


class QtoptAgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(QtoptAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._observation_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._observation_spec)
    self._action_spec = (
        tensor_spec.BoundedTensorSpec([1], tf.float32, 0.0, 1.0))

    # Initiate random mean and var.
    self._num_samples = 32
    action_size = 1
    np.random.seed(1999)
    samples = np.random.rand(self._num_samples,
                             action_size).astype(np.float32)  # [N, a]
    self._mean = np.mean(samples, axis=0)
    self._var = np.var(samples, axis=0)
    self._sampler = qtopt_cem_actions_sampler_continuous.GaussianActionsSampler(
        action_spec=self._action_spec)

  def testCreateAgent(self):
    q_net = critic_network.CriticNetwork(
        (self._observation_spec, self._action_spec))
    agent = qtopt_agent.QtOptAgent(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        init_mean_cem=self._mean,
        init_var_cem=self._var,
        num_samples_cem=self._num_samples,
        actions_sampler=self._sampler)
    self.assertIsNotNone(agent.policy)

  def testInitializeAgent(self):
    q_net = critic_network.CriticNetwork(
        (self._observation_spec, self._action_spec))
    agent = qtopt_agent.QtOptAgent(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        init_mean_cem=self._mean,
        init_var_cem=self._var,
        num_samples_cem=self._num_samples,
        actions_sampler=self._sampler)
    agent.initialize()

  def testPolicy(self):
    q_net = critic_network.CriticNetwork(
        (self._observation_spec, self._action_spec))
    agent = qtopt_agent.QtOptAgent(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        init_mean_cem=self._mean,
        init_var_cem=self._var,
        num_samples_cem=self._num_samples,
        actions_sampler=self._sampler)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual(
        [2] + self._action_spec.shape.as_list(),
        action_step.action.shape,
    )
    self.evaluate(tf.compat.v1.initialize_all_variables())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(all(actions_ <= self._action_spec.maximum))
    self.assertTrue(all(actions_ >= self._action_spec.minimum))

  def testLoss(self):
    q_net = DummyNet((self._observation_spec, self._action_spec))
    agent = qtopt_agent.QtOptAgent(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None,
        init_mean_cem=self._mean,
        init_var_cem=self._var,
        num_samples_cem=self._num_samples,
        actions_sampler=self._sampler)

    agent._target_q_network_delayed = DummyNet(
        (self._observation_spec, self._action_spec), bias=1)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    actions = tf.constant([[0.0], [0.0]], dtype=tf.float32)
    action_steps = policy_step.PolicyStep(
        actions, info=())

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    experience = trajectories_test_utils.stacked_trajectory_from_transition(
        time_steps, action_steps, next_time_steps)

    # Using the kernel initializer [[2, 1], [1, 1]] and bias initializer
    # ([[2], [2]] for q_network/target_network, [[1], [1]] for delayed
    # target_network)
    # from DummyNet above, we can calculate the following values:
    # Q Network:
    # Q-value for first observation/action pair: 2 * 1 + 1 * 2 + 2 = 6
    # Q-value for second observation/action pair: 2 * 3 + 1 * 4 + 2 = 12
    # Target Network:
    # Q-value for first next_observation: 2 * 5 + 1 * 6 + 2 = 18
    # Q-value for second next_observation: 2 * 7 + 1 * 8 + 2 = 24
    # Delayed Target Network:
    # Q-value for first next_observation: 2 * 5 + 1 * 6 + 1 = 17
    # Q-value for second next_observation: 2 * 7 + 1 * 8 + 1 = 23
    # TD targets: 10 + 0.9 * min(17, 18) = 25.3; 20 + 0.9 * min(23, 24) = 40.7
    # TD errors: 25.3 - 6 = 19.3; 40.7 - 12 = 28.7
    # TD loss: 18.8 and 28.2 (Huber loss subtracts 0.5)
    # Overall loss: (18.8 + 28.2) / 2 = 23.5
    expected_td_loss = 23.5
    loss, loss_info = agent._loss(experience)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss), expected_td_loss)
    self.assertAllClose(self.evaluate(tf.reduce_mean(loss_info.td_loss)),
                        expected_td_loss)

  def VerifyVariableAssignAndRestore(self):
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
      # Use BehaviorCloningAgent instead of AWRAgent to test the network.
      q_net = critic_network.CriticNetwork(
          (self._observation_spec, self._action_spec))
      agent = qtopt_agent.QtOptAgent(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None,
          init_mean_cem=self._mean,
          init_var_cem=self._var,
          num_samples_cem=self._num_samples,
          actions_sampler=self._sampler)
    # Assign all vars to 0.
    for var in tf.nest.flatten(agent.variables):
      var.assign(tf.zeros_like(var))
    # Save checkpoint
    ckpt_dir = self.create_tempdir()
    checkpointer = common.Checkpointer(
        ckpt_dir=ckpt_dir, agent=agent)
    global_step = tf.constant(0)
    checkpointer.save(global_step)
    # Assign all vars to 1.
    for var in tf.nest.flatten(agent.variables):
      var.assign(tf.ones_like(var))
    # Restore to 0.
    checkpointer._checkpoint.restore(checkpointer._manager.latest_checkpoint)
    for var in tf.nest.flatten(agent.variables):
      value = var.numpy()
      if isinstance(value, np.int64):
        self.assertEqual(value, 0)
      else:
        self.assertAllEqual(
            value, np.zeros_like(value),
            msg='{} has var mean {}, expected 0.'.format(var.name, value))

  def VerifyTrainAndRestore(self):
    """Helper function for testing correct variable updating and restoring."""
    batch_size = 2
    seq_len = 2
    observations = tensor_spec.sample_spec_nest(
        self._observation_spec, outer_dims=(batch_size, seq_len))
    actions = tensor_spec.sample_spec_nest(
        self._action_spec, outer_dims=(batch_size, seq_len))
    rewards = tf.constant([[10, 10], [20, 20]], dtype=tf.float32)
    discounts = tf.constant([[0.9, 0.9], [0.9, 0.9]], dtype=tf.float32)
    experience = trajectory.first(
        observation=observations,
        action=actions,
        policy_info=(),
        reward=rewards,
        discount=discounts)
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
      q_net = critic_network.CriticNetwork(
          (self._observation_spec, self._action_spec))
      agent = qtopt_agent.QtOptAgent(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
          init_mean_cem=self._mean,
          init_var_cem=self._var,
          num_samples_cem=self._num_samples,
          actions_sampler=self._sampler,
          in_graph_bellman_update=True)
    loss_before_train = agent.loss(experience).loss
    # Check loss is stable.
    self.assertEqual(loss_before_train, agent.loss(experience).loss)
    # Train 1 step, verify that loss is decreased for the same input.
    agent.train(experience)
    loss_after_train = agent.loss(experience).loss
    self.assertLessEqual(loss_after_train, loss_before_train)
    # Assert loss evaluation is still stable, e.g. deterministic.
    self.assertLessEqual(loss_after_train, agent.loss(experience).loss)
    # Save checkpoint
    ckpt_dir = self.create_tempdir()
    checkpointer = common.Checkpointer(ckpt_dir=ckpt_dir, agent=agent)
    global_step = tf.constant(1)
    checkpointer.save(global_step)
    # Assign all vars to 0.
    for var in tf.nest.flatten(agent.variables):
      var.assign(tf.zeros_like(var))
    loss_after_zero = agent.loss(experience).loss
    self.assertEqual(loss_after_zero, agent.loss(experience).loss)
    self.assertNotEqual(loss_after_zero, loss_after_train)
    # Restore
    checkpointer._checkpoint.restore(checkpointer._manager.latest_checkpoint)
    loss_after_restore = agent.loss(experience).loss
    self.assertNotEqual(loss_after_restore, loss_after_zero)
    self.assertEqual(loss_after_restore, loss_after_train)

  def testAssignAndRestore(self):
    self.VerifyVariableAssignAndRestore()
    self.VerifyTrainAndRestore()


if __name__ == '__main__':
  tf.test.main()
