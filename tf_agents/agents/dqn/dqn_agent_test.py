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

"""Tests for agents.dqn.dqn_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import test_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tensorflow.python.eager import context  # pylint:disable=g-direct-tensorflow-import  # TF internal
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


class DummyNet(network.Network):

  def __init__(self, unused_observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(
        unused_observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self._layers.append(
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant([[2, 1],
                                                                   [1, 1]]),
            bias_initializer=tf.compat.v1.initializers.constant([[1], [1]])))

  def call(self, inputs, unused_step_type=None, network_state=()):
    inputs = tf.cast(inputs[0], tf.float32)
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs, network_state


class ComputeTDTargetsTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testComputeTDTargets(self):
    next_q_values = tf.constant([10, 20], dtype=tf.float32)
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    expected_td_targets = [19., 38.]
    td_targets = dqn_agent.compute_td_targets(next_q_values, rewards, discounts)
    self.assertAllClose(self.evaluate(td_targets), expected_td_targets)


@parameterized.named_parameters(
    ('DqnAgent_graph', dqn_agent.DqnAgent, context.graph_mode),
    ('DqnAgent_eager', dqn_agent.DqnAgent, context.eager_mode),
    ('DdqnAgent_graph', dqn_agent.DdqnAgent, context.graph_mode),
    ('DdqnAgent_eager', dqn_agent.DdqnAgent, context.eager_mode))
class AgentTest(tf.test.TestCase):

  def setUp(self):
    super(AgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._observation_spec = [tensor_spec.TensorSpec([2], tf.float32)]
    self._time_step_spec = ts.time_step_spec(self._observation_spec)
    self._action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)]

  def testCreateAgent(self, agent_class, run_mode):
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None)
      self.assertIsNotNone(agent.policy)

  def testInitializeAgent(self, agent_class, run_mode):
    if tf.executing_eagerly() and run_mode == context.graph_mode:
      self.skipTest('b/123778560')
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None)
      init_op = agent.initialize()
      if not tf.executing_eagerly():
        with self.cached_session() as sess:
          common.initialize_uninitialized_variables(sess)
          self.assertIsNone(sess.run(init_op))

  def testCreateAgentNestSizeChecks(self, agent_class, run_mode):
    with run_mode():
      action_spec = [
          tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
          tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
      ]

      q_net = DummyNet(self._observation_spec, action_spec)
      with self.assertRaisesRegexp(ValueError, '.*one dimensional.*'):
        agent_class(
            self._time_step_spec, action_spec, q_network=q_net, optimizer=None)

  def testCreateAgentDimChecks(self, agent_class, run_mode):
    with run_mode():
      action_spec = [tensor_spec.BoundedTensorSpec([1, 2], tf.int32, 0, 1)]
      q_net = DummyNet(self._observation_spec, action_spec)
      with self.assertRaisesRegexp(ValueError, '.*one dimensional.*'):
        agent_class(
            self._time_step_spec, action_spec, q_network=q_net, optimizer=None)

  # TODO(b/127383724): Add a test where the target network has different values.
  def testLoss(self, agent_class, run_mode):
    if tf.executing_eagerly() and run_mode == context.graph_mode:
      self.skipTest('b/123778560')
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None)

      observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
      time_steps = ts.restart(observations, batch_size=2)

      actions = [tf.constant([[0], [1]], dtype=tf.int32)]
      action_steps = policy_step.PolicyStep(actions)

      rewards = tf.constant([10, 20], dtype=tf.float32)
      discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
      next_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
      next_time_steps = ts.transition(next_observations, rewards, discounts)

      experience = test_utils.stacked_trajectory_from_transition(
          time_steps, action_steps, next_time_steps)

      # Using the kernel initializer [[2, 1], [1, 1]] and bias initializer
      # [[1], [1]] from DummyNet above, we can calculate the following values:
      # Q-value for first observation/action pair: 2 * 1 + 1 * 2 + 1 = 5
      # Q-value for second observation/action pair: 1 * 3 + 1 * 4 + 1 = 8
      # (Here we use the second row of the kernel initializer above, since the
      # chosen action is now 1 instead of 0.)
      # Q-value for first next_observation: 2 * 5 + 1 * 6 + 1 = 17
      # Q-value for second next_observation: 2 * 7 + 1 * 8 + 1 = 23
      # TD targets: 10 + 0.9 * 17 = 25.3 and 20 + 0.9 * 23 = 40.7
      # TD errors: 25.3 - 5 = 20.3 and 40.7 - 8 = 32.7
      # TD loss: 19.8 and 32.2 (Huber loss subtracts 0.5)
      # Overall loss: (19.8 + 32.2) / 2 = 26
      expected_loss = 26.0
      loss, _ = agent._loss(experience)

      self.evaluate(tf.compat.v1.initialize_all_variables())
      self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossNStep(self, agent_class, run_mode):
    if tf.executing_eagerly() and run_mode == context.graph_mode:
      self.skipTest('b/123778560')
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None,
          n_step_update=2)

      observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
      time_steps = ts.restart(observations, batch_size=2)

      actions = [tf.constant([[0], [1]], dtype=tf.int32)]
      action_steps = policy_step.PolicyStep(actions)

      rewards = tf.constant([10, 20], dtype=tf.float32)
      discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
      next_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
      next_time_steps = ts.transition(next_observations, rewards, discounts)

      third_observations = [tf.constant([[9, 10], [11, 12]], dtype=tf.float32)]
      third_time_steps = ts.transition(third_observations, rewards, discounts)

      experience1 = trajectory.from_transition(
          time_steps, action_steps, next_time_steps)
      experience2 = trajectory.from_transition(
          next_time_steps, action_steps, third_time_steps)
      experience3 = trajectory.from_transition(
          third_time_steps, action_steps, third_time_steps)

      experience = tf.nest.map_structure(
          lambda x, y, z: tf.stack([x, y, z], axis=1),
          experience1, experience2, experience3)

      # We can extend the analysis from testLoss above as follows:
      # Original Q-values are still 5 and 8 for the same reasons.
      # Q-value for first third_observation: 2 * 9 + 1 * 10 + 1 = 29
      # Q-value for second third_observation: 2 * 11 + 1 * 12 + 1 = 35
      # TD targets: 10 + 0.9 * (10 + 0.9 * 29) = 42.49
      # 20 + 0.9 * (20 + 0.9 * 35) = 66.35
      # TD errors: 42.49 - 5 = 37.49 and 66.35 - 8 = 58.35
      # TD loss: 36.99 and 57.85 (Huber loss subtracts 0.5)
      # Overall loss: (36.99 + 57.85) / 2 = 47.42
      expected_loss = 47.42
      loss, _ = agent._loss(experience)

      self.evaluate(tf.compat.v1.initialize_all_variables())
      self.assertAllClose(self.evaluate(loss), expected_loss)

  def testLossNStepMidMidLastFirst(self, agent_class, run_mode):
    """Tests that n-step loss handles LAST time steps properly."""
    if tf.executing_eagerly() and run_mode == context.graph_mode:
      self.skipTest('b/123778560')
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None,
          n_step_update=2)

      observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
      rewards = tf.constant([10, 20], dtype=tf.float32)
      discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
      # MID: use ts.transition
      time_steps = ts.transition(observations, rewards, discounts)

      actions = [tf.constant([[0], [1]], dtype=tf.int32)]
      action_steps = policy_step.PolicyStep(actions)

      second_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
      # MID: use ts.transition
      second_time_steps = ts.transition(second_observations, rewards, discounts)

      third_observations = [tf.constant([[9, 10], [11, 12]], dtype=tf.float32)]
      # LAST: use ts.termination
      third_time_steps = ts.termination(third_observations, rewards)

      fourth_observations = [tf.constant([[13, 14], [15, 16]],
                                         dtype=tf.float32)]
      # FIRST: use ts.restart
      fourth_time_steps = ts.restart(fourth_observations, batch_size=2)

      experience1 = trajectory.from_transition(
          time_steps, action_steps, second_time_steps)
      experience2 = trajectory.from_transition(
          second_time_steps, action_steps, third_time_steps)
      experience3 = trajectory.from_transition(
          third_time_steps, action_steps, fourth_time_steps)
      experience4 = trajectory.from_transition(
          fourth_time_steps, action_steps, fourth_time_steps)

      experience = tf.nest.map_structure(
          lambda w, x, y, z: tf.stack([w, x, y, z], axis=1),
          experience1, experience2, experience3, experience4)

      # Once again we can extend the analysis from testLoss above as follows:
      # Original Q-values are still 5 and 8 for the same reasons.
      # However next Q-values are now zeroed out due to the LAST time step in
      # between. Thus the TD targets become the discounted reward sums, or:
      # 10 + 0.9 * 10 = 19 and 20 + 0.9 * 20 = 38
      # TD errors: 19 - 5 = 14 and 38 - 8 = 30
      # TD loss: 13.5 and 29.5 (Huber loss subtracts 0.5)
      # Overall loss: (13.5 + 29.5) / 2 = 21.5
      expected_loss = 21.5
      loss, _ = agent._loss(experience)

      self.evaluate(tf.compat.v1.initialize_all_variables())
      self.assertAllClose(self.evaluate(loss), expected_loss)

  def testPolicy(self, agent_class, run_mode):
    if tf.executing_eagerly() and run_mode == context.graph_mode:
      self.skipTest('b/123778560')
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None)
      observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
      time_steps = ts.restart(observations, batch_size=2)
      policy = agent.policy
      action_step = policy.action(time_steps)
      # Batch size 2.
      self.assertAllEqual(
          [2] + self._action_spec[0].shape.as_list(),
          action_step.action[0].shape,
      )
      self.evaluate(tf.compat.v1.initialize_all_variables())
      actions_ = self.evaluate(action_step.action)
      self.assertTrue(all(actions_[0] <= self._action_spec[0].maximum))
      self.assertTrue(all(actions_[0] >= self._action_spec[0].minimum))

  def testInitializeRestoreAgent(self, agent_class, run_mode):
    if tf.executing_eagerly() and run_mode == context.graph_mode:
      self.skipTest('b/123778560')
    with run_mode():
      q_net = DummyNet(self._observation_spec, self._action_spec)
      agent = agent_class(
          self._time_step_spec,
          self._action_spec,
          q_network=q_net,
          optimizer=None)
      observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
      time_steps = ts.restart(observations, batch_size=2)
      policy = agent.policy
      action_step = policy.action(time_steps)
      self.evaluate(tf.compat.v1.initialize_all_variables())

      checkpoint = tf.train.Checkpoint(agent=agent)

      latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
      checkpoint_load_status = checkpoint.restore(latest_checkpoint)

      if tf.executing_eagerly():
        self.evaluate(checkpoint_load_status.initialize_or_restore())
        self.assertAllEqual(self.evaluate(action_step.action), [[[0], [0]]])
      else:
        with self.cached_session() as sess:
          checkpoint_load_status.initialize_or_restore(sess)
          self.assertAllEqual(sess.run(action_step.action), [[[0], [0]]])


if __name__ == '__main__':
  tf.test.main()
