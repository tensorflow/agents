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
from tf_agents.agents.dqn import q_network
from tf_agents.agents.dqn import test_utils
from tf_agents.environments import time_step as ts
from tf_agents.specs import tensor_spec


class ComputeTDTargetsTest(tf.test.TestCase):

  def testComputeTDTargets(self):
    next_q_values = tf.constant([10, 20], dtype=tf.float32)
    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)

    expected_td_targets = [19., 38.]
    td_targets = dqn_agent.compute_td_targets(next_q_values, rewards, discounts)
    self.assertAllClose(self.evaluate(td_targets), expected_td_targets)


@parameterized.named_parameters(('.DqnAgent', dqn_agent.DqnAgent),
                                ('.DdqnAgent', dqn_agent.DdqnAgent))
class AgentTest(tf.test.TestCase):

  def setUp(self):
    super(AgentTest, self).setUp()
    obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(obs_spec)
    self._action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)]
    self._observation_spec = self._time_step_spec.observation

  def testCreateAgent(self, agent_class):
    q_net = test_utils.DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    self.assertIsNotNone(agent.policy)

  def testAgentFollowsActionSpec(self, agent_class):
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_network.QNetwork(self._observation_spec, self._action_spec),
        optimizer=None)
    self.assertIsNotNone(agent.policy)
    policy = agent.policy
    observation = tensor_spec.sample_spec_nest(
        self._time_step_spec, seed=42, outer_dims=(1,))
    action_op = policy.action(observation).action
    self.evaluate(tf.compat.v1.initialize_all_variables())

    action = self.evaluate(action_op)
    self.assertEqual([1] + self._action_spec[0].shape.as_list(),
                     list(action[0].shape))

  def testAgentFollowsActionSpecWithScalarAction(self, agent_class):
    action_spec = [tensor_spec.BoundedTensorSpec((), tf.int32, 0, 1)]
    agent = agent_class(
        self._time_step_spec,
        action_spec,
        q_network=q_network.QNetwork(self._observation_spec, action_spec),
        optimizer=None)
    self.assertIsNotNone(agent.policy)
    policy = agent.policy
    observation = tensor_spec.sample_spec_nest(
        self._time_step_spec, seed=42, outer_dims=(1,))

    action_op = policy.action(observation).action
    self.evaluate(tf.compat.v1.initialize_all_variables())
    action = self.evaluate(action_op)
    self.assertEqual([1] + action_spec[0].shape.as_list(),
                     list(action[0].shape))

  def testCreateAgentNestSizeChecks(self, agent_class):
    action_spec = [
        tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    ]

    q_net = test_utils.DummyNet(self._observation_spec, action_spec)
    with self.assertRaisesRegexp(ValueError, '.*one dimensional.*'):
      agent_class(
          self._time_step_spec, action_spec, q_network=q_net, optimizer=None)

  def testCreateAgentDimChecks(self, agent_class):
    action_spec = [tensor_spec.BoundedTensorSpec([1, 2], tf.int32, 0, 1)]
    q_net = test_utils.DummyNet(self._observation_spec, action_spec)
    with self.assertRaisesRegexp(ValueError, '.*one dimensional.*'):
      agent_class(
          self._time_step_spec, action_spec, q_network=q_net, optimizer=None)

  # TODO(b/123890005): Add a test where the target network has different values.
  def testLoss(self, agent_class):
    q_net = test_utils.DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    actions = [tf.constant([[0], [1]], dtype=tf.int32)]

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    expected_loss = 26.0
    loss_info = agent._loss(time_steps, actions, next_time_steps)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    total_loss, _ = self.evaluate(loss_info)

    self.assertAllClose(total_loss, expected_loss)

  def testPolicy(self, agent_class):
    q_net = test_utils.DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
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

  def testInitializeRestoreAgent(self, agent_class):
    q_net = test_utils.DummyNet(self._observation_spec, self._action_spec)
    agent = agent_class(
        self._time_step_spec,
        self._action_spec,
        q_network=q_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.initialize_all_variables())

    checkpoint = tf.train.Checkpoint(agent=agent)

    latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)

    with self.cached_session() as sess:
      checkpoint_load_status.initialize_or_restore(sess)
      self.assertAllEqual(sess.run(action_step.action), [[[0], [0]]])


if __name__ == '__main__':
  tf.test.main()
