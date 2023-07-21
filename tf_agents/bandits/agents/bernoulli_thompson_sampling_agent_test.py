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

"""Tests for bernoulli_thompson_sampling_agent.py."""

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import bernoulli_thompson_sampling_agent as bern_ts_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


def _get_initial_and_final_steps(observations, rewards):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  if isinstance(observations, np.ndarray):
    observations = tf.constant(
        observations, dtype=tf.float32, name='observation')
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(rewards, dtype=tf.float32, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  return initial_step, final_step


def _get_initial_and_final_steps_with_action_mask(observations, rewards):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observations[0], observations[1]))
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(rewards, dtype=tf.float32, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size],
                  name='discount'), (tf.nest.map_structure(
                      lambda x: x + 100., observations[0]), observations[1]))
  return initial_step, final_step


def _get_action_step(action):
  return policy_step.PolicyStep(
      action=tf.convert_to_tensor(action),
      info=policy_utilities.PolicyInfo())


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1),
      single_experience)


class AgentTest(tf.test.TestCase):

  def setUp(self):
    super(AgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._num_actions = 3
    self._observation_spec = self._time_step_spec.observation

  def testCreateAgent(self):
    agent = bern_ts_agent.BernoulliThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec)
    self.assertIsNotNone(agent.policy)

  def testInitializeAgent(self):
    agent = bern_ts_agent.BernoulliThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec)
    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

  def testPolicy(self):
    agent = bern_ts_agent.BernoulliThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec,
        batch_size=2)
    observations = tf.constant([[1, 1]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual([2], action_step.action.shape)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)

  def testTrainAgent(self):
    observations = np.array([[1, 1]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.0, 1.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)

    agent = bern_ts_agent.BernoulliThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec,
        batch_size=2)
    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))
    loss, _ = agent._train(experience, weights=None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    # The loss is -sum(rewards).
    self.assertAllClose(self.evaluate(loss), -1.0)

  def testTrainAgentWithMask(self):
    time_step_spec = ts.time_step_spec((tensor_spec.TensorSpec([], tf.float32),
                                        tensor_spec.TensorSpec([3], tf.int32)))
    agent = bern_ts_agent.BernoulliThompsonSamplingAgent(
        time_step_spec,
        self._action_spec,
        batch_size=2)
    observations = (np.array([1, 1], dtype=np.float32),
                    np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32))
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([1.0, 0.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss), -1.0)


if __name__ == '__main__':
  tf.test.main()
