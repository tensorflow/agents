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

"""Tests for dropout_thompson_sampling_agent.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.agents import dropout_thompson_sampling_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.policies import policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts


from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


def _get_initial_and_final_steps(observations, rewards):
  batch_size = observations.shape[0]
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST, dtype=tf.int32, shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(observations, dtype=tf.float32, name='observation'))
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST, dtype=tf.int32, shape=[batch_size],
          name='step_type'),
      tf.constant(rewards, dtype=tf.float32, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(observations + 100.0, dtype=tf.float32, name='observation'))
  return initial_step, final_step


def _get_initial_and_final_steps_with_action_mask(batch_size,
                                                  context_dim,
                                                  num_actions):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
  observation = tf.constant(observation, dtype=tf.float32)
  mask = 1 - tf.eye(batch_size, num_columns=num_actions, dtype=tf.int32)
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observation, mask))
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observation + 100.0, mask))
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


@test_util.run_all_in_graph_and_eager_modes
class AgentTest(tf.test.TestCase):

  def setUp(self):
    super(AgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)

  def testCreateAgent(self):
    agent = dropout_thompson_sampling_agent.DropoutThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec,
        optimizer=None,
        dropout_rate=0.1,
        network_layers=(20, 20, 20))
    self.assertIsNotNone(agent.policy)

  def testTrainAgent(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = dropout_thompson_sampling_agent.DropoutThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec,
        optimizer=optimizer,
        dropout_rate=0.1,
        network_layers=(20, 20, 20),
        dropout_only_top_layer=False)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllGreater(self.evaluate(loss_before), 0)
    self.assertAllGreater(self.evaluate(loss_after), 0)

  def testAgentWithMask(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    obs_spec = (tensor_spec.TensorSpec([2], tf.float32),
                tensor_spec.TensorSpec([3], tf.int32))
    agent = dropout_thompson_sampling_agent.DropoutThompsonSamplingAgent(
        ts.time_step_spec(obs_spec),
        self._action_spec,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]),
        dropout_rate=0.1,
        network_layers=(20, 20, 20),
        dropout_only_top_layer=False)
    actions = np.array([0, 1], dtype=np.int32)
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        2, 2, 3)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllGreater(self.evaluate(loss_before), 0)
    self.assertAllGreater(self.evaluate(loss_after), 0)

  def testTrainAgentHeteroscedastic(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = dropout_thompson_sampling_agent.DropoutThompsonSamplingAgent(
        self._time_step_spec,
        self._action_spec,
        optimizer=optimizer,
        dropout_rate=0.1,
        network_layers=(20, 20, 20),
        dropout_only_top_layer=False,
        heteroscedastic=True)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertIsNotNone(self.evaluate(loss_before))
    self.assertIsNotNone(self.evaluate(loss_after))

  def testAgentWithMaskHeteroscedastic(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    obs_spec = (tensor_spec.TensorSpec([2], tf.float32),
                tensor_spec.TensorSpec([3], tf.int32))
    agent = dropout_thompson_sampling_agent.DropoutThompsonSamplingAgent(
        ts.time_step_spec(obs_spec),
        self._action_spec,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]),
        dropout_rate=0.1,
        network_layers=(20, 20, 20),
        dropout_only_top_layer=False,
        heteroscedastic=True)
    actions = np.array([0, 1], dtype=np.int32)
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        2, 2, 3)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertIsNotNone(self.evaluate(loss_before))
    self.assertIsNotNone(self.evaluate(loss_after))


if __name__ == '__main__':
  tf.test.main()
