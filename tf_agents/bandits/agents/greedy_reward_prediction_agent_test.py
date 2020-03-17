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

"""Tests for greedy_reward_prediction_agent.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.agents import greedy_reward_prediction_agent as greedy_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.policies import policy_utilities
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from tensorflow.python.framework import errors  # pylint:disable=g-direct-tensorflow-import  # TF internal
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


class DummyNet(network.Network):

  def __init__(self, unused_observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(
        unused_observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant(
                [[1, 1.5, 2],
                 [1, 1.5, 4]]),
            bias_initializer=tf.compat.v1.initializers.constant(
                [[1], [1], [-10]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


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


def _get_initial_and_final_steps_with_action_mask(observations, rewards):
  batch_size = observations[0].shape[0]
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (tf.constant(observations[0]), tf.constant(observations[1])))
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(rewards, dtype=tf.float32, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (tf.constant(observations[0] + 100.0), tf.constant(observations[1])))
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
    self._observation_spec = self._time_step_spec.observation

  def testCreateAgent(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=None)
    self.assertIsNotNone(agent.policy)

  def testInitializeAgent(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=None)
    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

  def testLoss(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    actions = tf.constant([0, 1], dtype=tf.int32)
    rewards = tf.constant([0.5, 3.0], dtype=tf.float32)

    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=None)
    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))
    loss, _ = agent.loss(observations,
                         actions,
                         rewards)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss), 42.25)

  def testPolicy(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual([2], action_step.action.shape)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [1, 2])

  def testInitializeRestoreAgent(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=None)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.initialize_all_variables())

    checkpoint = tf.train.Checkpoint(agent=agent)

    latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)

    if tf.executing_eagerly():
      self.evaluate(checkpoint_load_status.initialize_or_restore())
      self.assertAllEqual(self.evaluate(action_step.action), [1, 2])
    else:
      with self.cached_session() as sess:
        checkpoint_load_status.initialize_or_restore(sess)
        self.assertAllEqual(sess.run(action_step.action), [1, 2])

  def testTrainAgent(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=optimizer)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss_before), 42.25)
    self.assertAllClose(self.evaluate(loss_after), 93.46)

  def testTrainAgentWithMask(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    time_step_spec = ts.time_step_spec((tensor_spec.TensorSpec([2], tf.float32),
                                        tensor_spec.TensorSpec([3], tf.int32)))
    agent = greedy_agent.GreedyRewardPredictionAgent(
        time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]))
    observations = (np.array([[1, 2], [3, 4]], dtype=np.float32),
                    np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int32))
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss_before), 42.25)
    self.assertAllClose(self.evaluate(loss_after), 93.46)

  def testTrainAgentWithLaplacianSmoothing(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    laplacian_matrix = tf.constant([[1.0, -1.0, 0.0],
                                    [-1.0, 2.0, -1.0],
                                    [0.0, -1.0, 1.0]])
    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=optimizer,
        laplacian_matrix=laplacian_matrix,
        laplacian_smoothing_weight=1.0)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    # The Laplacian smoothing term ends up adding 22.5 to the loss.
    self.assertAllClose(self.evaluate(loss_before), 42.25 + 22.5)

  def testTrainAgentWithLaplacianSmoothingInvalidMatrix(self):
    if tf.executing_eagerly:
      return

    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)

    with self.assertRaisesRegexp(errors.InvalidArgumentError, ''):
      reward_net = DummyNet(self._observation_spec, self._action_spec)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
      # Set the Laplacian matrix to be the identity, which is not a valid
      # Laplacian.
      laplacian_matrix = tf.eye(3)
      agent = greedy_agent.GreedyRewardPredictionAgent(
          self._time_step_spec,
          self._action_spec,
          reward_network=reward_net,
          optimizer=optimizer,
          laplacian_matrix=laplacian_matrix,
          laplacian_smoothing_weight=1.0)
      self.evaluate(tf.compat.v1.initialize_all_variables())
      loss_before, _ = agent.train(experience, None)
      self.evaluate(loss_before)


if __name__ == '__main__':
  tf.test.main()
