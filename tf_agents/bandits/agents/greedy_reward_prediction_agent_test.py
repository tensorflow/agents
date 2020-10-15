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
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


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
            kernel_initializer=tf.constant_initializer([[1, 1.5, 2],
                                                        [1, 1.5, 4]]),
            bias_initializer=tf.constant_initializer([[1], [1], [-10]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


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


def _get_initial_and_final_steps_nested_rewards(observations, rewards):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  if isinstance(observations, np.ndarray):
    observations = tf.constant(
        observations, dtype=tf.float32, name='observation')
  zero_rewards = {
      'reward': tf.constant(0.0, dtype=tf.float32, shape=[batch_size]),
      'constraint': tf.constant(0.0, dtype=tf.float32, shape=[batch_size])
  }
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      zero_rewards,
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  rewards_nest = tf.nest.map_structure(
      lambda t: tf.convert_to_tensor(t, dtype=tf.float32), rewards)
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      rewards_nest,
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


def _get_initial_and_final_steps_action_mask_nested_rewards(
    observations, rewards):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  zero_rewards = {
      'reward': tf.constant(0.0, dtype=tf.float32, shape=[batch_size]),
      'constraint': tf.constant(0.0, dtype=tf.float32, shape=[batch_size])
  }
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      zero_rewards,
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observations[0], observations[1]))
  rewards_nest = tf.nest.map_structure(
      lambda t: tf.convert_to_tensor(t, dtype=tf.float32), rewards)
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      rewards_nest,
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

  def testTrainAgentWithConstraint(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    neural_constraint = constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

    reward_spec = {
        'reward': tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name='reward'),
        'constraint': tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name='constraint')
    }
    self._time_step_spec = ts.time_step_spec(self._obs_spec, reward_spec)

    agent = greedy_agent.GreedyRewardPredictionAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=optimizer,
        constraints=[neural_constraint])
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    rewards = {
        'reward': np.array([0.5, 3.0], dtype=np.float32),
        'constraint': np.array([6.0, 4.0], dtype=np.float32)
    }
    initial_step, final_step = _get_initial_and_final_steps_nested_rewards(
        observations, rewards)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    # The loss is the sum of the reward loss and the constraint loss.
    self.assertAllClose(self.evaluate(loss_before), 42.25 + 30.125)

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

  def testTrainAgentWithMaskAndConstraint(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    reward_spec = {
        'reward': tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name='reward'),
        'constraint': tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name='constraint')
    }
    observation_and_mask_spec = (tensor_spec.TensorSpec([2], tf.float32),
                                 tensor_spec.TensorSpec([3], tf.int32))
    time_step_spec = ts.time_step_spec(observation_and_mask_spec, reward_spec)

    constraint_net = DummyNet(self._observation_spec, self._action_spec)
    neural_constraint = constraints.NeuralConstraint(
        self._time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

    agent = greedy_agent.GreedyRewardPredictionAgent(
        time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]),
        constraints=[neural_constraint])
    observations = (np.array([[1, 2], [3, 4]], dtype=np.float32),
                    np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int32))
    actions = np.array([0, 1], dtype=np.int32)
    rewards = {
        'reward': np.array([0.5, 3.0], dtype=np.float32),
        'constraint': np.array([6.0, 4.0], dtype=np.float32)
    }
    initial_step, final_step = (
        _get_initial_and_final_steps_action_mask_nested_rewards(
            observations, rewards))
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    # The loss is the sum of the reward loss and the constraint loss.
    self.assertAllClose(self.evaluate(loss_before), 42.25 + 30.125)

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

    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ''):
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

  def testTrainPerArmAgent(self):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        2, 3, 4, add_num_actions_feature=True)
    time_step_spec = ts.time_step_spec(obs_spec)
    reward_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = greedy_agent.GreedyRewardPredictionAgent(
        time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        accepts_per_arm_features=True,
        optimizer=optimizer)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tf.cast(
                tf.reshape(tf.range(24), shape=[2, 4, 3]), dtype=tf.float32),
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
            tf.ones([2], dtype=tf.int32)
    }
    actions = np.array([0, 3], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, rewards)
    action_step = policy_step.PolicyStep(
        action=tf.convert_to_tensor(actions),
        info=policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=np.array([[1, 2, 3], [3, 2, 1]],
                                         dtype=np.float32)))
    experience = _get_experience(initial_step, action_step, final_step)
    agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())

  def testTrainPerArmAgentWithConstraint(self):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(2, 3, 4)
    reward_spec = {
        'reward': tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name='reward'),
        'constraint': tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name='constraint')
    }
    time_step_spec = ts.time_step_spec(obs_spec, reward_spec)
    reward_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    constraint_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))
    neural_constraint = constraints.NeuralConstraint(
        time_step_spec,
        self._action_spec,
        constraint_network=constraint_net)

    agent = greedy_agent.GreedyRewardPredictionAgent(
        time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        accepts_per_arm_features=True,
        optimizer=optimizer,
        constraints=[neural_constraint])
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tf.cast(
                tf.reshape(tf.range(24), shape=[2, 4, 3]), dtype=tf.float32)
    }
    actions = np.array([0, 3], dtype=np.int32)
    rewards = {
        'reward': np.array([0.5, 3.0], dtype=np.float32),
        'constraint': np.array([6.0, 4.0], dtype=np.float32)
    }
    initial_step, final_step = _get_initial_and_final_steps_nested_rewards(
        observations, rewards)
    action_step = policy_step.PolicyStep(
        action=tf.convert_to_tensor(actions),
        info=policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=np.array([[1, 2, 3], [3, 2, 1]],
                                         dtype=np.float32)))
    experience = _get_experience(initial_step, action_step, final_step)
    agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())


if __name__ == '__main__':
  tf.test.main()
