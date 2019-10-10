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

"""Tests for tf_agents.bandits.agents.lin_thompson_sampling_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.drivers import driver_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10_float32',
          'batch_size': 1,
          'context_dim': 10,
          'dtype': tf.float32,
      }, {
          'testcase_name': '_batch4_contextdim5_float64',
          'batch_size': 4,
          'context_dim': 5,
          'dtype': tf.float64,
      })


def _get_initial_and_final_steps(batch_size, context_dim):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  initial_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(
          observation,
          dtype=tf.float32,
          shape=[batch_size, context_dim],
          name='observation'))
  final_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(
          observation + 100.0,
          dtype=tf.float32,
          shape=[batch_size, context_dim],
          name='observation'))
  return initial_step, final_step


def _get_initial_and_final_steps_with_action_mask(batch_size,
                                                  context_dim,
                                                  num_actions=None):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
  observation = tf.constant(observation, dtype=tf.float32)
  mask = 1 - tf.eye(batch_size, num_columns=num_actions, dtype=tf.int32)
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  initial_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observation, mask))
  final_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observation + 100.0, mask))
  return initial_step, final_step


def _get_action_step(action):
  return policy_step.PolicyStep(action=tf.convert_to_tensor(action))


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1), single_experience)


@test_util.run_all_in_graph_and_eager_modes
class LinearThompsonSamplingAgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(LinearThompsonSamplingAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()

  @test_cases()
  def testInitializeAgent(self, batch_size, context_dim, dtype):
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=dtype)
    self.evaluate(agent.initialize())

  @test_cases()
  def testLinearThompsonSamplingUpdate(self, batch_size, context_dim, dtype):
    """Check agent updates for specified actions and rewards."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update. Record initial and final
    # weights.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=dtype)
    self.evaluate(agent.initialize())
    initial_weight_covariances = self.evaluate(agent._weight_covariances)
    initial_parameter_estimators = self.evaluate(agent._parameter_estimators)

    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_weight_covariances = self.evaluate(agent.weight_covariances)
    final_parameter_estimators = self.evaluate(agent.parameter_estimators)
    actual_weight_covariances_update = [
        final_weight_covariances[k] - initial_weight_covariances[k]
        for k in range(num_actions)
    ]
    actual_parameter_estimators_update = [
        final_parameter_estimators[k] - initial_parameter_estimators[k]
        for k in range(num_actions)
    ]

    # Compute the expected updates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_weight_covariances_update = []
    expected_parameter_estimators_update = []
    for k, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      expected_weight_covariances_update.append(
          self.evaluate(
              tf.matmul(
                  observations_for_arm, observations_for_arm,
                  transpose_a=True)))
      expected_parameter_estimators_update.append(
          self.evaluate(
              bandit_utils.sum_reward_weighted_observations(
                  rewards_for_arm, observations_for_arm)))
    self.assertAllClose(expected_weight_covariances_update,
                        actual_weight_covariances_update)
    self.assertAllClose(expected_parameter_estimators_update,
                        actual_parameter_estimators_update)

  @test_cases()
  def testLinearThompsonSamplingUpdateWithMaskedActions(self, batch_size,
                                                        context_dim, dtype):
    """Check agent updates for specified actions and rewards."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        batch_size, context_dim, num_actions)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update. Record initial and final
    # weights.
    observation_spec = (tensor_spec.TensorSpec([context_dim], tf.float32),
                        tensor_spec.TensorSpec([num_actions], tf.int32))
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]),
        dtype=dtype)
    self.evaluate(agent.initialize())
    initial_weight_covariances = self.evaluate(agent._weight_covariances)
    initial_parameter_estimators = self.evaluate(agent._parameter_estimators)

    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_weight_covariances = self.evaluate(agent.weight_covariances)
    final_parameter_estimators = self.evaluate(agent.parameter_estimators)
    actual_weight_covariances_update = [
        final_weight_covariances[k] - initial_weight_covariances[k]
        for k in range(num_actions)
    ]
    actual_parameter_estimators_update = [
        final_parameter_estimators[k] - initial_parameter_estimators[k]
        for k in range(num_actions)
    ]

    # Compute the expected updates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation[0], [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_weight_covariances_update = []
    expected_parameter_estimators_update = []
    for k, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      expected_weight_covariances_update.append(
          self.evaluate(
              tf.matmul(
                  observations_for_arm, observations_for_arm,
                  transpose_a=True)))
      expected_parameter_estimators_update.append(
          self.evaluate(
              bandit_utils.sum_reward_weighted_observations(
                  rewards_for_arm, observations_for_arm)))
    self.assertAllClose(expected_weight_covariances_update,
                        actual_weight_covariances_update)
    self.assertAllClose(expected_parameter_estimators_update,
                        actual_parameter_estimators_update)

  @test_cases()
  def testLinearThompsonSamplingUpdateWithForgetting(
      self, batch_size, context_dim, dtype):
    """Check forgetting agent updates for specified actions and rewards."""
    gamma = 0.9

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update. Record initial and final
    # weights.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        gamma=gamma,
        dtype=dtype)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    initial_weight_covariances = self.evaluate(agent._weight_covariances)
    initial_parameter_estimators = self.evaluate(agent._parameter_estimators)

    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_weight_covariances = self.evaluate(agent.weight_covariances)
    final_parameter_estimators = self.evaluate(agent.parameter_estimators)

    # Compute the expected updates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_weight_covariances_update = []
    expected_parameter_estimators_update = []
    for k, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      expected_weight_covariances_update.append(
          self.evaluate(
              gamma * initial_weight_covariances[k] + tf.matmul(
                  observations_for_arm, observations_for_arm,
                  transpose_a=True)))
      expected_parameter_estimators_update.append(
          self.evaluate(
              gamma * initial_parameter_estimators[k] +
              bandit_utils.sum_reward_weighted_observations(
                  rewards_for_arm, observations_for_arm)))
    self.assertAllClose(expected_weight_covariances_update,
                        final_weight_covariances)
    self.assertAllClose(expected_parameter_estimators_update,
                        final_parameter_estimators)


if __name__ == '__main__':
  tf.test.main()
