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

"""Tests for tf_agents.bandits.agents.linear_bandit_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_bandit_agent as linear_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.utils import common

tfd = tfp.distributions


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name':
              '_batch1_contextdim10_float32',
          'batch_size':
              1,
          'context_dim':
              10,
          'exploration_policy':
              linear_agent.ExplorationPolicy.linear_ucb_policy,
          'dtype':
              tf.float32,
      }, {
          'testcase_name':
              '_batch4_contextdim5_float64_UCB',
          'batch_size':
              4,
          'context_dim':
              5,
          'exploration_policy':
              linear_agent.ExplorationPolicy.linear_ucb_policy,
          'dtype':
              tf.float64,
      }, {
          'testcase_name':
              '_batch4_contextdim5_float64_TS',
          'batch_size':
              4,
          'context_dim':
              5,
          'exploration_policy':
              linear_agent.ExplorationPolicy.linear_thompson_sampling_policy,
          'dtype':
              tf.float64,
      }, {
          'testcase_name':
              '_batch4_contextdim5_float64_decomp',
          'batch_size':
              4,
          'context_dim':
              5,
          'exploration_policy':
              linear_agent.ExplorationPolicy.linear_ucb_policy,
          'dtype':
              tf.float64,
          'use_eigendecomp':
              True,
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


def _get_initial_and_final_steps_with_per_arm_features(
    batch_size,
    global_context_dim,
    num_actions,
    arm_context_dim,
    apply_action_mask=False,
    num_actions_feature=False):
  global_observation = np.array(range(batch_size * global_context_dim)).reshape(
      [batch_size, global_context_dim])
  arm_observation = np.array(range(
      batch_size * num_actions * arm_context_dim)).reshape(
          [batch_size, num_actions, arm_context_dim])
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  observation = {
      'global':
          tf.constant(
              global_observation,
              dtype=tf.float32,
              shape=[batch_size, global_context_dim],
              name='global_observation'),
      'per_arm':
          tf.constant(
              arm_observation,
              dtype=tf.float32,
              shape=[batch_size, num_actions, arm_context_dim],
              name='arm_observation')
  }
  if num_actions_feature:
    observation.update({
        'num_actions': tf.ones([batch_size], dtype=tf.int32) * (num_actions - 1)
    })
  if apply_action_mask:
    observation = (observation,
                   tf.ones([batch_size, num_actions], dtype=tf.int32))
  initial_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observation)
  observation = {
      'global':
          tf.constant(
              global_observation + 100.0,
              dtype=tf.float32,
              shape=[batch_size, global_context_dim],
              name='global_observation'),
      'arm':
          tf.constant(
              arm_observation + 100.0,
              dtype=tf.float32,
              shape=[batch_size, num_actions, arm_context_dim],
              name='arm_observation')
  }
  if num_actions_feature:
    observation.update(
        {'num_actions': tf.ones([batch_size], dtype=tf.int32) * num_actions})
  if apply_action_mask:
    observation = (observation,
                   tf.ones([batch_size, num_actions], dtype=tf.int32))
  final_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observation)
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
  return policy_step.PolicyStep(
      action=tf.convert_to_tensor(action), info=policy_utilities.PolicyInfo())


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1), single_experience)


class LinearBanditAgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(LinearBanditAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()

  @test_cases()
  def testInitializeAgent(self,
                          batch_size,
                          context_dim,
                          exploration_policy,
                          dtype,
                          use_eigendecomp=False):
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=dtype)
    self.evaluate(agent.initialize())

  def testInitializeAgentEmptyObservationSpec(self):
    dtype = tf.float32
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec((), tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = linear_agent.LinearBanditAgent(
        exploration_policy=linear_agent.ExplorationPolicy.linear_ucb_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=dtype)
    self.evaluate(agent.initialize())

  @test_cases()
  def testLinearAgentUpdate(self,
                            batch_size,
                            context_dim,
                            exploration_policy,
                            dtype,
                            use_eigendecomp=False):
    """Check that the agent updates for specified actions and rewards."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    variable_collection = linear_agent.LinearBanditVariableCollection(
        context_dim, num_actions, use_eigendecomp, dtype)
    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        variable_collection=variable_collection,
        use_eigendecomp=use_eigendecomp,
        dtype=dtype)
    self.evaluate(agent.initialize())
    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)
    final_theta = self.evaluate(agent.theta)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    expected_theta_updated_list = []
    for _, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      num_samples_for_arm_current = tf.cast(
          tf.shape(rewards_for_arm)[0], tf.float32)
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            observations_for_arm, observations_for_arm, transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, observations_for_arm)
        return a_new, b_new

      def false_fn():
        return tf.zeros([context_dim, context_dim]), tf.zeros([context_dim])

      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0, true_fn, false_fn)
      theta_new = tf.squeeze(
          tf.linalg.solve(
              tf.eye(context_dim) + a_new, tf.expand_dims(b_new, axis=-1)),
          axis=-1)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))
      expected_theta_updated_list.append(self.evaluate(theta_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)
    self.assertAllClose(
        self.evaluate(tf.stack(expected_theta_updated_list)),
        final_theta,
        atol=0.1,
        rtol=0.05)

  @test_cases()
  def testLinearAgentUpdatePerArmFeatures(self,
                                          batch_size,
                                          context_dim,
                                          exploration_policy,
                                          dtype,
                                          use_eigendecomp=False):
    """Check that the agent updates for specified actions and rewards."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    global_context_dim = context_dim
    arm_context_dim = 3
    initial_step, final_step = (
        _get_initial_and_final_steps_with_per_arm_features(
            batch_size,
            global_context_dim,
            num_actions,
            arm_context_dim,
            num_actions_feature=True))
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = policy_step.PolicyStep(
        action=tf.convert_to_tensor(action),
        info=policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=np.arange(
                batch_size * arm_context_dim, dtype=np.float32).reshape(
                    [batch_size, arm_context_dim])))
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = bandit_spec_utils.create_per_arm_observation_spec(
        context_dim, arm_context_dim, num_actions, add_num_actions_feature=True)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        use_eigendecomp=use_eigendecomp,
        accepts_per_arm_features=True,
        dtype=dtype)
    self.evaluate(agent.initialize())
    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)

    # Compute the expected updated estimates.
    global_observation = experience.observation[
        bandit_spec_utils.GLOBAL_FEATURE_KEY]
    arm_observation = experience.policy_info.chosen_arm_features
    overall_observation = tf.squeeze(
        tf.concat([global_observation, arm_observation], axis=-1), axis=1)
    rewards = tf.squeeze(experience.reward, axis=1)

    expected_a_new = tf.matmul(
        overall_observation, overall_observation, transpose_a=True)
    expected_b_new = bandit_utils.sum_reward_weighted_observations(
        rewards, overall_observation)
    self.assertAllClose(expected_a_new, final_a[0])
    self.assertAllClose(expected_b_new, final_b[0])

  @test_cases()
  def testLinearAgentUpdateWithBias(self,
                                    batch_size,
                                    context_dim,
                                    exploration_policy,
                                    dtype,
                                    use_eigendecomp=False):
    """Check that the agent updates for specified actions and rewards."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    variable_collection = linear_agent.LinearBanditVariableCollection(
        context_dim + 1, num_actions, use_eigendecomp, dtype)
    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        variable_collection=variable_collection,
        use_eigendecomp=use_eigendecomp,
        add_bias=True,
        dtype=dtype)
    self.evaluate(agent.initialize())
    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)
    final_theta = self.evaluate(agent.theta)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    expected_theta_updated_list = []
    for _, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      observations_for_arm = tf.concat(
          [observations_for_arm,
           tf.ones_like(observations_for_arm[:, 0:1])],
          axis=1)
      num_samples_for_arm_current = tf.cast(
          tf.shape(rewards_for_arm)[0], tf.float32)
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            observations_for_arm, observations_for_arm, transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, observations_for_arm)
        return a_new, b_new

      def false_fn():
        return tf.zeros([context_dim + 1,
                         context_dim + 1]), tf.zeros([context_dim + 1])

      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0, true_fn, false_fn)
      theta_new = tf.squeeze(
          tf.linalg.solve(a_new + tf.eye(context_dim + 1),
                          tf.expand_dims(b_new, axis=-1)),
          axis=-1)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))
      expected_theta_updated_list.append(self.evaluate(theta_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)
    self.assertAllClose(
        self.evaluate(tf.stack(expected_theta_updated_list)),
        final_theta,
        atol=0.1,
        rtol=0.05)

  @test_cases()
  def testLinearAgentUpdateWithMaskedActions(self,
                                             batch_size,
                                             context_dim,
                                             exploration_policy,
                                             dtype,
                                             use_eigendecomp=False):
    """Check that the agent updates for specified actions and rewards."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        batch_size, context_dim, num_actions=num_actions)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = (tensor_spec.TensorSpec([context_dim], tf.float32),
                        tensor_spec.TensorSpec([num_actions], tf.int32))
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)

    def observation_and_action_constraint_splitter(obs):
      return obs[0], obs[1]

    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        dtype=dtype)
    self.evaluate(agent.initialize())
    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(
            observation_and_action_constraint_splitter(
                experience.observation)[0], [batch_size, -1]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    for _, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      num_samples_for_arm_current = tf.cast(
          tf.shape(rewards_for_arm)[0], tf.float32)
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            observations_for_arm, observations_for_arm, transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, observations_for_arm)
        return a_new, b_new

      def false_fn():
        return tf.zeros([context_dim, context_dim]), tf.zeros([context_dim])

      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0, true_fn, false_fn)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)

  @test_cases()
  def testLinearAgentUpdateWithForgetting(self,
                                          batch_size,
                                          context_dim,
                                          exploration_policy,
                                          dtype,
                                          use_eigendecomp=False):
    """Check that the agent updates for specified actions and rewards."""
    # We should rewrite this test as it currently does not depend on
    # the value of `gamma`. To properly test the forgetting factor, we need to
    # call `train` twice.
    gamma = 0.9

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        gamma=gamma,
        dtype=dtype,
        use_eigendecomp=use_eigendecomp)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = agent.train(experience)
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)
    final_eig_vals = self.evaluate(agent.eig_vals)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    expected_eigvals_updated_list = []
    for _, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      num_samples_for_arm_current = tf.cast(
          tf.shape(rewards_for_arm)[0], tf.float32)
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            observations_for_arm, observations_for_arm, transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, observations_for_arm)
        eigmatrix_new = tf.constant([], dtype=dtype)
        eigvals_new = tf.constant([], dtype=dtype)
        if use_eigendecomp:
          eigvals_new, eigmatrix_new = tf.linalg.eigh(a_new)
        return a_new, b_new, eigvals_new, eigmatrix_new

      def false_fn():
        if use_eigendecomp:
          return (tf.zeros([context_dim, context_dim]), tf.zeros([context_dim]),
                  tf.ones([context_dim]), tf.eye(context_dim))
        else:
          return (tf.zeros([context_dim, context_dim]), tf.zeros([context_dim]),
                  tf.constant([], dtype=dtype), tf.constant([], dtype=dtype))

      a_new, b_new, eig_vals_new, _ = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0, true_fn, false_fn)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))
      expected_eigvals_updated_list.append(self.evaluate(eig_vals_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)
    self.assertAllClose(
        expected_eigvals_updated_list, final_eig_vals, atol=2.2e-4, rtol=1e-4)

  @test_cases()
  def testDistributedLinearAgentUpdate(self,
                                       batch_size,
                                       context_dim,
                                       exploration_policy,
                                       dtype,
                                       use_eigendecomp=False):
    """Same as above, but uses the distributed train function of the agent."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)

    agent = linear_agent.LinearBanditAgent(
        exploration_policy=exploration_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=dtype)
    self.evaluate(agent.initialize())
    train_fn = common.function_in_tf1()(agent._distributed_train_step)
    loss_info = train_fn(experience=experience)
    self.evaluate(loss_info)

    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    expected_theta_updated_list = []
    for _, (observations_for_arm,
            rewards_for_arm) in enumerate(zip(observations_list, rewards_list)):
      num_samples_for_arm_current = tf.cast(
          tf.shape(rewards_for_arm)[0], tf.float32)
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            observations_for_arm, observations_for_arm, transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, observations_for_arm)
        return a_new, b_new

      def false_fn():
        return tf.zeros([context_dim, context_dim]), tf.zeros([context_dim])

      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0, true_fn, false_fn)
      theta_new = tf.squeeze(
          tf.linalg.solve(a_new + tf.eye(context_dim),
                          tf.expand_dims(b_new, axis=-1)),
          axis=-1)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))
      expected_theta_updated_list.append(self.evaluate(theta_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)

  def testInitializeRestoreVariableCollection(self):
    if not tf.executing_eagerly():
      self.skipTest('Test only works in eager mode.')
    context_dim = 7
    num_actions = 5
    variable_collection = linear_agent.LinearBanditVariableCollection(
        context_dim=context_dim, num_models=num_actions)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(variable_collection.num_samples_list)
    checkpoint = tf.train.Checkpoint(variable_collection=variable_collection)
    checkpoint_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint')
    checkpoint.save(file_prefix=checkpoint_prefix)

    variable_collection.num_samples_list[2].assign(14)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)
    self.evaluate(checkpoint_load_status.initialize_or_restore())
    self.assertEqual(self.evaluate(variable_collection.num_samples_list[2]), 0)

  def testUCBandThompsonSamplingShareVariables(self):
    if not tf.executing_eagerly():
      self.skipTest('Test only works in eager mode.')
    context_dim = 9
    num_actions = 4
    batch_size = 7
    variable_collection = linear_agent.LinearBanditVariableCollection(
        context_dim=context_dim, num_models=num_actions)
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    ucb_agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        variable_collection=variable_collection)
    ts_agent = linear_thompson_sampling_agent.LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        variable_collection=variable_collection)
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)
    self.evaluate(ucb_agent.train(experience))
    self.assertAllEqual(ucb_agent._variable_collection.cov_matrix_list[0],
                        ts_agent._variable_collection.cov_matrix_list[0])
    self.evaluate(ts_agent.train(experience))
    self.assertAllEqual(ucb_agent._variable_collection.data_vector_list[0],
                        ts_agent._variable_collection.data_vector_list[0])


if __name__ == '__main__':
  tf.test.main()
