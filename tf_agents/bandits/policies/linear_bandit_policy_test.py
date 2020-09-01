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

"""Tests for tf_agents.bandits.policies.linear_bandit_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import linear_bandit_policy as linear_policy
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

_POLICY_VARIABLES_OFFSET = 10.0


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': 'batch1UCB',
          'batch_size': 1,
          'exploration_strategy': linear_policy.ExplorationStrategy.optimistic,
      }, {
          'testcase_name': 'batch4UCB',
          'batch_size': 4,
          'exploration_strategy': linear_policy.ExplorationStrategy.optimistic,
      })


def test_cases_with_strategy():
  return parameterized.named_parameters(
      {
          'testcase_name': 'batch1UCB',
          'batch_size': 1,
          'exploration_strategy': linear_policy.ExplorationStrategy.optimistic,
      }, {
          'testcase_name': 'batch4UCB',
          'batch_size': 4,
          'exploration_strategy': linear_policy.ExplorationStrategy.optimistic,
      }, {
          'testcase_name': 'batch1TS',
          'batch_size': 1,
          'exploration_strategy': linear_policy.ExplorationStrategy.sampling,
      }, {
          'testcase_name': 'batch4TS',
          'batch_size': 4,
          'exploration_strategy': linear_policy.ExplorationStrategy.sampling,
      })


def test_cases_with_decomposition():
  return parameterized.named_parameters(
      {
          'testcase_name': 'batch1',
          'batch_size': 1,
          'use_decomposition': False
      }, {
          'testcase_name': 'batch4',
          'batch_size': 4,
          'use_decomposition': True
      })


class LinearBanditPolicyTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(LinearBanditPolicyTest, self).setUp()
    self._obs_dim = 2
    self._num_actions = 5
    self._obs_spec = tensor_spec.TensorSpec([self._obs_dim], tf.float32)
    self._obs_spec_with_mask = (tensor_spec.TensorSpec([self._obs_dim],
                                                       tf.float32),
                                tensor_spec.TensorSpec([self._num_actions],
                                                       tf.int32))
    self._per_arm_obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        self._obs_dim, 4, self._num_actions, add_num_actions_feature=True)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._time_step_spec_with_mask = ts.time_step_spec(self._obs_spec_with_mask)
    self._per_arm_time_step_spec = ts.time_step_spec(self._per_arm_obs_spec)
    self._alpha = 1.0
    self._action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=self._num_actions - 1,
        name='action')

  @property
  def _a(self):
    a_for_one_arm = tf.constant([[4, 1], [1, 4]], dtype=tf.float32)
    return [a_for_one_arm] * self._num_actions

  @property
  def _a_numpy(self):
    a_for_one_arm = np.array([[4, 1], [1, 4]], dtype=np.float32)
    return [a_for_one_arm] * self._num_actions

  @property
  def _b(self):
    return [tf.constant([r, r], dtype=tf.float32)
            for r in range(self._num_actions)]

  @property
  def _b_numpy(self):
    return [np.array([r, r], dtype=np.float32)
            for r in range(self._num_actions)]

  @property
  def _num_samples_per_arm(self):
    a_for_one_arm = tf.constant([1], dtype=tf.float32)
    return [a_for_one_arm] * self._num_actions

  @property
  def _num_samples_per_arm_numpy(self):
    return np.ones(self._num_actions)

  def _time_step_batch(self, batch_size):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, dtype=tf.int32, shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        tf.constant(np.array(range(batch_size * self._obs_dim)),
                    dtype=tf.float32, shape=[batch_size, self._obs_dim],
                    name='observation'))

  def _per_arm_time_step_batch(self, batch_size):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST,
            dtype=tf.int32,
            shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        {
            bandit_spec_utils.GLOBAL_FEATURE_KEY:
                tf.constant(
                    np.array(range(batch_size * self._obs_dim)),
                    dtype=tf.float32,
                    shape=[batch_size, self._obs_dim],
                    name='observation'),
            bandit_spec_utils.PER_ARM_FEATURE_KEY:
                tf.constant(
                    np.array(range(batch_size * self._num_actions * 4)),
                    dtype=tf.float32,
                    shape=[batch_size, self._num_actions, 4],
                    name='observation'),
            bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
                tf.ones([batch_size], dtype=tf.int32) * 2

        })

  def _time_step_batch_with_mask(self, batch_size):
    no_mask_observation = tf.constant(
        np.array(range(batch_size * self._obs_dim)),
        dtype=tf.float32,
        shape=[batch_size, self._obs_dim])
    mask = tf.eye(batch_size, num_columns=self._num_actions, dtype=tf.int32)
    observation = (no_mask_observation, mask)
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST,
            dtype=tf.int32,
            shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        observation)

  @parameterized.parameters([
      linear_policy.ExplorationStrategy.optimistic,
      linear_policy.ExplorationStrategy.sampling
  ])
  def testBuild(self, exploration_strategy):
    policy = linear_policy.LinearBanditPolicy(self._action_spec, self._a,
                                              self._b,
                                              self._num_samples_per_arm,
                                              self._time_step_spec,
                                              exploration_strategy)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)

  @test_cases_with_strategy()
  def testObservationShapeMismatch(self, batch_size, exploration_strategy):
    policy = linear_policy.LinearBanditPolicy(self._action_spec, self._a,
                                              self._b,
                                              self._num_samples_per_arm,
                                              self._time_step_spec,
                                              exploration_strategy)

    current_time_step = ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST,
            dtype=tf.int32,
            shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        tf.constant(
            np.array(range(batch_size * (self._obs_dim + 1))),
            dtype=tf.float32,
            shape=[batch_size, self._obs_dim + 1],
            name='observation'))
    with self.assertRaisesRegex(
        ValueError, r'Global observation shape is expected to be \[None, 2\].'
        r' Got \[%d, 3\].' % batch_size):
      policy.action(current_time_step)

  @test_cases_with_strategy()
  def testActionBatch(self, batch_size, exploration_strategy):
    policy = linear_policy.LinearBanditPolicy(self._action_spec, self._a,
                                              self._b,
                                              self._num_samples_per_arm,
                                              self._time_step_spec,
                                              exploration_strategy)

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    actions_ = self.evaluate(action_step.action)
    self.assertAllGreaterEqual(actions_, self._action_spec.minimum)
    self.assertAllLessEqual(actions_, self._action_spec.maximum)

  @test_cases_with_strategy()
  def testActionBatchWithBias(self, batch_size, exploration_strategy):
    a = [tf.constant([[4, 1, 2], [1, 5, 3], [2, 3, 6]], dtype=tf.float32)
        ] * self._num_actions
    b = [
        tf.constant([r, r, r], dtype=tf.float32)
        for r in range(self._num_actions)
    ]
    policy = linear_policy.LinearBanditPolicy(
        self._action_spec,
        a,
        b,
        self._num_samples_per_arm,
        self._time_step_spec,
        exploration_strategy,
        add_bias=True)

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    actions_ = self.evaluate(action_step.action)
    self.assertAllGreaterEqual(actions_, self._action_spec.minimum)
    self.assertAllLessEqual(actions_, self._action_spec.maximum)

  @test_cases_with_strategy()
  def testActionBatchWithMask(self, batch_size, exploration_strategy):

    def split_fn(obs):
      return obs[0], obs[1]

    policy = linear_policy.LinearBanditPolicy(
        self._action_spec,
        self._a,
        self._b,
        self._num_samples_per_arm,
        self._time_step_spec_with_mask,
        exploration_strategy,
        observation_and_action_constraint_splitter=split_fn)

    action_step = policy.action(
        self._time_step_batch_with_mask(batch_size=batch_size))
    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    actions_ = self.evaluate(action_step.action)
    self.assertAllEqual(actions_, range(batch_size))

  @test_cases()
  def testActionBatchWithVariablesAndPolicyUpdate(self, batch_size,
                                                  exploration_strategy):
    a_list = []
    a_new_list = []
    b_list = []
    b_new_list = []
    num_samples_list = []
    num_samples_new_list = []
    for k in range(1, self._num_actions + 1):
      a_initial_value = tf.constant(
          [[2 * k + 1, k + 1], [k + 1, 2 * k+1]],
          dtype=tf.float32)
      a_for_one_arm = tf.compat.v2.Variable(a_initial_value)
      a_list.append(a_for_one_arm)
      b_initial_value = tf.constant([k, k], dtype=tf.float32)
      b_for_one_arm = tf.compat.v2.Variable(b_initial_value)
      b_list.append(b_for_one_arm)
      num_samples_initial_value = tf.constant([1], dtype=tf.float32)
      num_samples_for_one_arm = tf.compat.v2.Variable(num_samples_initial_value)
      num_samples_list.append(num_samples_for_one_arm)

      # Variables for the new policy (they differ by an offset).
      a_new_for_one_arm = tf.compat.v2.Variable(
          a_initial_value + _POLICY_VARIABLES_OFFSET)
      a_new_list.append(a_new_for_one_arm)
      b_new_for_one_arm = tf.compat.v2.Variable(
          b_initial_value + _POLICY_VARIABLES_OFFSET)
      b_new_list.append(b_new_for_one_arm)
      num_samples_for_one_arm_new = tf.compat.v2.Variable(
          num_samples_initial_value + _POLICY_VARIABLES_OFFSET)
      num_samples_new_list.append(num_samples_for_one_arm_new)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    policy = linear_policy.LinearBanditPolicy(self._action_spec, a_list, b_list,
                                              num_samples_list,
                                              self._time_step_spec,
                                              exploration_strategy)
    self.assertLen(policy.variables(), 3 * self._num_actions)

    new_policy = linear_policy.LinearBanditPolicy(self._action_spec, a_new_list,
                                                  b_new_list,
                                                  num_samples_new_list,
                                                  self._time_step_spec,
                                                  exploration_strategy)
    self.assertLen(new_policy.variables(), 3 * self._num_actions)

    self.evaluate(new_policy.update(policy))

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    new_action_step = new_policy.action(
        self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)
    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  @test_cases()
  def testPerArmActionBatchWithVariablesAndPolicyUpdate(self, batch_size,
                                                        exploration_strategy):
    a_value = tf.reshape(tf.range(36, dtype=tf.float32), shape=[6, 6])
    a_list = [tf.compat.v2.Variable(a_value)]
    a_new_list = [tf.compat.v2.Variable(a_value + _POLICY_VARIABLES_OFFSET)]
    b_value = tf.constant([2, 2, 2, 2, 2, 2], dtype=tf.float32)
    b_list = [tf.compat.v2.Variable(b_value)]
    b_new_list = [tf.compat.v2.Variable(b_value + _POLICY_VARIABLES_OFFSET)]
    num_samples_list = [
        tf.compat.v2.Variable(tf.constant([1], dtype=tf.float32))
    ]
    num_samples_new_list = [
        tf.compat.v2.Variable(
            tf.constant([1 + _POLICY_VARIABLES_OFFSET], dtype=tf.float32))
    ]
    self.evaluate(tf.compat.v1.global_variables_initializer())
    policy = linear_policy.LinearBanditPolicy(
        self._action_spec,
        a_list,
        b_list,
        num_samples_list,
        self._per_arm_time_step_spec,
        exploration_strategy,
        accepts_per_arm_features=True)
    self.assertLen(policy.variables(), 3)

    new_policy = linear_policy.LinearBanditPolicy(
        self._action_spec,
        a_new_list,
        b_new_list,
        num_samples_new_list,
        self._per_arm_time_step_spec,
        exploration_strategy,
        accepts_per_arm_features=True)
    self.assertLen(new_policy.variables(), 3)

    self.evaluate(new_policy.update(policy))

    step_batch = self._per_arm_time_step_batch(batch_size=batch_size)
    action_step = policy.action(step_batch)
    new_action_step = new_policy.action(step_batch)
    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)
    actions_, new_actions_, info = self.evaluate(
        [action_step.action, new_action_step.action, action_step.info])
    self.assertAllEqual(actions_, new_actions_)
    arm_obs = step_batch.observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]
    first_action = actions_[0]
    first_arm_features = arm_obs[0]
    self.assertAllEqual(info.chosen_arm_features[0],
                        first_arm_features[first_action])

  @test_cases_with_decomposition()
  def testComparisonWithNumpy(self, batch_size, use_decomposition=False):
    eig_matrix_list = ()
    eig_vals_list = ()
    if use_decomposition:
      eig_vals_one_arm, eig_matrix_one_arm = tf.linalg.eigh(self._a[0])
      eig_vals_list = [eig_vals_one_arm] * self._num_actions
      eig_matrix_list = [eig_matrix_one_arm] * self._num_actions

    policy = linear_policy.LinearBanditPolicy(
        self._action_spec,
        self._a,
        self._b,
        self._num_samples_per_arm,
        self._time_step_spec,
        eig_vals=eig_vals_list,
        eig_matrix=eig_matrix_list)

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    actions_ = self.evaluate(action_step.action)

    observation_numpy = np.array(
        range(batch_size * self._obs_dim), dtype=np.float32).reshape(
            [batch_size, self._obs_dim])

    p_values = []
    for k in range(self._num_actions):
      a_inv = np.linalg.inv(self._a_numpy[k] + np.eye(self._obs_dim))
      theta = np.matmul(
          a_inv, self._b_numpy[k].reshape([self._obs_dim, 1]))
      confidence_intervals = np.sqrt(np.diag(
          np.matmul(observation_numpy,
                    np.matmul(a_inv, np.transpose(observation_numpy)))))
      p_value = (np.matmul(observation_numpy, theta) +
                 self._alpha * confidence_intervals.reshape([-1, 1]))
      p_values.append(p_value)

    actions_numpy = np.argmax(np.stack(p_values, axis=-1), axis=-1).reshape(
        [batch_size])
    self.assertAllEqual(actions_.reshape([batch_size]), actions_numpy)

  @test_cases_with_strategy()
  def testPredictedRewards(self, batch_size, exploration_strategy):
    policy = linear_policy.LinearBanditPolicy(
        self._action_spec,
        self._a,
        self._b,
        self._num_samples_per_arm,
        self._time_step_spec,
        exploration_strategy,
        emit_policy_info=(policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,))

    action_step = policy.action(self._time_step_batch(batch_size=batch_size))
    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)

    observation_numpy = np.array(
        range(batch_size * self._obs_dim), dtype=np.float32).reshape(
            [batch_size, self._obs_dim])

    p_values = []
    predicted_rewards_expected = []
    for k in range(self._num_actions):
      a_inv = np.linalg.inv(self._a_numpy[k] + np.eye(self._obs_dim))
      theta = np.matmul(
          a_inv, self._b_numpy[k].reshape([self._obs_dim, 1]))
      confidence_intervals = np.sqrt(np.diag(
          np.matmul(observation_numpy,
                    np.matmul(a_inv, np.transpose(observation_numpy)))))
      est_mean_reward = np.matmul(observation_numpy, theta)
      predicted_rewards_expected.append(est_mean_reward)
      p_value = (est_mean_reward +
                 self._alpha * confidence_intervals.reshape([-1, 1]))
      p_values.append(p_value)

    predicted_rewards_expected_array = np.stack(
        predicted_rewards_expected, axis=-1).reshape(
            batch_size, self._num_actions)
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)


if __name__ == '__main__':
  tf.test.main()
