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

"""Tests for tf_agents.policies.epsilon_greedy_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import policy_utilities as policy_util
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import fixed_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils


class EpsilonGreedyPolicyTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(EpsilonGreedyPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._num_actions = 3
    self._greedy_action = 1
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0,
                                                      self._num_actions-1)
    self._policy = fixed_policy.FixedPolicy(
        np.asarray(self._greedy_action, dtype=np.int32),
        self._time_step_spec,
        self._action_spec)
    self._bandit_policy_type = tf.constant([1, 1])
    self._bandit_policy_type_spec = (
        policy_util.create_bandit_policy_type_tensor_spec(shape=()))
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    self._time_step = ts.restart(observations, batch_size=2)

  def checkActionDistribution(self, actions, epsilon, num_steps, log_probs):
    # Check that the distribution of sampled actions is aligned with the epsilon
    # values.
    action_counts = np.bincount(np.hstack(actions), minlength=self._num_actions)
    greedy_prob = 1.0 - epsilon
    expected_counts = [(epsilon * num_steps) / self._num_actions
                       for _ in range(self._num_actions)]
    expected_counts[self._greedy_action] += greedy_prob * num_steps
    delta = num_steps * 0.1
    # Check that action_counts[i] \in [expected-delta, expected+delta]
    for i in range(self._num_actions):
      self.assertLessEqual(action_counts[i], expected_counts[i] + delta)
      self.assertGreaterEqual(action_counts[i], expected_counts[i] - delta)

    # Check that the log probs are correct.
    explore_prob = epsilon / self._num_actions
    for action, log_prob in zip(actions, log_probs):
      if not np.allclose([math.exp(log_prob)], [explore_prob]):
        # If it's not exploring, it must be taking the greedy action.
        self.assertEqual(self._greedy_action, action)
        self.assertAlmostEqual(math.exp(log_prob), greedy_prob)

  @parameterized.named_parameters(
      ('Tensor0.0', 0.0, True), ('Tensor0.2', 0.2, True),
      ('Tensor0.7', 0.7, True), ('Tensor1.0', 1.0, True),
      ('Fixed0.0', 0.0, False), ('Fixed0.2', 0.2, False),
      ('Fixed0.7', 0.7, False), ('Fixed1.0', 1.0, False))
  def testEpsilon(self, float_epsilon, is_tensor):
    epsilon = tf.constant(float_epsilon) if is_tensor else float_epsilon
    policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        self._policy, epsilon=epsilon)
    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

    policy_state = policy.get_initial_state(batch_size=2)
    time_step = tf.nest.map_structure(tf.convert_to_tensor, self._time_step)

    @common.function
    def action_step_fn(time_step=time_step):
      return policy.action(time_step, policy_state, seed=54)

    tf.nest.assert_same_structure(
        self._action_spec,
        self.evaluate(action_step_fn(time_step)).action)

    if tf.executing_eagerly():
      action_step = action_step_fn
    else:
      action_step = action_step_fn()

    actions = []
    log_probs = []

    num_steps = 1000
    for _ in range(num_steps):
      step = self.evaluate(action_step)
      action_ = step.action[0]
      log_prob_ = step.info.log_probability[0]
      self.assertIn(action_, [0, 1, 2])
      actions.append(action_)
      log_probs.append(log_prob_)

    # Verify that action distribution changes as we vary epsilon.
    self.checkActionDistribution(actions, float_epsilon, num_steps, log_probs)

  def checkBanditPolicyTypeShape(self, bandit_policy_type, batch_size):
    self.assertAllEqual(bandit_policy_type.shape, [batch_size])

  def testInfoSpec(self):
    PolicyInfo = collections.namedtuple(  # pylint: disable=invalid-name
        'PolicyInfo',
        ('log_probability', 'predicted_rewards', 'bandit_policy_type'))
    # Set default empty tuple for all fields.
    PolicyInfo.__new__.__defaults__ = ((),) * len(PolicyInfo._fields)

    info_spec = PolicyInfo(
        bandit_policy_type=self._bandit_policy_type_spec,
        log_probability=tensor_spec.BoundedTensorSpec(
            shape=(),
            dtype=tf.float32,
            maximum=0,
            minimum=-float('inf'),
            name='log_probability'))

    policy_with_info_spec = fixed_policy.FixedPolicy(
        np.asarray(self._greedy_action, dtype=np.int32),
        self._time_step_spec,
        self._action_spec,
        policy_info=PolicyInfo(bandit_policy_type=self._bandit_policy_type),
        info_spec=info_spec)

    epsilon = 0.2
    policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        policy_with_info_spec, epsilon=epsilon)
    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

    time_step = tf.nest.map_structure(tf.convert_to_tensor, self._time_step)

    @common.function
    def action_step_fn(time_step=time_step):
      return policy.action(time_step, policy_state=(), seed=54)

    tf.nest.assert_same_structure(
        self._action_spec,
        self.evaluate(action_step_fn(time_step)).action)

    if tf.executing_eagerly():
      action_step = action_step_fn
    else:
      action_step = action_step_fn()

    step = self.evaluate(action_step)
    tf.nest.assert_same_structure(
        info_spec,
        step.info)

    self.checkBanditPolicyTypeShape(step.info.bandit_policy_type, batch_size=2)

if __name__ == '__main__':
  tf.test.main()
