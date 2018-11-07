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

"""Tests for learning.reinforcement_learning.policies.epsilon_greedy_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.environments import time_step as ts
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import fixed_policy
from tf_agents.specs import tensor_spec

nest = tf.contrib.framework.nest


class EpsilonGreedyPolicyTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(EpsilonGreedyPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._num_actions = 3
    self._greedy_action = 1
    self._action_spec = tensor_spec.BoundedTensorSpec((1,), tf.int32, 0,
                                                      self._num_actions-1)
    self._policy = fixed_policy.FixedPolicy(
        np.asarray([self._greedy_action], dtype=np.int32),
        self._time_step_spec,
        self._action_spec)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    self._time_step = ts.restart(observations, batch_size=2)

  def checkActionDistribution(self, actions, epsilon, num_steps):
    # Check that the distribution of sampled actions is aligned with the epsilon
    # values.
    action_counts = np.bincount(np.hstack(actions), minlength=self._num_actions)
    greedy_prob = 1.0-epsilon
    expected_counts = [(epsilon*num_steps)/self._num_actions
                       for _ in range(self._num_actions)]
    expected_counts[self._greedy_action] += greedy_prob*num_steps
    delta = num_steps*0.1
    # Check that action_counts[i] \in [expected-delta, expected+delta]
    for i in range(self._num_actions):
      self.assertLessEqual(action_counts[i], expected_counts[i]+delta)
      self.assertGreaterEqual(action_counts[i], expected_counts[i]-delta)

  @parameterized.parameters({'epsilon': 0.0},
                            {'epsilon': 0.2},
                            {'epsilon': 0.7},
                            {'epsilon': 1.0})
  def testFixedEpsilon(self, epsilon):
    policy = epsilon_greedy_policy.EpsilonGreedyPolicy(self._policy,
                                                       epsilon=epsilon)
    self.assertEqual(policy.time_step_spec(), self._time_step_spec)
    self.assertEqual(policy.action_spec(), self._action_spec)

    policy_state = policy.get_initial_state(batch_size=2)
    action_step = policy.action(self._time_step, policy_state, seed=54)
    nest.assert_same_structure(self._action_spec, action_step.action)

    self.evaluate(tf.global_variables_initializer())
    # Collect 100 steps with the current value of epsilon.
    actions = []
    num_steps = 100
    for _ in range(num_steps):
      action_ = self.evaluate(action_step.action)[0]
      self.assertIn(action_, [0, 1, 2])
      actions.append(action_)

    self.checkActionDistribution(actions, epsilon, num_steps)

  def testTensorEpsilon(self):
    epsilon_ph = tf.placeholder(tf.float32, shape=())
    policy = epsilon_greedy_policy.EpsilonGreedyPolicy(self._policy,
                                                       epsilon=epsilon_ph)
    self.assertEqual(policy.time_step_spec(), self._time_step_spec)
    self.assertEqual(policy.action_spec(), self._action_spec)

    policy_state = policy.get_initial_state(batch_size=2)
    action_step = policy.action(self._time_step, policy_state, seed=54)
    nest.assert_same_structure(self._action_spec, action_step.action)

    self.evaluate(tf.global_variables_initializer())
    with self.test_session() as sess:
      for epsilon in [0.0, 0.2, 0.7, 1.0]:
        # Collect 100 steps with the current value of epsilon.
        actions = []
        num_steps = 1000
        for _ in range(num_steps):
          action_ = sess.run(action_step.action, {epsilon_ph: epsilon})[0]
          self.assertIn(action_, [0, 1, 2])
          actions.append(action_)

        # Verify that action distribution changes as we vary epsilon.
        self.checkActionDistribution(actions, epsilon, num_steps)


if __name__ == '__main__':
  tf.test.main()
