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

"""Tests for py_epsilon_greedy_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing.absltest import mock
from tf_agents.policies import py_epsilon_greedy_policy
from tf_agents.trajectories import policy_step


class EpsilonGreedyPolicyTest(absltest.TestCase):

  def setUp(self):
    self.greedy_policy = mock.MagicMock()
    self.random_policy = mock.MagicMock()
    self.random_policy.action.return_value = policy_step.PolicyStep(0, ())

  def testCtorAutoRandomPolicy(self):
    self.greedy_policy.action_spec = mock.MagicMock()
    policy = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
        self.greedy_policy, 0.5)
    self.assertEqual(self.greedy_policy.action_spec,
                     policy._random_policy.action_spec)

  def testCtorValueErrorNegativeEpsilon(self):
    with self.assertRaises(ValueError):
      py_epsilon_greedy_policy.EpsilonGreedyPolicy(
          self.greedy_policy, -0.00001, random_policy=self.random_policy)

  def testCtorValueErrorEpsilonMorThanOne(self):
    with self.assertRaises(ValueError):
      py_epsilon_greedy_policy.EpsilonGreedyPolicy(
          self.greedy_policy, 1.00001, random_policy=self.random_policy)

  def testCtorValueErrorMissingEpsilonEndValue(self):
    with self.assertRaises(ValueError):
      py_epsilon_greedy_policy.EpsilonGreedyPolicy(
          self.greedy_policy, 0.99,
          random_policy=self.random_policy,
          epsilon_decay_end_count=100)

  def testZeroState(self):
    policy = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
        self.greedy_policy, 0.5, random_policy=self.random_policy)
    policy.get_initial_state()
    self.greedy_policy.reset.assert_called_once_with(batch_size=None)
    self.random_policy.reset.assert_called_once_with(batch_size=None)

  def testActionAlwaysRandom(self):
    policy = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
        self.greedy_policy, 1, random_policy=self.random_policy)
    time_step = mock.MagicMock()
    for _ in range(5):
      policy.action(time_step)
    self.random_policy.action.assert_called_with(time_step)
    self.assertEqual(5, self.random_policy.action.call_count)
    self.assertEqual(0, self.greedy_policy.action.call_count)

  def testActionAlwaysGreedy(self):
    policy = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
        self.greedy_policy, 0, random_policy=self.random_policy)
    time_step = mock.MagicMock()
    for _ in range(5):
      policy.action(time_step)
    self.greedy_policy.action.assert_called_with(time_step, policy_state=())
    self.assertEqual(0, self.random_policy.action.call_count)
    self.assertEqual(5, self.greedy_policy.action.call_count)

  def testActionSelection(self):
    policy = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
        self.greedy_policy, 0.9, random_policy=self.random_policy)
    time_step = mock.MagicMock()
    # Replace the random generator with fixed behaviour
    random = mock.MagicMock()
    policy._rng = random

    # 0.8 < 0.9, so random policy should be used.
    policy._rng.rand.return_value = 0.8
    policy.action(time_step)
    self.random_policy.action.assert_called_with(time_step)
    self.assertEqual(1, self.random_policy.action.call_count)
    self.assertEqual(0, self.greedy_policy.action.call_count)

    # 0.91 > 0.9, so greedy policy should be used.
    policy._rng.rand.return_value = 0.91
    policy.action(time_step)
    self.greedy_policy.action.assert_called_with(time_step, policy_state=())
    self.assertEqual(1, self.random_policy.action.call_count)
    self.assertEqual(1, self.greedy_policy.action.call_count)

  def testActionSelectionWithEpsilonDecay(self):
    policy = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
        self.greedy_policy, 0.9, random_policy=self.random_policy,
        epsilon_decay_end_count=10,
        epsilon_decay_end_value=0.4)
    time_step = mock.MagicMock()
    # Replace the random generator with fixed behaviour
    random = mock.MagicMock()
    policy._rng = random

    # 0.8 < 0.9 and 0.8 < 0.85, so random policy should be used.
    policy._rng.rand.return_value = 0.8
    for _ in range(2):
      policy.action(time_step)
      self.random_policy.action.assert_called_with(time_step)
    self.assertEqual(2, self.random_policy.action.call_count)
    self.assertEqual(0, self.greedy_policy.action.call_count)

    # epislon will change from [0.8 to 0.4], and greedy policy should be used
    for _ in range(8):
      policy.action(time_step)
      self.greedy_policy.action.assert_called_with(time_step, policy_state=())
    self.assertEqual(2, self.random_policy.action.call_count)
    self.assertEqual(8, self.greedy_policy.action.call_count)

    # 0.399 < 0.4, random policy should be used.
    policy._rng.rand.return_value = 0.399
    self.random_policy.reset_mock()
    for _ in range(5):
      policy.action(time_step)
      self.random_policy.action.assert_called_with(time_step)
    self.assertEqual(5, self.random_policy.action.call_count)
    # greedy policy should not be called any more
    self.assertEqual(8, self.greedy_policy.action.call_count)


if __name__ == '__main__':
  absltest.main()
