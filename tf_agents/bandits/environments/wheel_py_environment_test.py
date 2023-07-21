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

"""Tests for the Wheel Bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import wheel_py_environment


class WheelBanditPyEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_delta_0',
           delta=0.0),
      dict(testcase_name='_delta_2',
           delta=2.0)
  )
  def test_delta_out_of_bound_parameter(self, delta):
    with self.assertRaisesRegexp(
        ValueError, r'Delta must be in \(0, 1\)\, but saw delta: %g' % delta):
      wheel_py_environment.WheelPyEnvironment(
          delta=delta, mu_base=[1.2, 1.0, 1.0, 1.0, 1.0],
          std_base=0.01 * np.ones(5), mu_high=50.0, std_high=0.01)

  def test_mu_base_out_of_bound_parameter(self):
    mu_base = [1.2, 1.0, 1.0, 1.0, 1.0, 1.0]
    with self.assertRaisesRegexp(
        ValueError, 'The length of \'mu_base\' must be 5, but saw '
        '\'mu_base\':.*'):
      wheel_py_environment.WheelPyEnvironment(
          delta=0.5, mu_base=mu_base,
          std_base=0.01 * np.ones(5), mu_high=50.0, std_high=0.01)

  def test_std_base_out_of_bound_parameter(self):
    with self.assertRaisesRegexp(
        ValueError, r'The length of \'std_base\' must be 5\.'):
      wheel_py_environment.WheelPyEnvironment(
          delta=0.5,
          mu_base=[1.2, 1.0, 1.0, 1.0, 1.0],
          std_base=0.01 * np.ones(6),
          mu_high=50.0,
          std_high=0.01)

  def test_compute_optimal_action_and_reward(self):
    observation = np.array([[0.1, 0.2], [0.3, -0.7], [-0.3, -0.7], [0.3, 0.7],
                            [0.1, 0.3]])
    actual_actions = wheel_py_environment.compute_optimal_action(
        observation, 0.5)
    expected_actions = [0, 2, 4, 1, 0]
    self.assertAllEqual(actual_actions, expected_actions)
    actual_rewards = wheel_py_environment.compute_optimal_reward(
        observation, 0.5, 1.5, 3.0)
    expected_rewards = [1.5, 3.0, 3.0, 3.0, 1.5]
    self.assertAllEqual(actual_rewards, expected_rewards)

  @parameterized.named_parameters(
      dict(testcase_name='_batch_1',
           batch_size=1),
      dict(testcase_name='_batch_4',
           batch_size=4)
  )
  def test_observation_validity(self, batch_size):
    """Tests that the observations fall into the unit circle."""
    env = wheel_py_environment.WheelPyEnvironment(
        delta=0.5, mu_base=[1.2, 1.0, 1.0, 1.0, 1.0],
        std_base=0.01 * np.ones(5), mu_high=50.0, std_high=0.01,
        batch_size=batch_size)

    for _ in range(5):
      observation = env.reset().observation
      self.assertEqual(list(observation.shape),
                       [batch_size] + list(env.observation_spec().shape))
      for i in range(batch_size):
        self.assertLessEqual(np.linalg.norm(observation[i, :]), 1)

  @parameterized.named_parameters(
      dict(testcase_name='_batch_1',
           batch_size=1),
      dict(testcase_name='_batch_4',
           batch_size=4),
  )
  def test_rewards_validity(self, batch_size):
    """Tests that the rewards are valid."""
    env = wheel_py_environment.WheelPyEnvironment(
        delta=0.5, mu_base=[1.2, 1.0, 1.0, 1.0, 1.0],
        std_base=0.01 * np.ones(5), mu_high=50.0, std_high=0.01,
        batch_size=batch_size)
    time_step = env.reset()
    time_step = env.step(np.arange(batch_size))
    self.assertEqual(time_step.reward.shape, (batch_size,))


if __name__ == '__main__':
  tf.test.main()
