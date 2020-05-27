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

"""Tests for the Stationary Stochastic Bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


def normal_with_sigma_1_sampler(mu):
  return np.random.normal(mu, 1)


def check_unbatched_time_step_spec(time_step, time_step_spec, batch_size):
  """Checks if time step conforms array spec, even if batched."""
  if batch_size is None:
    return array_spec.check_arrays_nest(time_step, time_step_spec)

  if not all([spec.shape[0] == batch_size for spec in time_step]):
    return False

  unbatched_time_step = ts.TimeStep(
      step_type=time_step.step_type[0],
      reward=time_step.reward[0],
      discount=time_step.discount[0],
      observation=time_step.observation[0])
  return array_spec.check_arrays_nest(unbatched_time_step, time_step_spec)


class LinearNormalReward(object):

  def __init__(self, theta):
    self.theta = theta

  def __call__(self, x):
    mu = np.dot(x, self.theta)
    return np.random.normal(mu, 1)


class LinearDeterministicReward(object):

  def __init__(self, theta):
    self.theta = theta

  def __call__(self, x):
    return np.dot(x, self.theta)


class LinearDeterministicMultipleRewards(object):

  def __init__(self, thetas):
    self.thetas = thetas

  def __call__(self, x):
    return [np.dot(x, theta) for theta in self.thetas]


class StationaryStochasticBanditPyEnvironmentTest(tf.test.TestCase):

  def test_with_uniform_context_and_normal_mu_reward(self):

    def _context_sampling_fn():
      return np.random.randint(-10, 10, [1, 4])

    reward_fns = [
        LinearNormalReward(theta)
        for theta in ([0, 1, 2, 3], [3, 2, 1, 0], [-1, -2, -3, -4])
    ]

    env = sspe.StationaryStochasticPyEnvironment(_context_sampling_fn,
                                                 reward_fns)
    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()

    random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    for _ in range(5):
      time_step = env.reset()
      self.assertTrue(
          check_unbatched_time_step_spec(
              time_step=time_step,
              time_step_spec=time_step_spec,
              batch_size=env.batch_size))

      action = random_policy.action(time_step).action
      time_step = env.step(action)

  def test_with_normal_context_and_normal_reward(self):

    def _context_sampling_fn():
      return np.random.normal(0, 3, [1, 2])

    def _reward_fn(x):
      return np.random.normal(2 * x[0], abs(x[1]) + 1)

    env = sspe.StationaryStochasticPyEnvironment(_context_sampling_fn,
                                                 [_reward_fn])
    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()

    random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    for _ in range(5):
      time_step = env.reset()
      self.assertTrue(
          check_unbatched_time_step_spec(
              time_step=time_step,
              time_step_spec=time_step_spec,
              batch_size=env.batch_size))

      action = random_policy.action(time_step).action
      time_step = env.step(action)

  def test_deterministic_with_batch_2(self):

    def _context_sampling_fn():
      return np.array([[4, 3], [4, 3]])

    reward_fns = [
        LinearDeterministicReward(theta)
        for theta in ([0, 1], [1, 2], [2, 3], [3, 4])
    ]
    env = sspe.StationaryStochasticPyEnvironment(
        _context_sampling_fn, reward_fns, batch_size=2)
    time_step = env.reset()
    self.assertAllEqual(time_step.observation, [[4, 3], [4, 3]])
    time_step = env.step([0, 1])
    self.assertAllEqual(time_step.reward, [3, 10])
    env.reset()
    time_step = env.step([2, 3])
    self.assertAllEqual(time_step.reward, [17, 24])

  def test_non_scalar_rewards(self):

    def _context_sampling_fn():
      return np.array([[4, 3], [4, 3], [5, 6]])

    # Build a case with 4 arms and 2-dimensional rewards and batch size 3.
    reward_fns = [
        LinearDeterministicMultipleRewards(theta)  # pylint: disable=g-complex-comprehension
        for theta in [np.array([[0, 1], [1, 0]]),
                      np.array([[1, 2], [2, 1]]),
                      np.array([[2, 3], [3, 2]]),
                      np.array([[3, 4], [4, 3]])]
    ]
    env = sspe.StationaryStochasticPyEnvironment(
        _context_sampling_fn, reward_fns, batch_size=3)
    time_step = env.reset()
    self.assertAllEqual(time_step.observation, [[4, 3], [4, 3], [5, 6]])
    time_step = env.step([0, 1, 2])
    self.assertAllEqual(time_step.reward,
                        [[3., 4.],
                         [10., 11.],
                         [28., 27.]])
    env.reset()
    time_step = env.step([2, 3, 0])
    self.assertAllEqual(time_step.reward,
                        [[17., 18.],
                         [24., 25.],
                         [6., 5.]])
    # Check that the reward vectors in the reward spec are 2-dimensional.
    time_step_spec = env.time_step_spec()
    self.assertEqual(time_step_spec.reward.shape[0], 2)

  def test_non_scalar_rewards_and_constraints(self):

    def _context_sampling_fn():
      return np.array([[4, 3], [4, 3], [5, 6]])

    # Build a case with 4 arms and 2-dimensional rewards and batch size 3.
    reward_fns = [
        LinearDeterministicMultipleRewards(theta)  # pylint: disable=g-complex-comprehension
        for theta in [np.array([[0, 1], [1, 0]]),
                      np.array([[1, 2], [2, 1]]),
                      np.array([[2, 3], [3, 2]]),
                      np.array([[3, 4], [4, 3]])]
    ]
    constraint_fns = reward_fns
    env = sspe.StationaryStochasticPyEnvironment(
        _context_sampling_fn, reward_fns, constraint_fns, batch_size=3)
    time_step = env.reset()
    self.assertAllEqual(time_step.observation, [[4, 3], [4, 3], [5, 6]])
    time_step = env.step([0, 1, 2])

    self.assertAllEqual(time_step.reward['reward'],
                        [[3., 4.],
                         [10., 11.],
                         [28., 27.]])
    self.assertAllEqual(time_step.reward['constraint'],
                        [[3., 4.],
                         [10., 11.],
                         [28., 27.]])

    env.reset()
    time_step = env.step([2, 3, 0])
    self.assertAllEqual(time_step.reward['reward'],
                        [[17., 18.],
                         [24., 25.],
                         [6., 5.]])
    self.assertAllEqual(time_step.reward['constraint'],
                        [[17., 18.],
                         [24., 25.],
                         [6., 5.]])
    # Check that the reward vectors in the reward spec and in the
    # constraint_spec are 2-dimensional.
    time_step_spec = env.time_step_spec()
    self.assertEqual(time_step_spec.reward['reward'].shape[0], 2)
    self.assertEqual(time_step_spec.reward['constraint'].shape[0], 2)


if __name__ == '__main__':
  tf.test.main()
