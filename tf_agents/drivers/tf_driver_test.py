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

"""Tests for tf_agents.drivers.tf_driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.drivers import tf_driver
from tf_agents.environments import batched_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils


class MockReplayBufferObserver(object):

  def __init__(self):
    self._trajectories = []

  def __call__(self, trajectory_):

    def _add_trajectory(*t):
      self._trajectories.append(tf.nest.pack_sequence_as(trajectory_, t))

    tf.numpy_function(_add_trajectory, tf.nest.flatten(trajectory_), [])

  def gather_all(self):
    return self._trajectories


class TFDriverTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(TFDriverTest, self).setUp()
    f0 = np.array(0., dtype=np.float32)
    f1 = np.array(1., dtype=np.float32)

    # Order of args for trajectory methods:
    # (observation, action, policy_info, reward, discount)
    trajectories = [
        trajectory.first(0, 1, 2, f1, f1),
        trajectory.last(1, 2, 4, f1, f0),
        trajectory.boundary(3, 1, 2, f0, f1),
        trajectory.first(0, 1, 2, f1, f1),
        trajectory.last(1, 2, 4, f1, f0),
        trajectory.boundary(3, 1, 2, f0, f1),
        trajectory.first(0, 1, 2, f1, f1),
    ]
    self._trajectories = nest_utils.batch_nested_array(trajectories)

  @parameterized.named_parameters([
      ('NoneStepsTwoEpisodes', None, 2, 5),
      ('TwoStepsTwoEpisodes', 2, 2, 2),
      ('FourStepsTwoEpisodes', 4, 2, 5),
      ('FourStepsOneEpisodes', 4, 1, 2),
      ('FourStepsNoneEpisodes', 4, None, 5),
  ])
  def testRunOnce(self, max_steps, max_episodes, expected_steps):
    env = driver_test_utils.PyEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    policy = driver_test_utils.TFPolicyMock(tf_env.time_step_spec(),
                                            tf_env.action_spec())

    replay_buffer_observer = MockReplayBufferObserver()
    transition_replay_buffer_observer = MockReplayBufferObserver()
    driver = tf_driver.TFDriver(
        tf_env,
        policy,
        observers=[replay_buffer_observer],
        transition_observers=[transition_replay_buffer_observer],
        max_steps=max_steps,
        max_episodes=max_episodes)

    initial_time_step = tf_env.reset()
    initial_policy_state = policy.get_initial_state(batch_size=1)
    self.evaluate(driver.run(initial_time_step, initial_policy_state))
    trajectories = replay_buffer_observer.gather_all()
    self.assertEqual(trajectories, self._trajectories[:expected_steps])

    transitions = transition_replay_buffer_observer.gather_all()
    self.assertLen(transitions, expected_steps)
    # TimeStep, Action, NextTimeStep
    self.assertLen(transitions[0], 3)

  def testMultipleRunMaxSteps(self):
    num_steps = 3
    num_expected_steps = 4

    env = driver_test_utils.PyEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    policy = driver_test_utils.TFPolicyMock(tf_env.time_step_spec(),
                                            tf_env.action_spec())

    replay_buffer_observer = MockReplayBufferObserver()
    driver = tf_driver.TFDriver(
        tf_env,
        policy,
        observers=[replay_buffer_observer],
        max_steps=1,
        max_episodes=None,
    )

    time_step = tf_env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    for _ in range(num_steps):
      time_step, policy_state = self.evaluate(
          driver.run(time_step, policy_state))
    trajectories = replay_buffer_observer.gather_all()
    self.assertEqual(trajectories, self._trajectories[:num_expected_steps])

  def testMultipleRunMaxEpisodes(self):
    num_episodes = 2
    num_expected_steps = 5

    env = driver_test_utils.PyEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    policy = driver_test_utils.TFPolicyMock(tf_env.time_step_spec(),
                                            tf_env.action_spec())

    replay_buffer_observer = MockReplayBufferObserver()
    driver = tf_driver.TFDriver(
        tf_env,
        policy,
        observers=[replay_buffer_observer],
        max_steps=None,
        max_episodes=1,
    )

    time_step = tf_env.reset()
    policy_state = policy.get_initial_state(batch_size=1)
    for _ in range(num_episodes):
      time_step, policy_state = self.evaluate(
          driver.run(time_step, policy_state))
    trajectories = replay_buffer_observer.gather_all()
    self.assertEqual(trajectories, self._trajectories[:num_expected_steps])

  @parameterized.named_parameters([
      ('NoneStepsNoneEpisodes', None, None),
      ('ZeroStepsNoneEpisodes', 0, None),
      ('NoneStepsZeroEpisodes', None, 0),
      ('ZeroStepsZeroEpisodes', 0, 0),
  ])
  def testValueErrorOnInvalidArgs(self, max_steps, max_episodes):
    env = driver_test_utils.PyEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(env)

    policy = driver_test_utils.TFPolicyMock(tf_env.time_step_spec(),
                                            tf_env.action_spec())

    replay_buffer_observer = MockReplayBufferObserver()
    with self.assertRaises(ValueError):
      tf_driver.TFDriver(
          tf_env,
          policy,
          observers=[replay_buffer_observer],
          max_steps=max_steps,
          max_episodes=max_episodes,
      )

  @parameterized.named_parameters([
      ('FourStepsNoneEpisodesBoundaryNotCounted', 4, None, 2),
      ('FiveStepsNoneEpisodesBoundaryNotCounted', 5, None, 3),
      ('NoneStepsTwoEpisodesBoundaryNotCounted', None, 2, 3),
      ('TwoStepsTwoEpisodesBoundaryNotCounted', 2, 2, 1),
      ('FourStepsTwoEpisodesBoundaryNotCounted', 4, 2, 2),
  ])
  def testBatchedEnvironment(self, max_steps, max_episodes, expected_length):

    expected_trajectories = [
        trajectory.Trajectory(
            step_type=np.array([0, 0]),
            observation=np.array([0, 0]),
            action=np.array([2, 1]),
            policy_info=np.array([4, 2]),
            next_step_type=np.array([1, 1]),
            reward=np.array([1., 1.]),
            discount=np.array([1., 1.])),
        trajectory.Trajectory(
            step_type=np.array([1, 1]),
            observation=np.array([2, 1]),
            action=np.array([1, 2]),
            policy_info=np.array([2, 4]),
            next_step_type=np.array([2, 1]),
            reward=np.array([1., 1.]),
            discount=np.array([0., 1.])),
        trajectory.Trajectory(
            step_type=np.array([2, 1]),
            observation=np.array([3, 3]),
            action=np.array([2, 1]),
            policy_info=np.array([4, 2]),
            next_step_type=np.array([0, 2]),
            reward=np.array([0., 1.]),
            discount=np.array([1., 0.]))
    ]

    env1 = driver_test_utils.PyEnvironmentMock(final_state=3)
    env2 = driver_test_utils.PyEnvironmentMock(final_state=4)
    env = batched_py_environment.BatchedPyEnvironment([env1, env2])
    tf_env = tf_py_environment.TFPyEnvironment(env)

    policy = driver_test_utils.TFPolicyMock(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        batch_size=2,
        initial_policy_state=tf.constant([1, 2], dtype=tf.int32))

    replay_buffer_observer = MockReplayBufferObserver()

    driver = tf_driver.TFDriver(
        tf_env,
        policy,
        observers=[replay_buffer_observer],
        max_steps=max_steps,
        max_episodes=max_episodes,
    )
    initial_time_step = tf_env.reset()
    initial_policy_state = tf.constant([1, 2], dtype=tf.int32)
    self.evaluate(driver.run(initial_time_step, initial_policy_state))
    trajectories = replay_buffer_observer.gather_all()

    self.assertEqual(
        len(trajectories), len(expected_trajectories[:expected_length]))

    for t1, t2 in zip(trajectories, expected_trajectories[:expected_length]):
      for t1_field, t2_field in zip(t1, t2):
        self.assertAllEqual(t1_field, t2_field)


if __name__ == '__main__':
  test_utils.main()
