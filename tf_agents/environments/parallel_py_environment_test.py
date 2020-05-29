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

"""Tests for the parallel_py_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import time

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import parallel_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import random_py_environment
from tf_agents.specs import array_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import time_step as ts


class SlowStartingEnvironment(random_py_environment.RandomPyEnvironment):

  def __init__(self, *args, **kwargs):
    time_sleep = kwargs.pop('time_sleep', 1.0)
    time.sleep(time_sleep)
    super(SlowStartingEnvironment, self).__init__(*args, **kwargs)


class ParallelPyEnvironmentTest(tf.test.TestCase):

  def _set_default_specs(self):
    self.observation_spec = array_spec.ArraySpec((3, 3), np.float32)
    self.time_step_spec = ts.time_step_spec(self.observation_spec)
    self.action_spec = array_spec.BoundedArraySpec([7],
                                                   dtype=np.float32,
                                                   minimum=-1.0,
                                                   maximum=1.0)

  def _make_parallel_py_environment(self,
                                    constructor=None,
                                    num_envs=2,
                                    start_serially=True,
                                    blocking=True):
    self._set_default_specs()
    constructor = constructor or functools.partial(
        random_py_environment.RandomPyEnvironment, self.observation_spec,
        self.action_spec)
    return parallel_py_environment.ParallelPyEnvironment(
        env_constructors=[constructor] * num_envs, blocking=blocking,
        start_serially=start_serially)

  def test_close_no_hang_after_init(self):
    env = self._make_parallel_py_environment()
    env.close()

  def test_get_specs(self):
    env = self._make_parallel_py_environment()
    self.assertEqual(self.observation_spec, env.observation_spec())
    self.assertEqual(self.time_step_spec, env.time_step_spec())
    self.assertEqual(self.action_spec, env.action_spec())

    env.close()

  def test_step(self):
    num_envs = 2
    env = self._make_parallel_py_environment(num_envs=num_envs)
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    rng = np.random.RandomState()
    action = np.array([
        array_spec.sample_bounded_spec(action_spec, rng)
        for _ in range(num_envs)
    ])
    env.reset()

    # Take one step and assert observation is batched the right way.
    time_step = env.step(action)
    self.assertEqual(num_envs, time_step.observation.shape[0])
    self.assertAllEqual(observation_spec.shape, time_step.observation.shape[1:])
    self.assertEqual(num_envs, action.shape[0])
    self.assertAllEqual(action_spec.shape, action.shape[1:])

    # Take another step and assert that observations have the same shape.
    time_step2 = env.step(action)
    self.assertAllEqual(time_step.observation.shape,
                        time_step2.observation.shape)
    env.close()

  def test_non_blocking_start_processes_in_parallel(self):
    self._set_default_specs()
    constructor = functools.partial(
        SlowStartingEnvironment,
        self.observation_spec,
        self.action_spec,
        time_sleep=1.0)
    start_time = time.time()
    env = self._make_parallel_py_environment(
        constructor=constructor, num_envs=10, start_serially=False,
        blocking=False)
    end_time = time.time()
    self.assertLessEqual(
        end_time - start_time,
        5.0,
        msg=('Expected all processes to start together, '
             'got {} wait time').format(end_time - start_time))
    env.close()

  def test_blocking_start_processes_one_after_another(self):
    self._set_default_specs()
    constructor = functools.partial(
        SlowStartingEnvironment,
        self.observation_spec,
        self.action_spec,
        time_sleep=1.0)
    start_time = time.time()
    env = self._make_parallel_py_environment(
        constructor=constructor, num_envs=10, start_serially=True,
        blocking=True)
    end_time = time.time()
    self.assertGreater(
        end_time - start_time,
        10,
        msg=('Expected all processes to start one '
             'after another, got {} wait time').format(end_time - start_time))
    env.close()

  def test_unstack_actions(self):
    num_envs = 2
    env = self._make_parallel_py_environment(num_envs=num_envs)
    action_spec = env.action_spec()
    rng = np.random.RandomState()
    batched_action = np.array([
        array_spec.sample_bounded_spec(action_spec, rng)
        for _ in range(num_envs)
    ])

    # Test that actions are correctly unstacked when just batched in np.array.
    unstacked_actions = env._unstack_actions(batched_action)
    for action in unstacked_actions:
      self.assertAllEqual(action_spec.shape, action.shape)
    env.close()

  def test_unstack_nested_actions(self):
    num_envs = 2
    env = self._make_parallel_py_environment(num_envs=num_envs)
    action_spec = env.action_spec()
    rng = np.random.RandomState()
    batched_action = np.array([
        array_spec.sample_bounded_spec(action_spec, rng)
        for _ in range(num_envs)
    ])

    # Test that actions are correctly unstacked when nested in namedtuple.
    class NestedAction(
        collections.namedtuple('NestedAction', ['action', 'other_var'])):
      pass

    nested_action = NestedAction(
        action=batched_action, other_var=np.array([13.0] * num_envs))
    unstacked_actions = env._unstack_actions(nested_action)
    for nested_action in unstacked_actions:
      self.assertAllEqual(action_spec.shape, nested_action.action.shape)
      self.assertEqual(13.0, nested_action.other_var)
    env.close()

  def test_seedable(self):
    seeds = [0, 1]
    env = self._make_parallel_py_environment()
    env.seed(seeds)
    self.assertEqual(
        np.random.RandomState(0).get_state()[1][-1],
        env._envs[0].access('_rng').get_state()[1][-1])

    self.assertEqual(
        np.random.RandomState(1).get_state()[1][-1],
        env._envs[1].access('_rng').get_state()[1][-1])
    env.close()


class ProcessPyEnvironmentTest(tf.test.TestCase):

  def test_close_no_hang_after_init(self):
    constructor = functools.partial(
        random_py_environment.RandomPyEnvironment,
        array_spec.ArraySpec((3, 3), np.float32),
        array_spec.BoundedArraySpec([1], np.float32, minimum=-1.0, maximum=1.0),
        episode_end_probability=0,
        min_duration=2,
        max_duration=2)
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    env.start()
    env.close()

  def test_close_no_hang_after_step(self):
    constructor = functools.partial(
        random_py_environment.RandomPyEnvironment,
        array_spec.ArraySpec((3, 3), np.float32),
        array_spec.BoundedArraySpec([1], np.float32, minimum=-1.0, maximum=1.0),
        episode_end_probability=0,
        min_duration=5,
        max_duration=5)
    rng = np.random.RandomState()
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    env.start()
    action_spec = env.action_spec()
    env.reset()
    env.step(array_spec.sample_bounded_spec(action_spec, rng))
    env.step(array_spec.sample_bounded_spec(action_spec, rng))
    env.close()

  def test_reraise_exception_in_init(self):
    constructor = MockEnvironmentCrashInInit
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    with self.assertRaises(Exception):
      env.start()

  def test_reraise_exception_in_reset(self):
    constructor = MockEnvironmentCrashInReset
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    env.start()
    with self.assertRaises(Exception):
      env.reset()

  def test_reraise_exception_in_step(self):
    constructor = functools.partial(MockEnvironmentCrashInStep, crash_at_step=3)
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    env.start()
    env.reset()
    action_spec = env.action_spec()
    rng = np.random.RandomState()
    env.step(array_spec.sample_bounded_spec(action_spec, rng))
    env.step(array_spec.sample_bounded_spec(action_spec, rng))
    with self.assertRaises(Exception):
      env.step(array_spec.sample_bounded_spec(action_spec, rng))


class MockEnvironmentCrashInInit(py_environment.PyEnvironment):
  """Raise an error when instantiated."""

  def __init__(self, *unused_args, **unused_kwargs):
    raise RuntimeError()

  def observation_spec(self):
    return []

  def action_spec(self):
    return []

  def _reset(self):
    return ()

  def _step(self, action):
    return ()


class MockEnvironmentCrashInReset(py_environment.PyEnvironment):
  """Raise an error when instantiated."""

  def __init__(self, *unused_args, **unused_kwargs):
    pass

  def observation_spec(self):
    return []

  def action_spec(self):
    return []

  def _reset(self):
    raise RuntimeError()

  def _step(self, action):
    return ()


class MockEnvironmentCrashInStep(random_py_environment.RandomPyEnvironment):
  """Raise an error after specified number of steps in an episode."""

  def __init__(self, crash_at_step):
    super(MockEnvironmentCrashInStep, self).__init__(
        array_spec.ArraySpec((3, 3), np.float32),
        array_spec.BoundedArraySpec([1], np.float32, minimum=-1.0, maximum=1.0),
        episode_end_probability=0,
        min_duration=crash_at_step + 1,
        max_duration=crash_at_step + 1)
    self._crash_at_step = crash_at_step
    self._steps = 0

  def _step(self, *args, **kwargs):
    transition = super(MockEnvironmentCrashInStep, self)._step(*args, **kwargs)
    self._steps += 1
    if self._steps == self._crash_at_step:
      raise RuntimeError()
    return transition


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
