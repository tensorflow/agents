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

import numpy as np
import tensorflow as tf

from tf_agents.environments import parallel_py_environment
from tf_agents.environments import random_py_environment
from tf_agents.environments import time_step as ts
from tf_agents.specs import array_spec


class ParallelPyEnvironmentTest(tf.test.TestCase):

  def _make_parallel_py_environment(self, constructor=None, num_envs=2):
    self.observation_spec = array_spec.ArraySpec((3, 3), np.float32)
    self.time_step_spec = ts.time_step_spec(self.observation_spec)
    self.action_spec = array_spec.BoundedArraySpec(
        [7], dtype=np.float32, minimum=-1.0, maximum=1.0)
    constructor = constructor or functools.partial(
        random_py_environment.RandomPyEnvironment,
        self.observation_spec,
        self.action_spec)
    return parallel_py_environment.ParallelPyEnvironment(
        env_constructors=[constructor] * num_envs, blocking=True)

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
    action = np.array([array_spec.sample_bounded_spec(action_spec, rng)
                       for _ in range(num_envs)])
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

  def test_unstack_actions(self):
    num_envs = 2
    env = self._make_parallel_py_environment(num_envs=num_envs)
    action_spec = env.action_spec()
    rng = np.random.RandomState()
    batched_action = np.array([array_spec.sample_bounded_spec(action_spec, rng)
                               for _ in range(num_envs)])

    # Test that actions are correctly unstacked when just batched in np.array.
    unstacked_actions = env._unstack_actions(batched_action)
    for action in unstacked_actions:
      self.assertAllEqual(action_spec.shape,
                          action.shape)
    env.close()

  def test_unstack_nested_actions(self):
    num_envs = 2
    env = self._make_parallel_py_environment(num_envs=num_envs)
    action_spec = env.action_spec()
    rng = np.random.RandomState()
    batched_action = np.array([array_spec.sample_bounded_spec(action_spec, rng)
                               for _ in range(num_envs)])

    # Test that actions are correctly unstacked when nested in namedtuple.
    class NestedAction(collections.namedtuple(
        'NestedAction', ['action', 'other_var'])):
      pass
    nested_action = NestedAction(action=batched_action,
                                 other_var=np.array([13.0]*num_envs))
    unstacked_actions = env._unstack_actions(nested_action)
    for nested_action in unstacked_actions:
      self.assertAllEqual(action_spec.shape,
                          nested_action.action.shape)
      self.assertEqual(13.0, nested_action.other_var)
    env.close()


class ProcessPyEnvironmentTest(tf.test.TestCase):

  def test_close_no_hang_after_init(self):
    constructor = functools.partial(
        random_py_environment.RandomPyEnvironment,
        array_spec.ArraySpec((3, 3), np.float32),
        array_spec.BoundedArraySpec([1], np.float32, minimum=-1.0, maximum=1.0),
        episode_end_probability=0, min_duration=2, max_duration=2)
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    env.start()
    env.close()

  def test_close_no_hang_after_step(self):
    constructor = functools.partial(
        random_py_environment.RandomPyEnvironment,
        array_spec.ArraySpec((3, 3), np.float32),
        array_spec.BoundedArraySpec([1], np.float32, minimum=-1.0, maximum=1.0),
        episode_end_probability=0, min_duration=5, max_duration=5)
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
    constructor = functools.partial(
        MockEnvironmentCrashInStep, crash_at_step=3)
    env = parallel_py_environment.ProcessPyEnvironment(constructor)
    env.start()
    env.reset()
    action_spec = env.action_spec()
    rng = np.random.RandomState()
    env.step(array_spec.sample_bounded_spec(action_spec, rng))
    env.step(array_spec.sample_bounded_spec(action_spec, rng))
    with self.assertRaises(Exception):
      env.step(array_spec.sample_bounded_spec(action_spec, rng))


class MockEnvironmentCrashInInit(object):
  """Raise an error when instantiated."""

  def __init__(self, *unused_args, **unused_kwargs):
    raise RuntimeError()

  def action_spec(self):
    return []


class MockEnvironmentCrashInReset(object):
  """Raise an error when instantiated."""

  def __init__(self, *unused_args, **unused_kwargs):
    pass

  def action_spec(self):
    return []

  def _reset(self):
    raise RuntimeError()


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
  tf.test.main()
