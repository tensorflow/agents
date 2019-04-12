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

"""Tests for the parallel environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import numpy as np
import tensorflow as tf

from tf_agents.environments import batched_py_environment
from tf_agents.environments import random_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class BatchedPyEnvironmentTest(tf.test.TestCase):

  @property
  def action_spec(self):
    return array_spec.BoundedArraySpec(
        [7], dtype=np.float32, minimum=-1.0, maximum=1.0)

  @property
  def observation_spec(self):
    return array_spec.ArraySpec((3, 3), np.float32)

  def _make_batched_py_environment(self, num_envs=3):
    self.time_step_spec = ts.time_step_spec(self.observation_spec)
    constructor = functools.partial(random_py_environment.RandomPyEnvironment,
                                    self.observation_spec, self.action_spec)
    return batched_py_environment.BatchedPyEnvironment(
        envs=[constructor() for _ in range(num_envs)])

  def test_close_no_hang_after_init(self):
    env = self._make_batched_py_environment()
    env.close()

  def test_get_specs(self):
    env = self._make_batched_py_environment()
    self.assertEqual(self.observation_spec, env.observation_spec())
    self.assertEqual(self.time_step_spec, env.time_step_spec())
    self.assertEqual(self.action_spec, env.action_spec())

    env.close()

  def test_step(self):
    num_envs = 5
    env = self._make_batched_py_environment(num_envs=num_envs)
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    rng = np.random.RandomState()
    action = np.stack([
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

  def test_unstack_actions(self):
    num_envs = 5
    action_spec = self.action_spec
    rng = np.random.RandomState()
    batched_action = np.array([
        array_spec.sample_bounded_spec(action_spec, rng)
        for _ in range(num_envs)
    ])

    # Test that actions are correctly unstacked when just batched in np.array.
    unstacked_actions = batched_py_environment.unstack_actions(batched_action)
    for action in unstacked_actions:
      self.assertAllEqual(action_spec.shape, action.shape)

  def test_unstack_nested_actions(self):
    num_envs = 5
    action_spec = self.action_spec
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
    unstacked_actions = batched_py_environment.unstack_actions(nested_action)
    for nested_action in unstacked_actions:
      self.assertAllEqual(action_spec.shape, nested_action.action.shape)
      self.assertEqual(13.0, nested_action.other_var)


if __name__ == '__main__':
  tf.test.main()
