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

"""Tests for utils.random_py_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from tf_agents.environments import random_py_environment
from tf_agents.specs import array_spec


class RandomPyEnvironmentTest(parameterized.TestCase, absltest.TestCase):

  def testEnvResetAutomatically(self):
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(obs_spec)

    time_step = env.step([0])
    self.assertTrue(np.all(time_step.observation >= -10))
    self.assertTrue(np.all(time_step.observation <= 10))
    self.assertTrue(time_step.is_first())

    while not time_step.is_last():
      time_step = env.step([0])
      self.assertTrue(np.all(time_step.observation >= -10))
      self.assertTrue(np.all(time_step.observation <= 10))

    time_step = env.step([0])
    self.assertTrue(np.all(time_step.observation >= -10))
    self.assertTrue(np.all(time_step.observation <= 10))
    self.assertTrue(time_step.is_first())

  @parameterized.named_parameters([
      ('OneStep', 1),
      ('FiveSteps', 5),
  ])
  def testEnvMinDuration(self, min_duration):
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        obs_spec, episode_end_probability=0.9, min_duration=min_duration)
    num_episodes = 100

    for _ in range(num_episodes):
      time_step = env.step([0])
      self.assertTrue(time_step.is_first())
      num_steps = 0
      while not time_step.is_last():
        time_step = env.step([0])
        num_steps += 1
      self.assertGreaterEqual(num_steps, min_duration)

  @parameterized.named_parameters([
      ('OneStep', 1),
      ('FiveSteps', 5),
  ])
  def testEnvMaxDuration(self, max_duration):
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        obs_spec, episode_end_probability=0.1, max_duration=max_duration)
    num_episodes = 100

    for _ in range(num_episodes):
      time_step = env.step([0])
      self.assertTrue(time_step.is_first())
      num_steps = 0
      while not time_step.is_last():
        time_step = env.step([0])
        num_steps += 1
      self.assertLessEqual(num_steps, max_duration)

  def testEnvChecksActions(self):
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    action_spec = array_spec.BoundedArraySpec((2, 2), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        obs_spec, action_spec=action_spec)

    env.step(np.array([[0, 0], [0, 0]]))

    with self.assertRaises(ValueError):
      env.step([0])

  def testRewardFnCalled(self):

    def reward_fn(unused_step_type, action, unused_observation):
      return action

    action_spec = array_spec.BoundedArraySpec((1,), np.int32, -10, 10)
    observation_spec = array_spec.BoundedArraySpec((1,), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        observation_spec, action_spec, reward_fn=reward_fn)

    time_step = env.step(1)  # No reward in first time_step
    self.assertEqual(0.0, time_step.reward)
    time_step = env.step(1)
    self.assertEqual(1, time_step.reward)

  def testRendersImage(self):
    action_spec = array_spec.BoundedArraySpec((1,), np.int32, -10, 10)
    observation_spec = array_spec.BoundedArraySpec((1,), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        observation_spec, action_spec, render_size=(4, 4, 3))

    env.reset()
    img = env.render()

    self.assertTrue(np.all(img < 256))
    self.assertTrue(np.all(img >= 0))
    self.assertEqual((4, 4, 3), img.shape)
    self.assertEqual(np.uint8, img.dtype)

  def testBatchSize(self):
    batch_size = 3
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(obs_spec,
                                                    batch_size=batch_size)

    time_step = env.step([0])
    self.assertEqual(time_step.observation.shape, (3, 2, 3))
    self.assertEqual(time_step.reward.shape[0], batch_size)
    self.assertEqual(time_step.discount.shape[0], batch_size)

  def testCustomRewardFn(self):
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    batch_size = 3
    env = random_py_environment.RandomPyEnvironment(
        obs_spec,
        reward_fn=lambda *_: np.ones(batch_size),
        batch_size=batch_size)
    env._done = False
    env.reset()
    time_step = env.step([0])
    self.assertSequenceAlmostEqual([1.0] * 3, time_step.reward)

  def testRewardCheckerBatchSizeOne(self):
    # Ensure batch size 1 with scalar reward works
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        obs_spec,
        reward_fn=lambda *_: np.array([1.0]),
        batch_size=1)
    env._done = False
    env.reset()
    time_step = env.step([0])
    self.assertEqual(time_step.reward, 1.0)

  def testRewardCheckerSizeMismatch(self):
    # Ensure custom scalar reward with batch_size greater than 1 raises
    # ValueError
    obs_spec = array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        obs_spec,
        reward_fn=lambda *_: 1.0,
        batch_size=5)
    env.reset()
    env._done = False
    with self.assertRaises(ValueError):
      env.step([0])


if __name__ == '__main__':
  absltest.main()
