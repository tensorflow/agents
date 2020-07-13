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

# Lint as: python3
"""Tests for tf_agents.environments.test_envs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.environments import test_envs
from tf_agents.environments import utils as env_utils
from tf_agents.utils import test_utils


class CountingEnvTest(test_utils.TestCase):

  def test_sequential(self):
    num_episodes = 3
    steps_per_episode = 4
    env = test_envs.CountingEnv(steps_per_episode)

    for episode in range(num_episodes):
      step = 0
      time_step = env.reset()
      self.assertEqual(episode * 10 + step, time_step.observation)
      while not time_step.is_last():
        time_step = env.step(0)
        step += 1
        self.assertEqual(episode * 10 + step, time_step.observation)
      self.assertEqual(episode * 10 + steps_per_episode, time_step.observation)

  def test_validate_specs(self):
    env = test_envs.CountingEnv(steps_per_episode=15)
    env_utils.validate_py_environment(env, episodes=10)


class EpisodeCountingEnvTest(test_utils.TestCase):

  def test_sequential(self):
    num_episodes = 3
    steps_per_episode = 4
    env = test_envs.EpisodeCountingEnv(steps_per_episode=steps_per_episode)

    for episode in range(num_episodes):
      step = 0
      time_step = env.reset()
      self.assertAllEqual((episode, step), time_step.observation)
      while not time_step.is_last():
        time_step = env.step(0)
        step += 1
        self.assertAllEqual((episode, step), time_step.observation)
      self.assertAllEqual((episode, steps_per_episode), time_step.observation)

  def test_validate_specs(self):
    env = test_envs.EpisodeCountingEnv(steps_per_episode=15)
    env_utils.validate_py_environment(env, episodes=10)


class NestedCountingEnvTest(test_utils.TestCase):

  def test_sequential(self):
    num_episodes = 3
    steps_per_episode = 4
    env = test_envs.NestedCountingEnv(steps_per_episode)

    for episode in range(num_episodes):
      step = 0
      time_step = env.reset()
      self.assertCountEqual(
          {
              'total_steps': episode * 10 + step,
              'current_steps': step,
          }, time_step.observation)
      while not time_step.is_last():
        time_step = env.step(0)
        step += 1
        self.assertCountEqual(
            {
                'total_steps': episode * 10 + step,
                'current_steps': step,
            }, time_step.observation)
      self.assertCountEqual(
          {
              'total_steps': episode * 10 + steps_per_episode,
              'current_steps': steps_per_episode,
          }, time_step.observation)

  def test_validate_specs(self):
    env = test_envs.NestedCountingEnv(steps_per_episode=15)
    env_utils.validate_py_environment(env, episodes=10)


if __name__ == '__main__':
  test_utils.main()
