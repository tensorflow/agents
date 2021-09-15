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

"""Test for tf_agents.environments.suite_gym."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing.absltest import mock

import gin
import gym

from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.utils import test_utils


class SuiteGymTest(test_utils.TestCase):

  def tearDown(self):
    gin.clear_config()
    super(SuiteGymTest, self).tearDown()

  def test_load_adds_time_limit_steps(self):
    env = suite_gym.load('CartPole-v1')
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)

  def test_load_disable_step_limit(self):
    env = suite_gym.load('CartPole-v1', max_episode_steps=0)
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertNotIsInstance(env, wrappers.TimeLimit)

  def test_load_disable_wrappers_applied(self):
    duration_wrapper = functools.partial(wrappers.TimeLimit, duration=10)
    env = suite_gym.load(
        'CartPole-v1', max_episode_steps=0, env_wrappers=(duration_wrapper,))
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)

  def test_custom_max_steps(self):
    env = suite_gym.load('CartPole-v1', max_episode_steps=5)
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)
    self.assertEqual(5, env._duration)

  def testGinConfig(self):
    gin.parse_config_file(
        test_utils.test_src_dir_path('environments/configs/suite_gym.gin')
    )
    env = suite_gym.load()
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)

  def test_gym_kwargs_argument(self):
    env = suite_gym.load('MountainCar-v0', gym_kwargs={'goal_velocity': 21})
    self.assertEqual(env.unwrapped.goal_velocity, 21)

    env = suite_gym.load('MountainCar-v0',
                         gym_kwargs={'goal_velocity': 50})
    self.assertEqual(env.unwrapped.goal_velocity, 50)

  def test_load_gym_render_kwargs(self):
    env = suite_gym.load('CartPole-v1',
                         render_kwargs={'width': 96, 'height': 128})
    gym_env = env.gym
    self.assertIsInstance(gym_env, gym.Env)
    gym_env.render = mock.MagicMock()
    # render_kwargs should be passed to the underlying gym env's render().
    env.render()
    gym_env.render.assert_called_with('rgb_array', width=96, height=128)

  def test_wrap_gym_render_kwargs(self):
    gym_env = gym.make('CartPole-v1')
    env = suite_gym.wrap_env(gym_env,
                             render_kwargs={'width': 96, 'height': 128})
    gym_env.render = mock.MagicMock()
    # render_kwargs should be passed to them underlying gym env's render().
    env.render()
    gym_env.render.assert_called_with('rgb_array', width=96, height=128)


if __name__ == '__main__':
  test_utils.main()
