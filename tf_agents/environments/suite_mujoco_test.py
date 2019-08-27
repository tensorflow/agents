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

"""Tests for tf_agents.environments.suite_mujoco."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import wrappers
from tf_agents.utils import test_utils


class SuiteMujocoTest(test_utils.TestCase):

  def setUp(self):
    super(SuiteMujocoTest, self).setUp()
    if not suite_mujoco.is_available():
      self.skipTest('suite_mujoco is not available.')

  def tearDown(self):
    gin.clear_config()
    super(SuiteMujocoTest, self).tearDown()

  def testMujocoEnvRegistered(self):
    env = suite_mujoco.load('HalfCheetah-v2')
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)

  def testObservationSpec(self):
    env = suite_mujoco.load('HalfCheetah-v2')
    self.assertEqual(np.float32, env.observation_spec().dtype)
    self.assertEqual((17,), env.observation_spec().shape)

  def testActionSpec(self):
    env = suite_mujoco.load('HalfCheetah-v2')
    self.assertEqual(np.float32, env.action_spec().dtype)
    self.assertEqual((6,), env.action_spec().shape)

  def testGinConfig(self):
    gin.parse_config_file(
        test_utils.test_src_dir_path('environments/configs/suite_mujoco.gin')
    )
    env = suite_mujoco.load()
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)


if __name__ == '__main__':
  test_utils.main()
