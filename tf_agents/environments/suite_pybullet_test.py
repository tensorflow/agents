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

"""Tests tf_agents.environments.suite_pybullet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tf_agents.environments import py_environment
from tf_agents.environments import suite_pybullet
from tf_agents.environments import wrappers
from tf_agents.utils import test_utils


class SuitePybulletTest(test_utils.TestCase):

  def tearDown(self):
    gin.clear_config()
    super(SuitePybulletTest, self).tearDown()

  def testPybulletEnvRegistered(self):
    env = suite_pybullet.load('InvertedPendulumBulletEnv-v0')
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)

  def testGinConfig(self):
    gin.parse_config_file(
        test_utils.test_src_dir_path('environments/configs/suite_pybullet.gin')
    )
    env = suite_pybullet.load()
    self.assertIsInstance(env, py_environment.PyEnvironment)
    self.assertIsInstance(env, wrappers.TimeLimit)


if __name__ == '__main__':
  test_utils.main()
