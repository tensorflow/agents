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

"""Tests for tf_agents.environments.suite_bsuite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tf_agents.environments import py_environment
from tf_agents.environments import suite_bsuite
from tf_agents.utils import test_utils


class SuiteBsuiteTest(test_utils.TestCase):

  def setUp(self):
    super(SuiteBsuiteTest, self).setUp()
    if not suite_bsuite.is_available():
      self.skipTest('bsuite is not available.')

  def tearDown(self):
    gin.clear_config()
    super(SuiteBsuiteTest, self).tearDown()

  def testBsuiteEnvRegisteredWithRecord(self):
    env = suite_bsuite.load(
        'deep_sea/0', record=True, save_path=None, logging_mode='terminal')
    self.assertIsInstance(env, py_environment.PyEnvironment)

  def testBsuiteEnvRegistered(self):
    env = suite_bsuite.load(
        'deep_sea/0', record=False)
    self.assertIsInstance(env, py_environment.PyEnvironment)

  def testGinConfig(self):
    gin.parse_config_file(
        test_utils.test_src_dir_path('environments/configs/suite_bsuite.gin')
    )
    env = suite_bsuite.load()
    self.assertIsInstance(env, py_environment.PyEnvironment)


if __name__ == '__main__':
  test_utils.main()
