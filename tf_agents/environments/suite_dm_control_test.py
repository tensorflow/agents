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

"""Tests for dm_control_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import suite_dm_control
from tf_agents.environments import utils


class SuiteDMControlTest(absltest.TestCase):

  def setUp(self):
    super(SuiteDMControlTest, self).setUp()
    if not suite_dm_control.is_available():
      self.skipTest('dm_control is not available.')

  def testEnvRegistered(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    self.assertIsInstance(env, py_environment.PyEnvironment)

    utils.validate_py_environment(env)

  def testObservationSpec(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    obs_spec = env.observation_spec()
    self.assertEqual(np.float64, obs_spec['position'].dtype)
    self.assertEqual((4,), obs_spec['position'].shape)

  def testActionSpec(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    obs_spec = env.observation_spec()
    self.assertEqual(np.float64, obs_spec['position'].dtype)
    self.assertEqual((4,), obs_spec['position'].shape)


if __name__ == '__main__':
  absltest.main()
