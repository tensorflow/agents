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

"""Tests for tf_agents.google.environments.dm_control_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from tf_agents.environments import suite_dm_control


class DmControlWrapperTest(absltest.TestCase):

  def setUp(self):
    if not suite_dm_control.is_available():
      self.skipTest('dm_control is not available.')

  def test_wrapped_cartpole_specs(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')

    action_spec = env.action_spec()
    self.assertEqual((2,), action_spec.shape)
    np.testing.assert_array_almost_equal([-1.0, -1.0], action_spec.minimum)
    np.testing.assert_array_almost_equal([1.0, 1.0], action_spec.maximum)

    observation_spec = env.observation_spec()
    self.assertEqual((4,), observation_spec['position'].shape)
    self.assertEqual((4,), observation_spec['velocity'].shape)

  def test_reset(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')

    first_time_step = env.reset()
    self.assertTrue(first_time_step.is_first())
    self.assertEqual(0.0, first_time_step.reward)
    self.assertEqual(1.0, first_time_step.discount)

  def test_transition(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    env.reset()
    transition_time_step = env.step([0, 0])

    self.assertTrue(transition_time_step.is_mid())
    self.assertNotEqual(None, transition_time_step.reward)
    self.assertEqual(1.0, transition_time_step.discount)

  def test_wrapped_cartpole_final(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    time_step = env.reset()

    while not time_step.is_last():
      time_step = env.step([1, 1])

    self.assertTrue(time_step.is_last())
    self.assertNotEqual(None, time_step.reward)
    # Discount is 1.0 as it's an infinite horizon task that DM is terminating
    # early.
    self.assertEqual(1.0, time_step.discount)

  def test_automatic_reset_after_create(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')

    first_time_step = env.step([0, 0])
    self.assertTrue(first_time_step.is_first())

  def test_automatic_reset_after_done(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    time_step = env.reset()

    while not time_step.is_last():
      time_step = env.step([0, 0])

    self.assertTrue(time_step.is_last())
    first_time_step = env.step([0, 0])
    self.assertTrue(first_time_step.is_first())

  def test_automatic_reset_after_done_not_using_reset_directly(self):
    env = suite_dm_control.load('ball_in_cup', 'catch')
    time_step = env.step([0, 0])

    while not time_step.is_last():
      time_step = env.step([0, 0])

    self.assertTrue(time_step.is_last())
    first_time_step = env.step([0, 0])
    self.assertTrue(first_time_step.is_first())


if __name__ == '__main__':
  absltest.main()
