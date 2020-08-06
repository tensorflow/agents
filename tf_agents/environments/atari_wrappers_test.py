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

"""Tests for environments.atari_wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing.absltest import mock

from tf_agents.environments import atari_wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class AtariTimeLimitTest(test_utils.TestCase):

  def test_game_over_after_limit(self):
    max_steps = 5
    base_env = mock.MagicMock()
    wrapped_env = atari_wrappers.AtariTimeLimit(base_env, max_steps)

    base_env.gym.game_over = False
    base_env.reset.return_value = ts.restart(1)  # pytype: disable=wrong-arg-types
    base_env.step.return_value = ts.transition(2, 0)  # pytype: disable=wrong-arg-types
    action = 1

    self.assertFalse(wrapped_env.game_over)

    for _ in range(max_steps):
      time_step = wrapped_env.step(action)  # pytype: disable=wrong-arg-types
      self.assertFalse(time_step.is_last())
      self.assertFalse(wrapped_env.game_over)

    time_step = wrapped_env.step(action)  # pytype: disable=wrong-arg-types
    self.assertTrue(time_step.is_last())
    self.assertTrue(wrapped_env.game_over)

  def test_resets_after_limit(self):
    max_steps = 5
    base_env = mock.MagicMock()
    wrapped_env = atari_wrappers.AtariTimeLimit(base_env, max_steps)

    base_env.gym.game_over = False
    base_env.reset.return_value = ts.restart(1)  # pytype: disable=wrong-arg-types
    base_env.step.return_value = ts.transition(2, 0)  # pytype: disable=wrong-arg-types
    action = 1

    for _ in range(max_steps + 1):
      wrapped_env.step(action)  # pytype: disable=wrong-arg-types

    self.assertTrue(wrapped_env.game_over)
    self.assertEqual(1, base_env.reset.call_count)

    wrapped_env.step(action)  # pytype: disable=wrong-arg-types
    self.assertFalse(wrapped_env.game_over)
    self.assertEqual(2, base_env.reset.call_count)


if __name__ == '__main__':
  test_utils.main()
