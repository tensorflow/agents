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
"""Tests for tf_agents.environments.examples.tic_tac_toe_environment."""

import numpy as np

from tf_agents.environments import utils as env_utils
from tf_agents.environments.examples.tic_tac_toe_environment import TicTacToeEnvironment
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.utils import test_utils


class TicTacToeEnvironmentTest(test_utils.TestCase):

  def setUp(self):
    super(TicTacToeEnvironmentTest, self).setUp()
    np.random.seed(0)
    self.discount = np.asarray(1., dtype=np.float32)
    self.env = TicTacToeEnvironment()
    ts = self.env.reset()
    np.testing.assert_array_equal(np.zeros((3, 3), np.int32), ts.observation)

  def test_validate_specs(self):
    env_utils.validate_py_environment(self.env, episodes=10)

  def test_check_states(self):
    self.assertEqual(
        (False, 0.),
        self.env._check_states(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])))
    self.assertEqual(
        (True, -1.),
        self.env._check_states(np.array([[2, 2, 2], [0, 1, 1], [0, 0, 0]])))
    self.assertEqual(
        (True, 1.),
        self.env._check_states(np.array([[2, 2, 0], [1, 1, 1], [0, 0, 0]])))
    self.assertEqual(
        (False, 0.),
        self.env._check_states(np.array([[2, 2, 1], [1, 2, 1], [1, 0, 0]])))
    self.assertEqual(
        (True, 0.),
        self.env._check_states(np.array([[2, 1, 2], [1, 2, 1], [1, 2, 1]])))

  def test_legal_actions(self):
    states = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]])
    self.assertEqual([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
                     self.env._legal_actions(states))

  def test_opponent_play_deterministic(self):
    # Chooses the first available space.
    self.assertEqual((0, 0),
                     self.env._opponent_play([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))
    self.assertEqual((2, 2),
                     self.env._opponent_play([[1, 1, 1], [1, 1, 1], [1, 1, 0]]))

  def test_opponent_play_random(self):
    self.env = TicTacToeEnvironment(rng=np.random.RandomState(0))
    s = set()
    states = np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]])
    legal_actions = self.env._legal_actions(states)

    # Make sure that each legal action has been played.
    for _ in range(100):
      s.add(self.env._opponent_play(states))
    self.assertEqual(set(legal_actions), s)

  def test_step_win(self):
    self.env.set_state(
        TimeStep(StepType.MID, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL,
                 self.discount, np.array([[2, 2, 0], [0, 1, 1], [0, 0, 0]])))

    current_time_step = self.env.current_time_step()
    self.assertEqual(StepType.MID, current_time_step.step_type)

    ts = self.env.step(np.array([1, 0]))

    np.testing.assert_array_equal([[2, 2, 0], [1, 1, 1], [0, 0, 0]],
                                  ts.observation)
    self.assertEqual(StepType.LAST, ts.step_type)
    self.assertEqual(1., ts.reward)

    # Reset if an action is taken after final state is reached.
    ts = self.env.step(np.array([2, 0]))
    self.assertEqual(StepType.FIRST, ts.step_type)
    self.assertEqual(0., ts.reward)

  def test_step_loss(self):
    self.env.set_state(
        TimeStep(StepType.MID, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL,
                 self.discount, np.array([[2, 2, 0], [0, 1, 1], [0, 0, 0]])))

    current_time_step = self.env.current_time_step()
    self.assertEqual(StepType.MID, current_time_step.step_type)

    ts = self.env.step(np.array([2, 0]))

    np.testing.assert_array_equal([[2, 2, 2], [0, 1, 1], [1, 0, 0]],
                                  ts.observation)
    self.assertEqual(StepType.LAST, ts.step_type)
    self.assertEqual(-1., ts.reward)

    # Reset if an action is taken after final state is reached.
    ts = self.env.step(np.array([2, 0]))
    self.assertEqual(StepType.FIRST, ts.step_type)
    self.assertEqual(0., ts.reward)

  def test_step_illegal_move(self):
    self.env.set_state(
        TimeStep(StepType.MID, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL,
                 self.discount, np.array([[2, 2, 0], [0, 1, 1], [0, 0, 0]])))

    current_time_step = self.env.current_time_step()
    self.assertEqual(StepType.MID, current_time_step.step_type)

    # Taking an illegal move.
    ts = self.env.step(np.array([0, 0]))

    np.testing.assert_array_equal([[2, 2, 0], [0, 1, 1], [0, 0, 0]],
                                  ts.observation)
    self.assertEqual(StepType.LAST, ts.step_type)
    self.assertEqual(TicTacToeEnvironment.REWARD_ILLEGAL_MOVE, ts.reward)

    # Reset if an action is taken after final state is reached.
    ts = self.env.step(np.array([2, 0]))
    self.assertEqual(StepType.FIRST, ts.step_type)
    self.assertEqual(0., ts.reward)


if __name__ == '__main__':
  test_utils.main()
