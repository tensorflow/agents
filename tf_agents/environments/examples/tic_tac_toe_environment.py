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

"""A state-settable environment for Tic-Tac-Toe game."""

import copy
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep


class TicTacToeEnvironment(py_environment.PyEnvironment):
  """A state-settable environment for Tic-Tac-Toe game.

  For MCTS/AlphaZero, we need to keep states of the environment in a node and
  later restore them once MCTS selects which node to visit. This requires
  calling into get_state() and set_state() functions.

  The states are a 3 x 3 array where 0 = empty, 1 = player, 2 = opponent.
  The action is a 2-d vector to indicate the position for the player's move.
  """
  REWARD_WIN = np.asarray(1., dtype=np.float32)
  REWARD_LOSS = np.asarray(-1., dtype=np.float32)
  REWARD_DRAW_OR_NOT_FINAL = np.asarray(0., dtype=np.float32)
  # A very small number such that it does not affect the value calculation.
  REWARD_ILLEGAL_MOVE = np.asarray(-.001, dtype=np.float32)

  REWARD_WIN.setflags(write=False)
  REWARD_LOSS.setflags(write=False)
  REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)

  def __init__(self, rng: np.random.RandomState = None, discount=1.0):
    """Initializes TicTacToeEnvironment.

    Args:
      rng: If a random generator is provided, the opponent will choose a random
        empty space. If None is provided, the opponent will choose the first
        empty space.
      discount: Discount for reward.
    """
    super(TicTacToeEnvironment, self).__init__(handle_auto_reset=True)
    self._rng = rng
    self._discount = np.asarray(discount, dtype=np.float32)
    self._states = None

  def action_spec(self):
    return BoundedArraySpec((2,), np.int32, minimum=0, maximum=2)

  def observation_spec(self):
    return BoundedArraySpec((3, 3), np.int32, minimum=0, maximum=2)

  def _reset(self):
    self._states = np.zeros((3, 3), np.int32)
    return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                    self._discount, self._states)

  def _legal_actions(self, states: np.ndarray):
    return list(zip(*np.where(states == 0)))

  def _opponent_play(self, states: np.ndarray):
    actions = self._legal_actions(np.array(states))
    if not actions:
      raise RuntimeError('There is no empty space for opponent to play at.')

    if self._rng:
      i = self._rng.randint(len(actions))
    else:
      i = 0
    return actions[i]

  def get_state(self) -> TimeStep:
    # Returning an unmodifiable copy of the state.
    return copy.deepcopy(self._current_time_step)

  def set_state(self, time_step: TimeStep):
    self._current_time_step = time_step
    self._states = time_step.observation

  def _step(self, action: np.ndarray):
    action = tuple(action)
    if self._states[action] != 0:
      return TimeStep(StepType.LAST, TicTacToeEnvironment.REWARD_ILLEGAL_MOVE,
                      self._discount, self._states)

    self._states[action] = 1

    is_final, reward = self._check_states(self._states)
    if is_final:
      return TimeStep(StepType.LAST, reward, self._discount,
                      self._states)

    # TODO(b/152638947): handle multiple agents properly.
    # Opponent places '2' on the board.
    opponent_action = self._opponent_play(self._states)
    self._states[opponent_action] = 2

    is_final, reward = self._check_states(self._states)

    step_type = StepType.MID
    if np.all(self._states == 0):
      step_type = StepType.FIRST
    elif is_final:
      step_type = StepType.LAST

    return TimeStep(step_type, reward, self._discount, self._states)

  def _check_states(self, states: np.ndarray):
    """Check if the given states are final and calculate reward.

    Args:
      states: states of the board.

    Returns:
      A tuple of (is_final, reward) where is_final means whether the states
      are final are not, and reward is the reward for stepping into the states
      The meaning of reward: 0 = not decided or draw, 1 = win, -1 = loss
    """
    seqs = np.array([
        # each row
        states[0, :], states[1, :], states[2, :],
        # each column
        states[:, 0], states[:, 1], states[:, 2],
        # diagonal
        states[(0, 1, 2), (0, 1, 2)],
        states[(2, 1, 0), (0, 1, 2)],
    ])
    seqs = seqs.tolist()
    if [1, 1, 1] in seqs:
      return True, TicTacToeEnvironment.REWARD_WIN  # win
    if [2, 2, 2] in seqs:
      return True, TicTacToeEnvironment.REWARD_LOSS  # loss
    if 0 in states:
      # Not final
      return False, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL
    return True, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL  # draw
