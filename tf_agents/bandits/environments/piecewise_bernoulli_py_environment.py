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

"""Class implementation of Python Bernoulli Bandit environment."""

from typing import Callable, Optional

import gin
import numpy as np

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.specs import array_spec
from tf_agents.typing import types


@gin.configurable
class PiecewiseBernoulliPyEnvironment(
    bandit_py_environment.BanditPyEnvironment):
  """Implements piecewise stationary finite-armed Bernoulli Bandits.

  This environment implements piecewise stationary finite-armed non-contextual
  Bernoulli Bandit environment as a subclass of BanditPyEnvironment.
  With respect to Bernoulli stationary environment, the reward distribution
  parameters undergo abrupt changes at given time steps. The current time is
  kept by the environment and increased by a unit at each call of _apply_action.
  For each stationary piece, the reward distribution is 0/1 (Bernoulli) with
  the parameter p valid for the current piece.

  Examples:

  means = [[0.1, 0.5], [0.5, 0.1], [0.5, 0.5]]  # 3 pieces, 2 arms.

  def constant_duration_gen(delta):
    while True:
      yield delta

  env_piecewise_10_steps = PiecewiseBernoulliPyEnvironment(
    means, constant_duration_gen(10))

  def random_duration_gen(a, b):
     while True:
       yield random.randint(a, b)

  env_rnd_piecewise_10_to_20_steps =  PiecewiseBernoulliPyEnvironment(
    means, random_duration_gen(10, 20))

  For a reference on bandits see e.g., Example 1.1 in "A Tutorial on Thompson
  Sampling" by Russo et al. (https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
  A paper using piecewise stationary environments is Qingyun Wu, Naveen Iyer,
  Hongning Wang, ``Learning Contextual Bandits in a Non-stationary
  Environment,'' Proceedings of the 2017 ACM on Conference on Information and
  Knowledge Management (https://arxiv.org/pdf/1805.09365.pdf.)
  """

  def __init__(self,
               piece_means: np.ndarray,
               change_duration_generator: Callable[[], int],
               batch_size: Optional[int] = 1):
    """Initializes a piecewise stationary Bernoulli Bandit environment.

    Args:
      piece_means: a matrix (list of lists) with shape (num_pieces, num_arms)
        containing floats in [0, 1]. Each list contains the mean rewards for
        the num_arms actions of the num_pieces pieces. The list is wrapped
        around after the last piece.
      change_duration_generator: a generator of the time durations. If this
        yields the values d0, d1, d2, ..., then the reward parameters change at
        steps d0, d0 + d1, d0 + d1 + d2, ..., as following:

        piece_means[0] for 0 <= t < d0
        piece_means[1] for d0 <= t < d0 + d1
        piece_means[2] for d0 + d1 <= t < d0 + d1 + d2
        ...

        Note that the values generated have to be non-negative. The value zero
        means that the corresponding parameters in the piece_means list are
        skipped, i.e. the duration of the piece is zero steps.
        If the generator ends (e.g. if it is obtained with iter(<list>)) and the
        step goes beyond the last piece, a StopIteration exception is raised.
      batch_size: If specified, this is the batch size for observation and
        actions.
    """
    self._batch_size = batch_size
    self._piece_means = np.asarray(piece_means, dtype=np.float32)
    if np.any(self._piece_means > 1.0) or np.any(self._piece_means < 0):
      raise ValueError('All parameters should be floats in [0, 1].')
    self._num_pieces, self._num_actions = self._piece_means.shape
    self._change_duration_generator = change_duration_generator
    self._current_time = -1
    self._current_piece = -1
    self._next_change = 0
    self._increment_time()

    action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=self._num_actions - 1,
        name='action')
    observation_spec = array_spec.ArraySpec(
        shape=(1,), dtype=np.int32, name='observation')
    super(PiecewiseBernoulliPyEnvironment, self).__init__(
        observation_spec, action_spec)

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def batched(self) -> bool:
    return True

  def _increment_time(self):
    self._current_time += 1
    while self._current_time >= self._next_change:
      duration = int(next(self._change_duration_generator))
      if duration < 0:
        raise ValueError(
            'Generated duration must be non-negative. Got {}.'.format(duration))
      self._next_change += duration
      self._current_piece = (self._current_piece + 1) % self._num_pieces

  def _observe(self) -> types.NestedArray:
    return np.zeros(
        shape=[self._batch_size, 1],
        dtype=self.observation_spec().dtype)

  def _apply_action(self, action: types.NestedArray) -> types.NestedArray:
    reward = np.floor(self._piece_means[self._current_piece, action] +
                      np.random.random((self._batch_size,)))
    self._increment_time()
    return reward
