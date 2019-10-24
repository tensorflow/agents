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

"""Class implementation of Python Wheel Bandit environment."""
from __future__ import absolute_import

import gin
import numpy as np
from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

_NUM_ACTIONS = 5
_CONTEXT_DIM = 2
_SIGNS_TO_OPT_ACTION = {
    (1.0, 1.0): 1,
    (1.0, -1.0): 2,
    (-1.0, 1.0): 3,
    (-1.0, -1.0): 4,
}


@gin.configurable
def compute_optimal_action(observation, delta):
  batch_size = observation.shape[0]
  optimal_actions = np.zeros(batch_size, dtype=np.int32)
  is_outer = np.int32(np.linalg.norm(observation, ord=2, axis=1) > delta)
  signs = np.sign(observation)
  optimal_actions += is_outer * [_SIGNS_TO_OPT_ACTION[tuple(x)] for x in signs]
  return optimal_actions


@gin.configurable
def compute_optimal_reward(observation, delta, mu_inside, mu_high):
  is_inside = np.float32(np.linalg.norm(observation, ord=2, axis=1) <= delta)
  return is_inside * mu_inside + (1 - is_inside) * mu_high


@gin.configurable
class WheelPyEnvironment(bandit_py_environment.BanditPyEnvironment):
  """Implements the Wheel Bandit environment.

  This environment implements the wheel bandit from Section 5.4 of [1] (please
  see references below) as a subclass of BanditPyEnvironment.

  Context features are sampled uniformly at random in the unit circle in R^2.
  There are 5 possible actions. There exists an exploration parameter `delta`
  in (0, 1) that determines the difficulty of the problem and the need for
  exploration.

  References:
  [1]. Carlos Riquelme, George Tucker, Jasper Snoek
  "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep
  Networks for Thompson Sampling", International Conference on Learning
  Representations (ICLR) 2018.
  https://arxiv.org/abs/1802.09127

  """

  def __init__(self, delta, mu_base, std_base, mu_high, std_high,
               batch_size=None):
    """Initializes the Wheel Bandit environment.

    Args:
      delta: float in `(0, 1)`. Exploration parameter.
      mu_base: (vector of float) Mean reward for each action, if the context
          norm is below delta. The size of the vector is expected to be 5 (i.e.,
          equal to the number of actions.)
      std_base: (vector of float) std of the Gaussian reward for each action if
          the context norm is below delta. The size of the vector is expected to
          be 5 (i.e., equal to the number of actions.)
      mu_high: (float) Mean reward for the optimal action if the context norm
          is above delta.
      std_high: (float) Reward std for optimal action if the context norm is
          above delta.
      batch_size: (optional) (int) Number of observations generated per call.
    """
    self._batch_size = 1 if batch_size is None else batch_size
    if (delta <= 0 or delta >= 1):
      raise ValueError('Delta must be in (0, 1), but saw delta: %g' % (delta,))
    self._delta = delta
    self._mu_base = np.asarray(mu_base, dtype=np.float32)
    if self._mu_base.shape != (5,):
      raise ValueError('The length of \'mu_base\' must be 5, but saw '
                       '\'mu_base\': %s' % (self._mu_base,))
    self._std_base = np.asarray(std_base, dtype=np.float32)
    if self._std_base.shape != (5,):
      raise ValueError('The length of \'std_base\' must be 5.')

    self._mu_high = mu_high
    self._std_high = std_high

    # The first action should have higher mean reward that the other actions.
    if self._mu_base[0] != max(self._mu_base):
      raise ValueError('The first action in mu_base should have the highest '
                       'reward; got {}'.format(self._mu_base))

    action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=_NUM_ACTIONS - 1,
        name='action')
    observation_spec = array_spec.ArraySpec(
        shape=(_CONTEXT_DIM,), dtype=np.float32, name='observation')
    self._time_step_spec = ts.time_step_spec(observation_spec)
    self._observation = np.zeros((self._batch_size, _CONTEXT_DIM))
    super(WheelPyEnvironment, self).__init__(observation_spec, action_spec)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def batched(self):
    return True

  def _reward_fn(self, observation, action):
    # Sample rewards for all actions.
    r_all = np.random.normal(
        self._mu_base, self._std_base, size=(self._batch_size, _NUM_ACTIONS))

    # Compute the reward inside.
    row_norms = np.linalg.norm(observation, ord=2, axis=1)
    is_norm_below_delta = np.float32(row_norms <= self._delta)
    reward_inside = (
        is_norm_below_delta * r_all[np.arange(self._batch_size), action])

    # Compute the reward outside.
    high_reward = np.random.normal(
        self._mu_high, self._std_high, size=(self._batch_size))
    signs = np.sign(observation)
    optimal_actions = [_SIGNS_TO_OPT_ACTION[tuple(x)] for x in signs]
    r_outside = r_all
    r_outside[np.arange(self._batch_size), optimal_actions] = high_reward

    reward_outside = ((1.0 - is_norm_below_delta) *
                      r_outside[np.arange(self._batch_size), action])

    reward_final = reward_inside + reward_outside
    return reward_final

  def _observe(self):
    """Returns 2-dim samples falling in the unit circle."""
    theta = np.random.uniform(0.0, 2.0 * np.pi, (self._batch_size))
    r = np.sqrt(np.random.uniform(size=self._batch_size))
    batched_observations = np.stack(
        [r * np.cos(theta), r * np.sin(theta)], axis=1)
    self._observation = batched_observations.astype(
        self._observation_spec.dtype)
    return self._observation

  def _apply_action(self, action):
    """Computes the reward for the input actions."""
    return self._reward_fn(self._observation, action)
