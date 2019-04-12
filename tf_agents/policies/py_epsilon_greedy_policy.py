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

"""epsilon-greedy policy in python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.trajectories import policy_step


class EpsilonGreedyPolicy(py_policy.Base):
  """Implementation of the epsilon-greedy policy."""

  def __init__(self, greedy_policy,
               epsilon,
               random_policy=None,
               epsilon_decay_end_count=None,
               epsilon_decay_end_value=None,
               random_seed=None):
    """Initializes the epsilon-greedy policy.

    Args:

      greedy_policy: An instance of py_policy.Base to use as the greedy policy.

      epsilon: The probability 0.0 <= epsilon <= 1.0 with which an
        action will be selected at random.

      random_policy: An instance of random_py_policy.RandomPyPolicy to
        use as the random policy, if None is provided, a
        RandomPyPolicy will be automatically created with the
        greedy_policy's action_spec and observation_spec and
        random_seed.

      epsilon_decay_end_count: if set, anneal the epislon every time
        this policy is used, until it hits the epsilon_decay_end_value.

      epsilon_decay_end_value: the value of epislon to use when the
        policy usage count hits epsilon_decay_end_count.

      random_seed: seed used to create numpy.random.RandomState.
        /dev/urandom will be used if it's None.

    Raises:

      ValueError: If epsilon is not between 0.0 and 1.0. Or if
      epsilon_decay_end_value is invalid when epsilon_decay_end_count is
      set.
    """
    if not 0 <= epsilon <= 1.0:
      raise ValueError('epsilon should be in [0.0, 1.0]')

    self._greedy_policy = greedy_policy
    if random_policy is None:
      self._random_policy = random_py_policy.RandomPyPolicy(
          time_step_spec=greedy_policy.time_step_spec,
          action_spec=greedy_policy.action_spec,
          seed=random_seed)
    else:
      self._random_policy = random_policy
    # TODO(b/110841809) consider making epsilon be provided by a function.
    self._epsilon = epsilon
    self._epsilon_decay_end_count = epsilon_decay_end_count
    if epsilon_decay_end_count is not None:
      if epsilon_decay_end_value is None or epsilon_decay_end_value >= epsilon:
        raise ValueError('Invalid value for epsilon_decay_end_value {}'.format(
            epsilon_decay_end_value))
      self._epsilon_decay_step_factor = float(
          epsilon - epsilon_decay_end_value) / epsilon_decay_end_count
    self._epsilon_decay_end_value = epsilon_decay_end_value

    self._random_seed = random_seed  # Keep it for copy method.
    self._rng = np.random.RandomState(random_seed)

    # Total times action method has been called.
    self._count = 0

  def _get_initial_state(self, batch_size):
    self._random_policy.reset(batch_size=batch_size)
    return self._greedy_policy.reset(batch_size=batch_size)

  def _get_epsilon(self):
    if self._epsilon_decay_end_count is not None:
      if self._count >= self._epsilon_decay_end_count:
        return self._epsilon_decay_end_value
      else:
        return (self._epsilon - (self._count - 1) *
                self._epsilon_decay_step_factor)
    else:
      return self._epsilon

  def _random_function(self):
    return self._rng.rand()

  def _action(self, time_step, policy_state=()):
    self._count += 1
    # _random_function()'s range should be [0, 1), so if epsilon is 1,
    # we should always use random policy, and if epislon is 0, it
    # should always use greedy_policy since the if condition won't be
    # met.
    if self._random_function() < self._get_epsilon():
      # Avoid mixing policy_state from greedy_policy and random_policy,
      # always return policy_state from greedy_policy.
      action_step = self._random_policy.action(time_step)
      return policy_step.PolicyStep(action_step.action, policy_state)
    else:
      return self._greedy_policy.action(time_step, policy_state=policy_state)
