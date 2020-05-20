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

"""A Driver that steps a python environment using a python policy."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import numpy as np
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from tf_agents.typing import types
from typing import Any, Callable, Optional, Sequence, Tuple


class PyDriver(driver.Driver):
  """A driver that runs a python policy in a python environment."""

  def __init__(
      self,
      env: py_environment.PyEnvironment,
      policy: py_policy.PyPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[types.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = None,
      max_episodes: Optional[types.Int] = None):
    """A driver that runs a python policy in a python environment.

    Args:
      env: A py_environment.Base environment.
      policy: A py_policy.PyPolicy policy.
      observers: A list of observers that are notified after every step
        in the environment. Each observer is a callable(trajectory.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)). The transition is shaped just as
        trajectories are for regular observers.
      max_steps: Optional maximum number of steps for each run() call.
        Also see below.  Default: 0.
      max_episodes: Optional maximum number of episodes for each run() call.
        At least one of max_steps or max_episodes must be provided. If both
        are set, run() terminates when at least one of the conditions is
        satisfied.  Default: 0.

    Raises:
      ValueError: If both max_steps and max_episodes are None.
    """
    max_steps = max_steps or 0
    max_episodes = max_episodes or 0
    if max_steps < 1 and max_episodes < 1:
      raise ValueError(
          'Either `max_steps` or `max_episodes` should be greater than 0.')

    super(PyDriver, self).__init__(env, policy, observers, transition_observers)
    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf

  def run(
      self,
      time_step: ts.TimeStep,
      policy_state: types.NestedArray = ()
  ) -> Tuple[ts.TimeStep, types.NestedArray]:
    """Run policy in environment given initial time_step and policy_state.

    Args:
      time_step: The initial time_step.
      policy_state: The initial policy_state.

    Returns:
      A tuple (final time_step, final policy_state).
    """
    num_steps = 0
    num_episodes = 0
    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      action_step = self.policy.action(time_step, policy_state)
      next_time_step = self.env.step(action_step.action)

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      for observer in self._transition_observers:
        observer((time_step, action_step, next_time_step))
      for observer in self.observers:
        observer(traj)

      num_episodes += np.sum(traj.is_last())
      num_steps += np.sum(~traj.is_boundary())

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state
