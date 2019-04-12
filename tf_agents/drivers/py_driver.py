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
from __future__ import print_function

import numpy as np
from tf_agents.drivers import driver
from tf_agents.trajectories import trajectory


class PyDriver(driver.Driver):
  """A driver that runs a python policy in a python environment."""

  def __init__(self,
               env,
               policy,
               observers,
               max_steps=None,
               max_episodes=None):
    """A driver that runs a python policy in a python environment.

    Args:
      env: A py_environment.Base environment.
      policy: A py_policy.Base policy.
      observers: A list of observers that are notified after every step
        in the environment. Each observer is a callable(trajectory.Trajectory).
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

    super(PyDriver, self).__init__(env, policy, observers)
    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf

  def run(self, time_step, policy_state=()):
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
      for observer in self.observers:
        observer(traj)

      num_episodes += np.sum(traj.is_last())
      num_steps += np.sum(~traj.is_boundary())

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state
