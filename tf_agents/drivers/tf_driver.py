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

"""A Driver that steps a TF environment using a TF policy."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import driver
from tf_agents.environments import tf_environment
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common

from typing import Any, Callable, Optional, Sequence, Tuple


class TFDriver(driver.Driver):
  """A driver that runs a TF policy in a TF environment."""

  def __init__(
      self,
      env: tf_environment.TFEnvironment,
      policy: tf_policy.TFPolicy,
      observers: Sequence[Callable[[trajectory.Trajectory], Any]],
      transition_observers: Optional[Sequence[Callable[[trajectory.Transition],
                                                       Any]]] = None,
      max_steps: Optional[types.Int] = None,
      max_episodes: Optional[types.Int] = None,
      disable_tf_function: bool = False):
    """A driver that runs a TF policy in a TF environment.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.TFPolicy policy.
      observers: A list of observers that are notified after every step
        in the environment. Each observer is a callable(trajectory.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)). The transition is shaped just as
        trajectories are for regular observers.
      max_steps: Optional maximum number of steps for each run() call. For
        batched or parallel environments, this is the maximum total number of
        steps summed across all environments. Also see below.  Default: 0.
      max_episodes: Optional maximum number of episodes for each run() call. For
        batched or parallel environments, this is the maximum total number of
        episodes summed across all environments. At least one of max_steps or
        max_episodes must be provided. If both are set, run() terminates when at
        least one of the conditions is
        satisfied.  Default: 0.
      disable_tf_function: If True the use of tf.function for the run method is
        disabled.

    Raises:
      ValueError: If both max_steps and max_episodes are None.
    """
    common.check_tf1_allowed()
    max_steps = max_steps or 0
    max_episodes = max_episodes or 0
    if max_steps < 1 and max_episodes < 1:
      raise ValueError(
          'Either `max_steps` or `max_episodes` should be greater than 0.')

    super(TFDriver, self).__init__(env, policy, observers, transition_observers)

    self._max_steps = max_steps or np.inf
    self._max_episodes = max_episodes or np.inf

    if not disable_tf_function:
      self.run = common.function(self.run, autograph=True)

  def run(
      self, time_step: ts.TimeStep,
      policy_state: types.NestedTensor = ()
  ) -> Tuple[ts.TimeStep, types.NestedTensor]:
    """Run policy in environment given initial time_step and policy_state.

    Args:
      time_step: The initial time_step.
      policy_state: The initial policy_state.

    Returns:
      A tuple (final time_step, final policy_state).
    """
    num_steps = tf.constant(0.0)
    num_episodes = tf.constant(0.0)

    while num_steps < self._max_steps and num_episodes < self._max_episodes:
      action_step = self.policy.action(time_step, policy_state)
      next_time_step = self.env.step(action_step.action)

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      for observer in self._transition_observers:
        observer((time_step, action_step, next_time_step))
      for observer in self.observers:
        observer(traj)

      num_episodes += tf.math.reduce_sum(tf.cast(traj.is_last(), tf.float32))
      num_steps += tf.math.reduce_sum(tf.cast(~traj.is_boundary(), tf.float32))

      time_step = next_time_step
      policy_state = action_step.state

    return time_step, policy_state
