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

"""Base class for drivers that takes steps in an environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Driver(object):
  """A driver that takes steps in an environment using a policy."""

  def __init__(self, env, policy, observers=None, transition_observers=None):
    """Creates a Driver.

    Args:
      env: An environment.Base environment.
      policy: A policy.Base policy.
      observers: A list of observers that are updated after the driver is run.
        Each observer is a callable(Trajectory) that returns the input.
        Trajectory.time_step is a stacked batch [N+1, batch_size, ...] of
        timesteps and Trajectory.action is a stacked batch
        [N, batch_size, ...] of actions in time major form.
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)). The transition is shaped just as
        trajectories are for regular observers.
    """

    self._env = env
    self._policy = policy
    self._observers = observers or []
    self._transition_observers = transition_observers or []

  @property
  def env(self):
    return self._env

  @property
  def policy(self):
    return self._policy

  @property
  def transition_observers(self):
    return self._transition_observers

  @property
  def observers(self):
    return self._observers

  @abc.abstractmethod
  def run(self):
    """Takes steps in the environment and updates observers."""
