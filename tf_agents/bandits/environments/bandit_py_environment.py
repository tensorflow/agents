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

"""Base class for Bandit Python environments."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
from typing import Optional
import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class BanditPyEnvironment(py_environment.PyEnvironment):
  """Base class for Bandit Python environments.

  Every bandit Python environment should derive from this class.
  Subclasses need to implement functions _observe() and _apply_action().

  Usage:

  To receive the first observation, the environment's reset() function should be
  called. To take an action, use the step(action) function. The time step
  returned by step(action) will contain the reward and the next observation.
  """

  def __init__(self,
               observation_spec: types.NestedArray,
               action_spec: types.NestedArray,
               reward_spec: Optional[types.NestedArray] = None):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._reward_spec = reward_spec
    super(BanditPyEnvironment, self).__init__()

  def _reset(self) -> ts.TimeStep:
    """Returns a time step containing an observation.

    It should not be overridden by Bandit environment implementations.

    Returns:
      A time step of type FIRST containing an observation.
    """
    return ts.restart(self._observe(), batch_size=self.batch_size,
                      reward_spec=self.reward_spec())

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Returns a time step containing the reward for the action taken.

    The returning time step also contains the next observation.
    It should not be overridden by bandit environment implementations.

    Args:
      action: The action taken by the Bandit policy.

    Returns:
      A time step of type LAST containing the reward for the action taken and
      the next observation.
    """
    # This step will take an action and return a reward.
    reward = self._apply_action(action)
    return ts.termination(self._observe(), reward)

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def reward_spec(self) -> types.NestedArraySpec:
    return self._reward_spec

  def _empty_observation(self):
    return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype),
                                 self.observation_spec())

  @abc.abstractmethod
  def _apply_action(self, action: types.NestedArray) -> types.Float:
    """Applies `action` to the Environment and returns the corresponding reward.

    Args:
      action: A value conforming action_spec that will be taken as action in the
        environment.

    Returns:
      A float value that is the reward received by the environment.
    """

  @abc.abstractmethod
  def _observe(self) -> types.NestedArray:
    """Returns an observation."""
