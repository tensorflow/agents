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
"""Collection of simple environments useful for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


class CountingEnv(py_environment.PyEnvironment):
  """Counds up in the observation as steps are taken."""

  def __init__(self, steps_per_episode=10):
    self._steps_per_episode = steps_per_episode

    self._episodes = 0
    self._current_step = np.array(0, dtype=np.int32)
    super(CountingEnv, self).__init__()

  def observation_spec(self):
    return specs.BoundedArraySpec((), dtype=np.int32)

  def action_spec(self):
    return specs.BoundedArraySpec((), dtype=np.int32, minimum=0, maximum=1)

  def _step(self, action):
    del action  # Unused.
    if self._current_time_step.is_last():
      return self._reset()
    self._current_step = np.array(1 + self._current_step, dtype=np.int32)
    if self._current_step < self._steps_per_episode:
      return ts.transition(self._get_observation(), 0)
    return ts.termination(self._get_observation(), 1)

  def _get_observation(self):
    if self._episodes:
      return np.array(10**self._episodes + self._current_step, dtype=np.int32)
    return self._current_step

  def _reset(self):
    if self._current_time_step and self._current_time_step.is_last():
      self._episodes += 1
    self._current_step = np.array(0, dtype=np.int32)
    return ts.restart(self._get_observation())
