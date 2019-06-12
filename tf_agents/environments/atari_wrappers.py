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

"""Wrappers for Atari Environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import numpy as np
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts


class FrameStack4(gym.Wrapper):
  """Stack previous four frames (must be applied to Gym env, not our envs)."""

  STACK_SIZE = 4

  def __init__(self, env):
    super(FrameStack4, self).__init__(env)
    self._env = env
    self._frames = collections.deque(maxlen=FrameStack4.STACK_SIZE)
    space = self._env.observation_space
    shape = space.shape[0:2] + (FrameStack4.STACK_SIZE,)
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=shape, dtype=np.uint8)

  def __getattr__(self, name):
    """Forward all other calls to the base environment."""
    return getattr(self._env, name)

  def _generate_observation(self):
    return np.concatenate(self._frames, axis=2)

  def _reset(self):
    observation = self._env.reset()
    for _ in range(FrameStack4.STACK_SIZE):
      self._frames.append(observation)
    return self._generate_observation()

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)
    self._frames.append(observation)
    return self._generate_observation(), reward, done, info


# TODO(sfishman): Add tests for this wrapper.
class AtariTimeLimit(wrappers.PyEnvironmentBaseWrapper):
  """End episodes after specified number of steps and reset after game_over.

  This differs from the default TimeLimit wrapper in that it looks at the
  game_over property before resetting. We need this to properly handle life
  loss terminations -- the default TimeLimit wrapper would .reset() the
  environment and the step count after such a termination, but we want the
  environment to keep going.
  """

  def __init__(self, env, duration):
    super(AtariTimeLimit, self).__init__(env)
    self._duration = duration
    self._num_steps = 0

  def _reset(self):
    self._num_steps = 0
    return self._env.reset()

  def _step(self, action):
    if self.game_over:
      return self.reset()

    time_step = self._env.step(action)

    self._num_steps += 1
    if self._num_steps >= self._duration:
      time_step = time_step._replace(step_type=ts.StepType.LAST)

    return time_step

  @property
  def game_over(self):
    return self._num_steps >= self._duration or self.gym.game_over


class FireOnReset(gym.Wrapper):
  """Start every episode with action 1 (FIRE) + another action (2).

  In some environments (e.g., BeamRider, Breakout, Tennis) nothing
  happens until the player presses the FIRE button. This wrapper can
  be helpful in those environments, but it is not necessary.
  """

  def reset(self):
    observation = self.env.reset()
    # The following code is from https://github.com/openai/gym/...
    # ...blob/master/gym/wrappers/atari_preprocessing.py
    action_meanings = self.env.unwrapped.get_action_meanings()
    if action_meanings[1] == 'FIRE' and len(action_meanings) >= 3:
      self.env.step(1)
      observation, _, _, _ = self.env.step(2)
    return observation
