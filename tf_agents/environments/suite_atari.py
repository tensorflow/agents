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

"""Suite for loading Atari Gym environments."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Dict, Optional, Sequence, Text

import atari_py  # pylint: disable=unused-import
import gin
import gym
import numpy as np

from tf_agents.environments import atari_preprocessing
from tf_agents.environments import atari_wrappers
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym

from tf_agents.typing import types


# Typical Atari 2600 Gym environment with some basic preprocessing.
DEFAULT_ATARI_GYM_WRAPPERS = (atari_preprocessing.AtariPreprocessing,)
# The following is just AtariPreprocessing with frame stacking. Performance wise
# it's much better to have stacking implemented as part of replay-buffer/agent.
# As soon as this functionality in TF-Agents is ready and verified, this set of
# wrappers will be removed.
DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING = DEFAULT_ATARI_GYM_WRAPPERS + (
    atari_wrappers.FrameStack4,)
gin.constant('DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING',
             DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)


@gin.configurable
def game(name: Text = 'Pong',
         obs_type: Text = 'image',
         mode: Text = 'NoFrameskip',
         version: Text = 'v0') -> Text:
  """Generates the full name for the game.

  Args:
    name: String. Ex. Pong, SpaceInvaders, ...
    obs_type: String, type of observation. Ex. 'image' or 'ram'.
    mode: String. Ex. '', 'NoFrameskip' or 'Deterministic'.
    version: String. Ex. 'v0' or 'v4'.

  Returns:
    The full name for the game.
  """
  assert obs_type in ['image', 'ram']
  assert mode in ['', 'NoFrameskip', 'Deterministic']
  assert version in ['v0', 'v4']
  if obs_type == 'ram':
    name = '{}-ram'.format(name)
  return '{}{}-{}'.format(name, mode, version)


@gin.configurable
def load(
    environment_name: Text,
    discount: types.Int = 1.0,
    max_episode_steps: Optional[types.Int] = None,
    gym_env_wrappers: Sequence[
        types.GymEnvWrapper] = DEFAULT_ATARI_GYM_WRAPPERS,
    env_wrappers: Sequence[types.PyEnvWrapper] = (),
    spec_dtype_map: Optional[Dict[gym.Space, np.dtype]] = None
) -> py_environment.PyEnvironment:
  """Loads the selected environment and wraps it with the specified wrappers."""
  if spec_dtype_map is None:
    spec_dtype_map = {gym.spaces.Box: np.uint8}

  gym_spec = gym.spec(environment_name)
  gym_env = gym_spec.make()

  if max_episode_steps is None and gym_spec.max_episode_steps is not None:
    max_episode_steps = gym_spec.max_episode_steps

  return suite_gym.wrap_env(
      gym_env,
      discount=discount,
      max_episode_steps=max_episode_steps,
      gym_env_wrappers=gym_env_wrappers,
      time_limit_wrapper=atari_wrappers.AtariTimeLimit,
      env_wrappers=env_wrappers,
      spec_dtype_map=spec_dtype_map,
      auto_reset=False)
