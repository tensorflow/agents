# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example registering of a new Gym environment.

See agents/dqn/examples/train_eval_gym_rnn.py for usage.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.envs.classic_control import cartpole
from gym.envs.registration import register
import numpy as np


class MaskedCartPoleEnv(cartpole.CartPoleEnv):
  """Cartpole environment with masked velocity components.

  This environment is useful as a unit tests for agents that utilize recurrent
  networks.
  """

  def __init__(self):
    super(MaskedCartPoleEnv, self).__init__()
    high = np.array([
        self.x_threshold * 2,
        self.theta_threshold_radians * 2,
    ])

    self.observation_space = gym.spaces.Box(-high, high)

  def _mask_observation(self, observation):
    return observation[[0, 2]]

  def reset(self):
    observation = super(MaskedCartPoleEnv, self).reset()
    # Get rid of velocity components at index 1, and 3.
    return self._mask_observation(observation)

  def step(self, action):
    observation, reward, done, info = super(MaskedCartPoleEnv,
                                            self).step(action)
    # Get rid of velocity components at index 1, and 3.
    return self._mask_observation(observation), reward, done, info


register(
    id='MaskedCartPole-v0',
    entry_point=MaskedCartPoleEnv,
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='MaskedCartPole-v1',
    entry_point=MaskedCartPoleEnv,
    max_episode_steps=500,
    reward_threshold=475.0,
)
