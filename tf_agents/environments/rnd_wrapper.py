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

"""Wrapper implementing Random Network Distillation (RND)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import gym
import gym.spaces
import numpy as np
import tensorflow as tf

from tf_agents import specs
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


@gin.configurable
class RNDWrapper(wrappers.PyEnvironmentBaseWrapper):
  def __init__(self, env):
    super(RNDWrapper, self).__init__(env)

    # TODO Initialize RND target network and RND predictor network

  def _reset(self):
    time_step = self._env.reset()
    # TODO Compute intrinsic reward
    intrinsic_reward = self._get_intrinsic_reward(time_step.observation)
    time_step = time_step._replace(reward=time_step.reward + intrinsic_reward)

    return time_step

  def _step(self, action):
    time_step = self._env.step(action)
    # TODO Compute intrinsic reward
    intrinsic_reward = self._get_intrinsic_reward(time_step.observation)
    time_step = time_step._replace(reward=time_step.reward + intrinsic_reward)

    return time_step
  
  def _get_intrinsic_reward(self, observation):
    # TODO Implement
    # TODO Check type / dtype / shape
    return 0
