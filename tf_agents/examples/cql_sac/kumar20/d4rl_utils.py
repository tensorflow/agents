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

"""Utils for using D4RL in TF-Agents."""
import d4rl  # pylint: disable=unused-import
import gin
import gym

from gym.wrappers.time_limit import TimeLimit
from tf_agents.environments import gym_wrapper


@gin.configurable
def load_d4rl(env_name, default_time_limit=1000):
  """Loads the python environment from D4RL."""
  gym_env = gym.make(env_name)
  gym_spec = gym.spec(env_name)

  # Default to env time limit unless it is not specified.
  if gym_spec.max_episode_steps in [0, None]:
    gym_env = TimeLimit(gym_env, max_episode_steps=default_time_limit)

  # Wrap TF-Agents environment.
  env = gym_wrapper.GymWrapper(gym_env)
  return env
