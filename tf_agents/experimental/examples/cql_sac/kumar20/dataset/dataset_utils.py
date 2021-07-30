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

"""Utilities to gather transition information from a D4RL dataset."""
# Lint as: python3
from typing import Dict, List, Union

import d4rl  # pylint: disable=unused-import
import numpy as np

from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory

# D4RL datasets are a dictionary of {key: list}, where each item in the list
# is either a list of floats (e.g. observation) or a bool (e.g. is_terminal).
D4RLListType = List[Union[List[float], bool]]

# Dictionary of {key: np.array}.
EpisodeDictType = Dict[str, np.ndarray]


def create_collect_data_spec(
    dataset_dict: EpisodeDictType,
    use_trajectories: bool = True
) -> Union[trajectory.Transition, trajectory.Trajectory]:
  """Create a spec that describes the data collected by agent.collect_policy."""
  reward = dataset_dict['rewards'][0]
  discount = dataset_dict['discounts'][0]
  observation = dataset_dict['states'][0]
  action = dataset_dict['actions'][0]
  step_type = np.asarray(0, dtype=np.int32)

  if use_trajectories:
    return trajectory.Trajectory(
        step_type=ArraySpec(shape=step_type.shape, dtype=step_type.dtype),
        observation=ArraySpec(shape=observation.shape, dtype=observation.dtype),
        action=ArraySpec(shape=action.shape, dtype=action.dtype),
        policy_info=(),
        next_step_type=ArraySpec(shape=step_type.shape, dtype=step_type.dtype),
        reward=ArraySpec(shape=reward.shape, dtype=reward.dtype),
        discount=ArraySpec(shape=discount.shape, dtype=discount.dtype))
  else:
    time_step_spec = time_step.TimeStep(
        step_type=ArraySpec(shape=step_type.shape, dtype=step_type.dtype),
        reward=ArraySpec(shape=reward.shape, dtype=reward.dtype),
        discount=ArraySpec(shape=discount.shape, dtype=discount.dtype),
        observation=ArraySpec(shape=observation.shape, dtype=observation.dtype))
    action_spec = policy_step.PolicyStep(
        action=ArraySpec(shape=action.shape, dtype=action.dtype),
        state=(),
        info=())
    return trajectory.Transition(
        time_step=time_step_spec,
        action_step=action_spec,
        next_time_step=time_step_spec)


def create_episode_dataset(
    d4rl_dataset: Dict[str, D4RLListType],
    exclude_timeouts: bool,
    observation_dtype: np.dtype = np.float32) -> EpisodeDictType:
  """Create a dataset of episodes."""
  dataset = dict(
      states=[], actions=[], rewards=[], discounts=[], episode_start_index=[])

  new_episode = True
  for i in range(len(d4rl_dataset['observations'])):
    if new_episode:
      dataset['episode_start_index'].append(len(dataset['states']))
    dataset['states'].append(d4rl_dataset['observations'][i])
    dataset['actions'].append(d4rl_dataset['actions'][i])
    dataset['rewards'].append(d4rl_dataset['rewards'][i])
    dataset['discounts'].append(1.0 - int(d4rl_dataset['terminals'][i]))

    is_terminal = d4rl_dataset['terminals'][i]
    is_timeout = d4rl_dataset['timeouts'][i]
    # Remove the last step if this is a timeout. Assumes timeouts occur
    # after at least 1 step in an episode.
    if exclude_timeouts and is_timeout and not is_terminal:
      for key in ['states', 'actions', 'rewards', 'discounts']:
        dataset[key].pop()
    new_episode = is_terminal or is_timeout

  for key, value in dataset.items():
    dtype = observation_dtype if key == 'states' else np.float32
    dataset[key] = np.asarray(value, dtype)

  return dataset
