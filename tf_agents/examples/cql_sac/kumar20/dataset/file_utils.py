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

"""Utilities to generate and save transitions to TFRecord files."""
from typing import Dict
from absl import logging

import d4rl  # pylint: disable=unused-import
import numpy as np

from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import example_encoding_dataset


def create_trajectory(state: types.Array, action: types.Array,
                      discount: types.Array, reward: types.Array,
                      step_type: types.Array,
                      next_step_type: types.Array) -> trajectory.Trajectory:
  """Creates a Trajectory from current and next state information."""
  return trajectory.Trajectory(
      step_type=step_type,
      observation=state,
      action=action,
      policy_info=(),
      next_step_type=next_step_type,
      reward=reward,
      discount=discount)


def create_transition(
    state: types.Array,
    action: types.Array,
    next_state: types.Array,
    discount: types.Array,
    reward: types.Array,
    step_type: types.Array,
    next_step_type: types.Array
) -> trajectory.Transition:
  """Creates a Transition from current and next state information."""
  tfagents_time_step = ts.TimeStep(
      step_type=step_type,
      reward=np.zeros_like(reward),  # unknown
      discount=np.zeros_like(discount),  # unknown
      observation=state)
  action_step = policy_step.PolicyStep(action=action, state=(), info=())
  tfagents_next_time_step = ts.TimeStep(
      step_type=next_step_type,
      reward=reward,
      discount=discount,
      observation=next_state)
  return trajectory.Transition(
      time_step=tfagents_time_step,
      action_step=action_step,
      next_time_step=tfagents_next_time_step)


def write_samples_to_tfrecord(dataset_dict: Dict[str, types.Array],
                              collect_data_spec: trajectory.Transition,
                              dataset_path: str,
                              start_episode: int,
                              end_episode: int,
                              use_trajectories: bool = True) -> None:
  """Creates and writes samples to a TFRecord file."""
  tfrecord_observer = example_encoding_dataset.TFRecordObserver(
      dataset_path, collect_data_spec, py_mode=True)
  states = dataset_dict['states']
  actions = dataset_dict['actions']
  discounts = dataset_dict['discounts']
  rewards = dataset_dict['rewards']
  num_episodes = len(dataset_dict['episode_start_index'])

  for episode_i in range(start_episode, end_episode):
    episode_start_index = dataset_dict['episode_start_index'][episode_i]
    # If this is the last episode, end at the final step.
    if episode_i == (num_episodes - 1):
      episode_end_index = len(states)
    else:
      # Otherwise, end before the next episode.
      episode_end_index = dataset_dict['episode_start_index'][episode_i + 1]

    for step_i in range(int(episode_start_index), int(episode_end_index)):
      # Set step type.
      if step_i == episode_end_index - 1:
        step_type = ts.StepType.LAST
      elif step_i == episode_start_index:
        step_type = ts.StepType.FIRST
      else:
        step_type = ts.StepType.MID

      # Set next state.
      # If at the last step in the episode, create a dummy next step.
      if step_type == ts.StepType.LAST:
        next_state = np.zeros_like(states[step_i])
        next_step_type = ts.StepType.FIRST
      else:
        next_state = states[step_i + 1]
        next_step_type = (
            ts.StepType.LAST if step_i == episode_end_index -
            2 else ts.StepType.MID)

      if use_trajectories:
        sample = create_trajectory(
            state=states[step_i],
            action=actions[step_i],
            discount=discounts[step_i],
            reward=rewards[step_i],
            step_type=step_type,
            next_step_type=next_step_type)
      else:
        sample = create_transition(
            state=states[step_i],
            action=actions[step_i],
            next_state=next_state,
            discount=discounts[step_i],
            reward=rewards[step_i],
            step_type=step_type,
            next_step_type=next_step_type)
      tfrecord_observer(sample)

  tfrecord_observer.close()
  logging.info('Wrote episodes [%d-%d] to %s', start_episode, end_episode,
               dataset_path)
