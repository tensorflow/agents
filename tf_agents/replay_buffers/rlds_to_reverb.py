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

"""A util to convert RLDS dataset to TF Agents trajectories and send it to Reverb."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List, Tuple, Union

from absl import logging

import numpy as np
import rlds
import tensorflow as tf

from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


def get_rlds_step_features() -> List[str]:
  """Returns a list representing features in an RLDS step."""
  return [
      rlds.OBSERVATION, rlds.ACTION, rlds.REWARD, rlds.DISCOUNT, rlds.IS_FIRST,
      rlds.IS_LAST, rlds.IS_TERMINAL
  ]


def _validate_rlds_episode_spec(rlds_data: tf.data.Dataset) -> None:
  """Validates the dataset of RLDS data.

  Validates that rlds_data has RLDS steps.

  Args:
    rlds_data: An RLDS dataset is tf.data.Dataset of RLDS episodes, where each
      episode contains a tf.data.Dataset of RLDS steps. An RLDS step is a
      dictionary of tensors containing is_first, is_last, observation, action,
      reward, is_terminal, and discount.

  Raises:
    ValueError: If no RLDS steps exist in rlds_data.
  """
  episode_spec = rlds_data.element_spec
  if rlds.STEPS not in episode_spec:
    raise ValueError(
        f'No dataset representing RLDS {rlds.STEPS} exist in the data.')


def _validate_rlds_step_spec(
    rlds_step_spec: tf.data.Dataset.element_spec) -> None:
  """Validates the RLDS step spec.

  Validates that rlds_step_spec is correct spec for RLDS steps.

  Args:
    rlds_step_spec: An element spec to be validated as correct RLDS steps spec.

  Raises:
    ValueError: If RLDS step spec is not valid.
  """
  rlds_step_features = get_rlds_step_features()
  if not all(item in rlds_step_spec for item in rlds_step_features):
    raise ValueError(
        'Invalid RLDS step spec. Features expected '
        f'are {rlds_step_features}, but found {list(rlds_step_spec.keys())}')


def create_trajectory_data_spec(
    rlds_data: tf.data.Dataset) -> trajectory.Trajectory:
  """Creates data spec for initializing Reverb server and Reverb Replay Buffer.

  Creates a data spec for the corresponding trajectory dataset that can be
  created using the rlds_data provided as input. This data spec is necessary for
  initializing Reverb server and Reverb Replay Buffer.

  Args:
    rlds_data: An RLDS dataset is tf.data.Dataset of RLDS episodes, where each
      episode contains a tf.data.Dataset of RLDS steps. An RLDS step is a
      dictionary of tensors containing is_first, is_last, observation, action,
      reward, is_terminal, and discount.

  Returns:
    A trajectory representing tensor specs for trajectory dataset that can be
    created using the rlds_data provided as input.

  Raises:
    ValueError: If no RLDS steps exist in rlds_data.
    ValueError: If step spec for rlds_data is not valid.
  """
  _validate_rlds_episode_spec(rlds_data)
  rlds_data = rlds_data.flat_map(lambda episode: episode[rlds.STEPS])
  element_spec = rlds_data.element_spec
  _validate_rlds_step_spec(element_spec)
  time_step_spec = ts.time_step_spec(element_spec[rlds.OBSERVATION],
                                     element_spec[rlds.REWARD])
  action_spec = policy_step.PolicyStep(element_spec[rlds.ACTION])
  return trajectory.from_transition(time_step_spec, action_spec, time_step_spec)


def convert_rlds_to_trajectories(rlds_data: tf.data.Dataset) -> tf.data.Dataset:
  """Converts the RLDS data to a dataset of trajectories.

  Converts the rlds_data provided to a dataset of TF Agents trajectories by
  flattening and converting it into batches and then tuples of overlapping pairs
  of adjacent RLDS steps.

  An end step of first step type is padded to the RLDS data to ensure that the
  trajectory created using the last step of the last episode has a valid next
  step type.

    Input    ------------------------------------------->       Output

   RLDS Data --> Flatten -->   Overlapping Pairs     -->  TF Agents trajectories
  -----------    --------   ------------------------      ---------------------
   episode 1     step 1      (step 1, step 2)                Trajectory 1
    --------     --------   ------------------------      ---------------------
     step 1      step 2      (step 2, step 3)                Trajectory 2
     step 2      --------   ------------------------      ---------------------
     step 3      step 3      (step 3, step 4)                Trajectory 3
    --------     --------   ------------------------      ---------------------
  -----------    step 4      (step 4, step 5)                Trajectory 4
   episode 2     --------   ------------------------      ---------------------
    --------     step 5      (step 5, step 6)                Trajectory 5
     step 4      --------   ------------------------      ---------------------
     step 5      step 6      (step 6, padded end step)       Trajectory 6
     step 6      --------   ------------------------      ---------------------
    --------

  Args:
    rlds_data: An RLDS dataset is tf.data.Dataset of RLDS episodes, where each
      episode contains a tf.data.Dataset of RLDS steps. An RLDS step is a
      dictionary of tensors containing is_first, is_last, observation, action,
      reward, is_terminal, and discount.

  Returns:
    A dataset of type tf.data.Dataset, elements of which are TF Agents
    trajectories corresponding to the RLDS steps provided in rlds_data.

  Raises:
    ValueError: If no RLDS steps exist in rlds_data.
    ValueError: If step spec for rlds_data is not valid.
    InvalidArgumentError: If the RLDS dataset provided has episodes that:
      - Incorrectly end i.e. does not end in last step.
      - Incorrectly terminate i.e. a terminal step is not the last step.
      - Incorrectly begin i.e.  a last step in not followed by the first step.
        Please note that the last step of the last episode is taken care of
        in the function and user does not need to make sure that the last step
        of the last episode is followed by a first step.
  """

  def _pair_to_tuple(
      pair: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Converts a batch of two adjacent RLDS steps to a Tuple of RLDS steps."""
    return ({
        rlds.IS_FIRST: pair[rlds.IS_FIRST][0],
        rlds.IS_LAST: pair[rlds.IS_LAST][0],
        rlds.IS_TERMINAL: pair[rlds.IS_TERMINAL][0],
        rlds.REWARD: pair[rlds.REWARD][0],
        rlds.DISCOUNT: pair[rlds.DISCOUNT][0],
        rlds.OBSERVATION: pair[rlds.OBSERVATION][0],
        rlds.ACTION: pair[rlds.ACTION][0]
    }, {
        rlds.IS_FIRST: pair[rlds.IS_FIRST][1],
        rlds.IS_LAST: pair[rlds.IS_LAST][1],
        rlds.IS_TERMINAL: pair[rlds.IS_TERMINAL][1],
        rlds.REWARD: pair[rlds.REWARD][1],
        rlds.DISCOUNT: pair[rlds.DISCOUNT][1],
        rlds.OBSERVATION: pair[rlds.OBSERVATION][1],
        rlds.ACTION: pair[rlds.ACTION][1]
    })

  def _get_step_type(step) -> np.ndarray:
    if step[rlds.IS_FIRST]:
      return ts.StepType.FIRST
    elif step[rlds.IS_LAST]:
      return ts.StepType.LAST
    else:
      return ts.StepType.MID

  def _is_complete(step: Dict[str, tf.Tensor]) -> tf.Tensor:
    return step[rlds.IS_LAST] and step[rlds.IS_TERMINAL]

  def _get_discount(step: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Returns 0 for complete episode, else returns the current discount."""
    return tf.constant(
        0.0, dtype=tf.float32) if _is_complete(step) else step[rlds.DISCOUNT]

  def _validate_step(current_step: Dict[str, tf.Tensor],
                     next_step: Dict[str, tf.Tensor]) -> None:
    """Validates a tuple of adjacent RLDS steps."""

    # This check validates any incorrectly terminated episodes in RLDS data.
    if current_step[rlds.IS_TERMINAL]:
      tf.Assert(current_step[rlds.IS_LAST],
                ['Terminal step must be the last step of an episode.'])

    # This check validates any incorrectly ending episodes in the RLDS dataset.
    #
    # Please note that the last step of last episode will also be validated by
    # this check since we pad the RLDS data with a step of first step type.
    if not current_step[rlds.IS_LAST] and not current_step[rlds.IS_FIRST]:
      tf.Assert(not next_step[rlds.IS_FIRST],
                ['Mid step should not be followed by a first step.'])

    # This check validates any episodes that begin incorrectly in the RLDS data.
    if current_step[rlds.IS_LAST]:
      tf.Assert(next_step[rlds.IS_FIRST],
                ['Last step of an episode must be followed by a first step.'])

  def _to_trajectory(current_step: Dict[str, tf.Tensor],
                     next_step: Dict[str, tf.Tensor]) -> trajectory.Trajectory:
    """Converts a tuple of adjacent RLDS steps to a trajectory."""
    _validate_step(current_step, next_step)
    step_type = _get_step_type(current_step)
    next_step_type = _get_step_type(next_step)

    # Represent single step episode with step_type=LAST and next_step_type=LAST.
    if step_type == ts.StepType.FIRST and next_step_type == ts.StepType.FIRST:
      step_type = ts.StepType.LAST
      next_step_type = ts.StepType.LAST

    return trajectory.Trajectory(
        step_type=step_type,
        observation=current_step[rlds.OBSERVATION],
        action=current_step[rlds.ACTION],
        policy_info=(),
        next_step_type=next_step_type,
        reward=current_step[rlds.REWARD],
        discount=_get_discount(current_step))

  _validate_rlds_episode_spec(rlds_data)
  rlds_data = rlds_data.flat_map(lambda episode: episode[rlds.STEPS])
  _validate_rlds_step_spec(rlds_data.element_spec)

  # Pad a step of first step type at the end of the RLDS data to ensure that the
  # trajectory created using the last step of the last episode has a valid next
  # step type.
  rlds_data = rlds.transformations.concatenate(
      rlds_data, tf.data.Dataset.from_tensors({rlds.IS_FIRST: True}))

  # Batch the RLDS data as pair of adjacent steps with an overlap of one step
  # and then create tuples of those batches to finally to create TF Agents
  # trajectories with correct next step type.
  return rlds.transformations.batch(
      dataset=rlds_data, size=2, shift=1,
      drop_remainder=True).map(_pair_to_tuple).map(_to_trajectory)


def push_rlds_to_reverb(
    rlds_data: tf.data.Dataset,
    reverb_observer: Union[reverb_utils.ReverbAddEpisodeObserver,
                           reverb_utils.ReverbAddTrajectoryObserver]
) -> int:
  """Pushes the RLDS data to Reverb server as TF Agents trajectories.

  Pushes the rlds_data provided to Reverb server using reverb_observer after
  converting it to TF Agents trajectories.

  Please note that the data spec used to initialize replay buffer and reverb
  server  for creating the reverb_observer must match the data
  spec for rlds_data.

  Args:
    rlds_data: An RLDS dataset is tf.data.Dataset of RLDS episodes, where each
      episode contains a tf.data.Dataset of RLDS steps. An RLDS step is a
      dictionary of tensors containing is_first, is_last, observation, action,
      reward, is_terminal, and discount.
    reverb_observer: A Reverb observer for writing trajectories data to Reverb.

  Returns:
    An int representing the number of trajectories successfully pushed to RLDS.

  Raises:
    ValueError: If no RLDS steps exist in rlds_data.
    ValueError: If step spec for rlds_data is not valid.
    ValueError: If data spec used to initialize replay buffer and reverb server
    for creating the reverb_observer does not match the data spec for trajectory
    dataset that can be created using rlds_data.
    InvalidArgumentError: If the RLDS dataset provided has episodes that:
      - Incorrectly end i.e. does not end in last step.
      - Incorrectly terminate i.e. a terminal step is not the last step.
      - Incorrectly begin i.e. a last step is not followed by the first step.
        Please note that the last step of the last episode is taken care of
        in the function and the user does not need to make sure that the last
        step of the last episode is followed by a first step.
  """

  table_signature_spec = reverb_observer.get_table_signature()
  if table_signature_spec != tensor_spec.add_outer_dim(
      create_trajectory_data_spec(rlds_data)):
    raise ValueError(
        'Replay buffer table signature spec should match RLDS data spec.')

  steps = 0
  for entry in convert_rlds_to_trajectories(rlds_data):
    reverb_observer(entry)
    steps += 1
  logging.info('Successfully wrote %d steps to Reverb.', steps)
  return steps
