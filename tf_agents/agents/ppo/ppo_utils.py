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

"""Utils functions for ppo_agent.py."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Sequence

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import nest_utils


def make_trajectory_mask(batched_traj: trajectory.Trajectory) -> types.Tensor:
  """Mask boundary trajectories and those with invalid returns and advantages.

  Args:
    batched_traj: Trajectory, doubly-batched [batch_dim, time_dim,...]. It must
      be preprocessed already.

  Returns:
    A mask, type tf.float32, that is 0.0 for all between-episode Trajectory
      (batched_traj.step_type is LAST) and 0.0 if the return value is
      unavailable.
  """
  # 1.0 for all valid trajectories. 0.0 where between episodes.
  not_between_episodes = ~batched_traj.is_boundary()

  # 1.0 for trajectories with valid return values. 0.0 where return and
  # advantage are both 0. This happens to the last item when the experience gets
  # preprocessed, as insufficient information was available for calculating
  # advantages.
  valid_return_value = ~(
      tf.equal(batched_traj.policy_info['return'], 0)
      & tf.equal(batched_traj.policy_info['normalized_advantage'], 0))

  return tf.cast(not_between_episodes & valid_return_value, tf.float32)


def make_timestep_mask(batched_next_time_step: ts.TimeStep,
                       allow_partial_episodes: bool = False) -> types.Tensor:
  """Create a mask for transitions and optionally final incomplete episodes.

  Args:
    batched_next_time_step: Next timestep, doubly-batched [batch_dim, time_dim,
      ...].
    allow_partial_episodes: If true, then steps on incomplete episodes are
      allowed.

  Returns:
    A mask, type tf.float32, that is 0.0 for all between-episode timesteps
      (batched_next_time_step is FIRST). If allow_partial_episodes is set to
      False, the mask has 0.0 for incomplete episode at the end of the sequence.
  """
  if allow_partial_episodes:
    episode_is_complete = None
  else:
    # 1.0 for timesteps of all complete episodes. 0.0 for incomplete episode at
    #   the end of the sequence.
    episode_is_complete = tf.cumsum(
        tf.cast(batched_next_time_step.is_last(), tf.float32),
        axis=1,
        reverse=True) > 0

  # 1.0 for all valid timesteps. 0.0 where between episodes.
  not_between_episodes = ~batched_next_time_step.is_first()

  if allow_partial_episodes:
    return tf.cast(not_between_episodes, tf.float32)
  else:
    return tf.cast(episode_is_complete & not_between_episodes, tf.float32)


def get_distribution_params(
    nested_distribution: types.NestedDistribution) -> types.NestedTensor:
  """Get the params for an optionally nested action distribution.

  Only returns parameters that have tf.Tensor values.

  Args:
    nested_distribution: The nest of distributions whose parameter tensors to
      extract.
  Returns:
    A nest of distribution parameters. Each leaf is a dict corresponding to one
      distribution, with keys as parameter name and values as tensors containing
      parameter values.
  """
  def _tensor_parameters_only(params):
    return {k: params[k] for k in params if isinstance(params[k], tf.Tensor)}

  return tf.nest.map_structure(
      lambda single_dist: _tensor_parameters_only(single_dist.parameters),
      nested_distribution)


def nested_kl_divergence(nested_from_distribution: types.NestedDistribution,
                         nested_to_distribution: types.NestedDistribution,
                         outer_dims: Sequence[int] = ()) -> types.Tensor:
  """Given two nested distributions, sum the KL divergences of the leaves."""
  nest_utils.assert_same_structure(nested_from_distribution,
                                   nested_to_distribution)

  # Make list pairs of leaf distributions.
  flat_from_distribution = tf.nest.flatten(nested_from_distribution)
  flat_to_distribution = tf.nest.flatten(nested_to_distribution)
  all_kl_divergences = [from_dist.kl_divergence(to_dist)
                        for from_dist, to_dist
                        in zip(flat_from_distribution, flat_to_distribution)]

  # Sum the kl of the leaves.
  summed_kl_divergences = tf.add_n(all_kl_divergences)

  # Reduce_sum over non-batch dimensions.
  reduce_dims = list(range(len(summed_kl_divergences.shape)))
  for dim in outer_dims:
    reduce_dims.remove(dim)
  total_kl = tf.reduce_sum(input_tensor=summed_kl_divergences, axis=reduce_dims)

  return total_kl


def get_metric_observers(metrics):
  """Returns a list of observers, one for each metric."""
  def get_metric_observer(metric):

    def metric_observer(time_step, action, next_time_step,
                        policy_state):
      action_step = policy_step.PolicyStep(action, policy_state, ())
      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      return metric(traj)
    return metric_observer
  return [get_metric_observer(m) for m in metrics]
