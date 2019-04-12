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
from __future__ import print_function

import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import trajectory


def make_timestep_mask(batched_next_time_step):
  """Create a mask for final incomplete episodes and episode transitions.

  Args:
    batched_next_time_step: Next timestep, doubly-batched
      [batch_dim, time_dim, ...].

  Returns:
    A mask, type tf.float32, that is 0.0 for all between-episode timesteps
      (batched_next_time_step is FIRST), or where the episode is
      not compelete, so the return computation would not be correct.
  """
  # 1.0 for timesteps of all complete episodes. 0.0 for incomplete episode at
  #   the end of the sequence.
  episode_is_complete = tf.cumsum(
      tf.cast(batched_next_time_step.is_last(), tf.float32),
      axis=1,
      reverse=True) > 0

  # 1.0 for all valid timesteps. 0.0 where between episodes.
  not_between_episodes = ~batched_next_time_step.is_first()

  return tf.cast(episode_is_complete & not_between_episodes, tf.float32)


def get_distribution_params(nested_distribution):
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


def nested_kl_divergence(nested_from_distribution, nested_to_distribution,
                         outer_dims=()):
  """Given two nested distributions, sum the KL divergences of the leaves."""
  tf.nest.assert_same_structure(nested_from_distribution,
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
