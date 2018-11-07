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

from tf_agents.environments import trajectory
from tf_agents.policies import policy_step

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

nest = tf.contrib.framework.nest


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
      tf.to_float(batched_next_time_step.is_last()), axis=1, reverse=True) > 0

  # 1.0 for all valid timesteps. 0.0 where between episodes.
  not_between_episodes = ~batched_next_time_step.is_first()

  return tf.to_float(episode_is_complete & not_between_episodes)


def get_distribution_class_spec(policy, time_step_spec):
  """Gets a nest of action distribution classes.

  Args:
    policy: Policy for constructing action distribution.
    time_step_spec: Spec for time_step for creating action distribution.
  Returns:
    The nest of distribution class references.
  """
  sample_distribution_step = policy.distribution(
      tensor_spec.sample_spec_nest(time_step_spec, outer_dims=[1]),
      policy_state=policy.get_initial_state(1))
  sample_distribution = sample_distribution_step.action
  return nest.map_structure(lambda dist: dist.__class__, sample_distribution)


# TODO(eholly): After moving to third_party, move these spec functions into
#   actor_policy.py.
def get_distribution_params_spec(policy, time_step_spec):
  """Gets a possibly nested spec describing distribution params tensors.

  Args:
    policy: Policy for constructing action distribution.
    time_step_spec: Spec for time_step for creating action distribution.
  Returns:
    The nest of distribution parameter specs.
  """
  distribution_step = policy._distribution(  # pylint: disable=protected-access
      tensor_spec.sample_spec_nest(time_step_spec, outer_dims=[1]),
      policy_state=policy.get_initial_state(1))
  distribution = distribution_step.action
  distribution_parameters = get_distribution_params(distribution)
  unbatched_distribution_parameters = nest.map_structure_up_to(
      distribution,  # Use leaves of distribution nest as map leaves.
      nest_utils.unbatch_nested_tensors,
      distribution_parameters)

  distribution_params_spec = nest.map_structure(
      tensor_spec.TensorSpec.from_tensor, unbatched_distribution_parameters)
  if isinstance(time_step_spec.step_type, array_spec.ArraySpec):
    distribution_params_spec = nest.map_structure(
        tensor_spec.to_array_spec, distribution_params_spec)
  return distribution_params_spec


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
  return nest.map_structure(
      lambda single_dist: _tensor_parameters_only(single_dist.parameters),
      nested_distribution)


def get_distribution_from_params_and_classes(distribution_params,
                                             distribution_class_spec):
  """Reconstruct distribution from nests of distribution params and classes.

  Args:
    distribution_params: A nest of dictionaries, which each contain named args
      for the corresponding tf distribution constructor.
    distribution_class_spec: A nest of tf.distribution classes to use as
      constructors.
  Returns:
    nested_distributions: A nest of distributions with same structure as
      distribution_class_spec.
  """
  return nest.map_structure_up_to(
      distribution_class_spec,  # Use leaves of dist_class_spec as map leaves.
      lambda single_dist_params, dist_class: dist_class(**single_dist_params),
      distribution_params, distribution_class_spec)


def nested_kl_divergence(nested_from_distribution, nested_to_distribution,
                         outer_dims=()):
  """Given two nested distributions, sum the KL divergences of the leaves."""
  nest.assert_same_structure(nested_from_distribution, nested_to_distribution)

  # Make list pairs of leaf distributions.
  flat_from_distribution = nest.flatten(nested_from_distribution)
  flat_to_distribution = nest.flatten(nested_to_distribution)
  all_kl_divergences = [from_dist.kl_divergence(to_dist)
                        for from_dist, to_dist
                        in zip(flat_from_distribution, flat_to_distribution)]

  # Sum the kl of the leaves.
  summed_kl_divergences = tf.add_n(all_kl_divergences)

  # Reduce_sum over non-batch dimensions.
  reduce_dims = list(range(len(summed_kl_divergences.shape)))
  for dim in outer_dims:
    reduce_dims.remove(dim)
  total_kl = tf.reduce_sum(summed_kl_divergences, axis=reduce_dims)

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
