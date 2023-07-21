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

"""Actions sampler that supports sampling only continuous actions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.policies.samplers import qtopt_cem_actions_sampler
from tf_agents.utils import common


@gin.configurable
class GaussianActionsSampler(qtopt_cem_actions_sampler.ActionsSampler):
  """Continuous Gaussian actions sampler.

  Supports in nested action_spec with 1d continuous actions or 1d
  discrete actions.

  Given a batch of distribution params(including mean and var), sample and
  return [B, N, A] actions, where 'B' means batch_size, 'N' means num_samples,
  'A' means action_size.
  """

  def __init__(self, action_spec, sample_clippers=None):

    super(GaussianActionsSampler, self).__init__(action_spec, sample_clippers)

    for flat_action_spec in tf.nest.flatten(action_spec):
      if flat_action_spec.shape.rank > 1:
        raise ValueError('Only 1d action is supported by this sampler. '
                         'The action_spec: {} contains action whose rank > 1. '
                         'Consider coverting it into multiple 1d '
                         'actions.'.format(action_spec))

  def refit_distribution_to(self, target_sample_indices, samples):
    """Refits distribution according to actions with index of ind.

    Args:
      target_sample_indices: A [B, M] sized tensor indicating the index
      samples: A nested structure corresponding to action_spec. Each action is
        a [B, N, A] sized tensor.

    Returns:
      mean: A nested structure containing [B, A] sized tensors where each row
        is the refitted mean.
      var: A nested structure containing [B, A] sized tensors where each row
        is the refitted var.
    """

    def get_mean(best_samples):
      mean, _ = tf.nn.moments(best_samples, axes=1)  # mu, var: [B, A]
      return tf.cast(mean, tf.float32)

    def get_var(best_samples):
      _, var = tf.nn.moments(best_samples, axes=1)  # mu, var: [B, A]
      return tf.cast(var, tf.float32)

    best_samples = tf.nest.map_structure(
        lambda s: tf.gather(s, target_sample_indices, batch_dims=1), samples)
    mean = tf.nest.map_structure(get_mean, best_samples)
    var = tf.nest.map_structure(get_var, best_samples)

    return mean, var

  def sample_batch_and_clip(self, num_samples, mean, var, state=None):
    """Samples and clips a batch of actions [B, N, A] with mean and var.

    Args:
      num_samples: Number of actions to sample each round.
      mean: A nested structure containing [B, A] shaped tensor representing the
        mean of the actions to be sampled.
      var: A nested structure containing [B, A] shaped tensor representing the
        variance of the actions to be sampled.
      state: Nested state tensor constructed according to oberservation_spec
        of the task.

    Returns:
      actions:  A nested structure containing tensor of sampled actions with
        shape [B, N, A]
    """

    def sample_and_transpose(mean, var, spec):
      dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
      sample = tf.transpose(dist.sample(num_samples), [1, 0, 2])
      return tf.cast(sample, spec.dtype)

    # [B, N, A]
    samples = tf.nest.map_structure(
        sample_and_transpose, mean, var, self._action_spec)

    actions = tf.nest.map_structure(
        common.clip_to_spec, samples, self._action_spec)
    if self._sample_clippers:
      for sample_clipper in self._sample_clippers:
        actions = sample_clipper(actions, state)

    return actions
