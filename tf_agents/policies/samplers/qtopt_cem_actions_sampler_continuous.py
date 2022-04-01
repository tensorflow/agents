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
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


@gin.configurable
class GaussianActionsSampler(qtopt_cem_actions_sampler.ActionsSampler):
  """Continuous Gaussian actions sampler.

  Supports nested action_spec with 1d continuous actions.

  Given a batch of distribution params(including mean and var), sample and
  return [B, N, A] actions, where 'B' means batch_size, 'N' means num_samples,
  'A' means action_size.
  """

  def __init__(self,
               action_spec,
               sample_clippers=None,
               sample_rejecters=None,
               max_rejection_iterations=10,
               support_integer=False):

    super(GaussianActionsSampler, self).__init__(action_spec, sample_clippers)

    self._sample_rejecters = sample_rejecters
    self._max_rejection_iterations = tf.constant(max_rejection_iterations)

    self._support_integer = support_integer
    if not support_integer:
      for flat_action_spec in tf.nest.flatten(action_spec):
        if flat_action_spec.dtype.is_integer:
          raise ValueError('Only continuous action is supported by this '
                           'sampler. The action_spec: {} contains discrete '
                           'action'.format(action_spec))
        if flat_action_spec.shape.rank > 1:
          raise ValueError('Only 1d action is supported by this sampler. '
                           'The action_spec: {} contains action whose rank > 1.'
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
      var: A nested structure containing  [B, A] sized tensors where each row
        is the refitted var.
    """

    def get_mean(best_samples):
      mean, _ = tf.nn.moments(best_samples, axes=1)  # mu, var: [B, A]
      return mean

    def get_var(best_samples):
      _, var = tf.nn.moments(best_samples, axes=1)  # mu, var: [B, A]
      return var

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

    def sample_and_transpose(mean, var):
      dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
      sample = tf.transpose(dist.sample(num_samples), [1, 0, 2])
      return sample

    batch_size = tf.shape(tf.nest.flatten(mean)[0])[0]

    def sample_fn(mean_sample, var_sample, state_sample):
      # [B, N, A]
      samples_continuous = tf.nest.map_structure(sample_and_transpose,
                                                 mean_sample, var_sample)

      if self._sample_clippers:
        for sample_clipper in self._sample_clippers:
          samples_continuous = sample_clipper(samples_continuous, state_sample)

      samples_continuous = tf.nest.map_structure(
          common.clip_to_spec, samples_continuous, self._action_spec)
      return samples_continuous

    @tf.function
    def rejection_sampling(sample_rejector):
      valid_batch_samples = tf.nest.map_structure(
          lambda spec: tf.TensorArray(spec.dtype, size=batch_size),
          self._action_spec)

      for b_indx in tf.range(batch_size):
        k = tf.constant(0)
        # pylint: disable=cell-var-from-loop
        valid_samples = tf.nest.map_structure(
            lambda spec: tf.TensorArray(spec.dtype, size=num_samples),
            self._action_spec)

        count = tf.constant(0)
        while count < self._max_rejection_iterations:
          count += 1
          mean_sample = tf.nest.map_structure(
              lambda t: tf.expand_dims(tf.gather(t, b_indx), axis=0), mean)
          var_sample = tf.nest.map_structure(
              lambda t: tf.expand_dims(tf.gather(t, b_indx), axis=0), var)
          if state is not None:
            state_sample = tf.nest.map_structure(
                lambda t: tf.expand_dims(tf.gather(t, b_indx), axis=0), state)
          else:
            state_sample = None

          samples = sample_fn(mean_sample, var_sample, state_sample)  # n, a

          mask = sample_rejector(samples, state_sample)

          mask = mask[0, ...]
          mask_index = tf.where(mask)[:, 0]

          num_mask = tf.shape(mask_index)[0]
          if num_mask == 0:
            continue

          good_samples = tf.nest.map_structure(
              lambda t: tf.gather(t, mask_index, axis=1)[0, ...], samples)

          for sample_idx in range(num_mask):
            if k >= num_samples:
              break
            valid_samples = tf.nest.map_structure(
                lambda gs, vs: vs.write(k, gs[sample_idx:sample_idx+1, ...]),
                good_samples, valid_samples)
            k += 1

        if k < num_samples:
          zero_samples = tensor_spec.zero_spec_nest(
              self._action_spec, outer_dims=(num_samples-k,))
          for sample_idx in range(num_samples-k):
            valid_samples = tf.nest.map_structure(
                lambda gs, vs: vs.write(k, gs[sample_idx:sample_idx+1, ...]),
                zero_samples, valid_samples)

        valid_samples = tf.nest.map_structure(lambda vs: vs.concat(),
                                              valid_samples)

        valid_batch_samples = tf.nest.map_structure(
            lambda vbs, vs: vbs.write(b_indx, vs), valid_batch_samples,
            valid_samples)

      samples_continuous = tf.nest.map_structure(
          lambda a: a.stack(), valid_batch_samples)
      return samples_continuous

    if self._sample_rejecters:
      samples_continuous = rejection_sampling(self._sample_rejecters)
      def set_b_n_shape(t):
        t.set_shape(tf.TensorShape([None, num_samples] + t.shape[2:].dims))

      tf.nest.map_structure(set_b_n_shape, samples_continuous)
    else:
      samples_continuous = sample_fn(mean, var, state)

    if self._support_integer:
      samples_continuous = tf.nest.map_structure(
          lambda t, s: tf.cast(t, s.dtype), samples_continuous,
          self._action_spec)
    return samples_continuous
