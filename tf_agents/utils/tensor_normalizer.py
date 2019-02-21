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

"""Tensor normalizer classses.

These encapsulate variables and function for tensor normalization.

Example usage:

observation = tf.placeholder(tf.float32, shape=[])
tensor_normalizer = StreamingTensorNormalizer(
    tensor_spec.TensorSpec([], tf.float32), scope='normalize_observation')
normalized_observation = tensor_normalizer.normalize(observation)
update_normalization = tensor_normalizer.update(observation)

with tf.Session() as sess:
  for o in observation_list:
    # Compute normalized observation given current observation vars.
    normalized_observation_ = sess.run(
        normalized_observation, feed_dict = {observation: o})

    # Update normalization params for next normalization op.
    sess.run(update_normalization, feed_dict = {observation: o})

    # Do something with normalized_observation_
    ...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tf_agents.utils import common

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


create_variable = common.create_variable


class TensorNormalizer(tf.Module):
  """Encapsulates tensor normalization and owns normalization variables."""

  def __init__(self,
               tensor_spec,
               scope='normalize_tensor'):
    super(TensorNormalizer, self).__init__(name=scope)
    self._scope = scope
    self._tensor_spec = tensor_spec
    self._create_variables()

  @property
  def nested(self):
    """True if tensor is nested, False otherwise."""
    return tf.nest.is_nested(self._tensor_spec)

  @abc.abstractmethod
  def copy(self, scope=None):
    """Copy constructor for TensorNormalizer."""

  @abc.abstractmethod
  def _create_variables(self):
    """Uses self._scope and creates all variables needed for the normalizer."""

  @property
  @abc.abstractmethod
  def variables(self):
    """Returns a tuple of tf variables owned by this normalizer."""

  @abc.abstractmethod
  def _update_ops(self, tensor, outer_dims):
    """Returns a list of ops which update normalizer variables for tensor.

    Args:
      tensor: The tensor, whose batch statistics to use for updating
        normalization variables.
      outer_dims: The dimensions to consider batch dimensions, to reduce over.
    """

  @abc.abstractmethod
  def _get_mean_var_estimates(self):
    """Returns this normalizer's current estimates for mean & variance."""

  def update(self, tensor, outer_dims=(0,)):
    """Updates tensor normalizer variables."""
    tensor = tf.nest.map_structure(
        lambda t: tf.cast(t, tf.float32),
        tensor)
    return tf.group(self._update_ops(tensor, outer_dims))

  def normalize(self,
                tensor,
                clip_value=5.0,
                center_mean=True,
                variance_epsilon=1e-3):
    """Applies normalization to tensor.

    Args:
      tensor: Tensor to normalize.
      clip_value: Clips normalized observations between +/- this value if
        clip_value > 0, otherwise does not apply clipping.
      center_mean: If true, subtracts off mean from normalized tensor.
      variance_epsilon: Epsilon to avoid division by zero in normalization.

    Returns:
      normalized_tensor: Tensor after applying normalization.
    """
    tf.nest.assert_same_structure(tensor, self._tensor_spec)
    tensor = tf.nest.map_structure(lambda t: tf.cast(t, tf.float32), tensor)

    with tf.name_scope(self._scope + '/normalize'):
      mean_estimate, var_estimate = self._get_mean_var_estimates()
      mean = (
          mean_estimate if center_mean else tf.nest.map_structure(
              tf.zeros_like, mean_estimate))

      def _normalize_single_tensor(single_tensor, single_mean, single_var):
        return tf.nn.batch_normalization(
            single_tensor, single_mean, single_var,
            offset=None, scale=None, variance_epsilon=variance_epsilon,
            name='normalized_tensor')

      normalized_tensor = nest.map_structure_up_to(self._tensor_spec,
                                                   _normalize_single_tensor,
                                                   tensor, mean, var_estimate)

      if clip_value > 0:
        def _clip(t):
          return tf.clip_by_value(t, -clip_value, clip_value,
                                  name='clipped_normalized_tensor')

        normalized_tensor = tf.nest.map_structure(_clip, normalized_tensor)

    return normalized_tensor


class EMATensorNormalizer(TensorNormalizer):
  """TensorNormalizer with exponential moving avg. mean and var estimates."""

  def __init__(self,
               tensor_spec,
               scope='normalize_tensor',
               norm_update_rate=0.001,):
    super(EMATensorNormalizer, self).__init__(tensor_spec, scope)
    self._norm_update_rate = norm_update_rate

  def copy(self, scope=None):
    """Copy constructor for EMATensorNormalizer."""
    scope = scope if scope is not None else self._scope
    return EMATensorNormalizer(
        self._tensor_spec, scope=scope, norm_update_rate=self._norm_update_rate)

  def _create_variables(self):
    """Creates the variables needed for EMATensorNormalizer."""
    self._mean_moving_avg = tf.nest.map_structure(
        lambda spec: create_variable('mean', 0, spec.shape, tf.float32),
        self._tensor_spec)
    self._var_moving_avg = tf.nest.map_structure(
        lambda spec: create_variable('var', 1, spec.shape, tf.float32),
        self._tensor_spec)

  @property
  def variables(self):
    """Returns a tuple of tf variables owned by this EMATensorNormalizer."""
    return self._mean_moving_avg, self._var_moving_avg

  def _update_ops(self, tensor, outer_dims):
    """Returns a list of update obs for EMATensorNormalizer mean and var.

    This normalizer tracks the mean & variance of the dimensions of the input
    tensor using an exponential moving average. The batch mean comes from just
    the batch statistics, and the batch variance comes from the squared
    difference of tensor values from the current mean estimate. The mean &
    variance are both updated as (old_value + update_rate *
    (batch_value - old_value)).

    Args:
      tensor: The tensor of values to be normalized.
      outer_dims: The batch dimensions over which to compute normalization
        statistics.
    Returns:
      A list of ops, which when run will update all necessary normaliztion
      variables.
    """
    def _tensor_update_ops(single_tensor, mean_var, var_var):
      """Make update ops for a single non-nested tensor."""
      # Take the moments across batch dimension. Calculate variance with
      #   moving avg mean, so that this works even with batch size 1.
      mean = tf.reduce_mean(input_tensor=single_tensor, axis=outer_dims)
      var = tf.reduce_mean(
          input_tensor=tf.square(single_tensor - mean_var),
          axis=outer_dims)

      # Ops to update moving average. Make sure that all stats are computed
      #   before updates are performed.
      with tf.control_dependencies([mean, var]):
        update_ops = [
            mean_var.assign_add(self._norm_update_rate * (mean - mean_var)),
            var_var.assign_add(self._norm_update_rate * (var - var_var))
        ]
      return update_ops

    # Aggregate update ops for all parts of potentially nested tensor.
    updates = tf.nest.map_structure(
        _tensor_update_ops, tensor, self._mean_moving_avg, self._var_moving_avg)
    all_update_ops = tf.nest.flatten(updates)

    return all_update_ops

  def _get_mean_var_estimates(self):
    """Returns EMANormalizer's current estimates for mean & variance."""
    return self._mean_moving_avg, self._var_moving_avg


class StreamingTensorNormalizer(TensorNormalizer):
  """Normalizes mean & variance based on full history of tensor values."""

  def _create_variables(self):
    """Uses self._scope and creates all variables needed for the normalizer."""
    self._count = tf.nest.map_structure(
        lambda spec: create_variable('count', 1e-8, spec.shape, tf.float32),
        self._tensor_spec)
    self._mean_sum = tf.nest.map_structure(
        lambda spec: create_variable('mean_sum', 0, spec.shape, tf.float32),
        self._tensor_spec)
    self._var_sum = tf.nest.map_structure(
        lambda spec: create_variable('var_sum', 0, spec.shape, tf.float32),
        self._tensor_spec)

  def copy(self, scope=None):
    """Copy constructor for StreamingTensorNormalizer."""
    scope = scope if scope is not None else self._scope
    return StreamingTensorNormalizer(self._tensor_spec, scope=scope)

  @property
  def variables(self):
    """Returns a tuple of tf variables owned by this normalizer."""
    return self._count, self._mean_sum, self._var_sum

  def _update_ops(self, tensor, outer_dims):
    """Returns a list of ops which update normalizer variables for tensor.

    This normalizer computes the absolute mean of all observed tensor values,
    and keeps a biased estimator of variance, by summing all observed mean and
    variance values and dividing the sum by the count of samples seen.

    Args:
      tensor: The tensor of values to be normalized.
      outer_dims: The batch dimensions over which to compute normalization
        statistics.
    Returns:
      A list of ops, which when run will update all necessary normaliztion
      variables.
    """
    def _tensor_update_ops(single_tensor, single_mean_est, count_var, mean_var,
                           var_var):
      """Make update ops for a single non-nested tensor."""
      # Num samples in batch is the product of batch dimensions.
      num_samples = tf.cast(
          tf.reduce_prod(
              input_tensor=tf.gather(
                  tf.shape(input=single_tensor), outer_dims)),
          tf.float32)
      mean_sum = tf.reduce_sum(input_tensor=single_tensor, axis=outer_dims)
      var_sum = tf.reduce_sum(
          input_tensor=tf.square(
              single_tensor - single_mean_est), axis=outer_dims)

      # Ops to update streaming norm. Make sure that all stats are computed
      #   before updates are performed.
      with tf.control_dependencies([num_samples, mean_sum, var_sum]):
        update_ops = [
            tf.compat.v1.assign_add(
                count_var,
                tf.ones_like(count_var) * num_samples,
                name='update_count'),
            tf.compat.v1.assign_add(
                mean_var, mean_sum, name='update_mean_sum'),
            tf.compat.v1.assign_add(
                var_var, var_sum, name='update_var_sum'),
        ]
      return update_ops

    mean_estimate, _ = self._get_mean_var_estimates()

    # Aggregate update ops for all parts of potentially nested tensor.
    updates = tf.nest.map_structure(
        _tensor_update_ops, tensor, mean_estimate, self._count, self._mean_sum,
        self._var_sum)
    all_update_ops = tf.nest.flatten(updates)

    return all_update_ops

  def _get_mean_var_estimates(self):
    """Returns this normalizer's current estimates for mean & variance."""
    mean_estimate = nest.map_structure_up_to(
        self._tensor_spec, lambda a, b: a / b, self._mean_sum, self._count)
    var_estimate = nest.map_structure_up_to(
        self._tensor_spec, lambda a, b: a / b, self._var_sum, self._count)
    return mean_estimate, var_estimate
