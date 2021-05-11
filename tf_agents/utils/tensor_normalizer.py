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

"""Tensor statistics and normalization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Tuple

import six
import tensorflow as tf

from tf_agents.specs import tensor_spec as tensor_spec_lib
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils

create_variable = common.create_variable

_EPS = 1e-8

# Use tf.float32 by default
_DTYPE_CONVERSION = {
    tf.int32: tf.float32,
    tf.int64: tf.float64,
    tf.float32: tf.float32,
    tf.float64: tf.float64
}


@six.add_metaclass(abc.ABCMeta)
class TensorNormalizer(tf.Module):
  """Encapsulates tensor normalization and owns normalization variables.

  Example usage:

  ```
  tensor_normalizer = StreamingTensorNormalizer(
      tf.TensorSpec([], tf.float32))
  observation_list = [list of float32 scalars or batches]
  normalized_list = []

  for o in observation_list:
    normalized_list.append(tensor_normalizer.normalize(o))
    tensor_normalizer.update(o)
  ```

  For float64 inputs do:
  ```
  tensor_normalizer = StreamingTensorNormalizer(
      tf.TensorSpec([], tf.float64), dtype=tf.float64)
  observation_list = [list of float64 scalars or batches]

  for o in observation_list:
    normalized_list.append(tensor_normalizer.normalize(o))
    tensor_normalizer.update(o)
  """

  def __init__(self, tensor_spec, scope='normalize_tensor'):
    """Initialize TensorNormalizer.

    Args:
      tensor_spec: The specs of the tensors to normalize.
      scope: Scope for the `tf.Module`.
    """

    super(TensorNormalizer, self).__init__(name=scope)
    self._scope = scope
    self._tensor_spec = tensor_spec
    self._flat_variable_spec = [
        self._map_spec_dtype(s) for s in tf.nest.flatten(tensor_spec)
    ]
    self._create_variables()

  def map_dtype(self, dtype):
    # Use tf.float32 by default
    return _DTYPE_CONVERSION.get(dtype, tf.float32)

  def _map_spec_dtype(self, spec):
    return tensor_spec_lib.with_dtype(spec, self.map_dtype(spec.dtype))

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
    """Returns this normalizer's current estimates for mean & var (flat)."""

  def update(self, tensor, outer_dims=(0,)):
    """Updates tensor normalizer variables."""
    nest_utils.assert_matching_dtypes_and_inner_shapes(
        tensor,
        self._tensor_spec,
        caller=self,
        tensors_name='tensor',
        specs_name='tensor_spec')
    tensor = tf.nest.map_structure(
        lambda t: tf.cast(t, self.map_dtype(t.dtype)), tensor)

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
    nest_utils.assert_matching_dtypes_and_inner_shapes(
        tensor, self._tensor_spec, caller=self,
        tensors_name='tensors', specs_name='tensor_spec')
    tensor = [
        tf.cast(t, self.map_dtype(t.dtype)) for t in tf.nest.flatten(tensor)
    ]

    with tf.name_scope(self._scope + '/normalize'):
      mean_estimate, var_estimate = self._get_mean_var_estimates()
      mean = (
          mean_estimate if center_mean else tf.nest.map_structure(
              tf.zeros_like, mean_estimate))

      def _normalize_single_tensor(single_tensor, single_mean, single_var):
        return tf.nn.batch_normalization(

            single_tensor,
            single_mean,
            single_var,
            offset=None,
            scale=None,
            variance_epsilon=variance_epsilon,
            name='normalized_tensor')

      normalized_tensor = nest_utils.map_structure_up_to(
          self._flat_variable_spec,
          _normalize_single_tensor,
          tensor,
          mean,
          var_estimate,
          check_types=False)

      if clip_value > 0:

        def _clip(t):
          return tf.clip_by_value(
              t, -clip_value, clip_value, name='clipped_normalized_tensor')

        normalized_tensor = tf.nest.map_structure(_clip, normalized_tensor)

    normalized_tensor = tf.nest.pack_sequence_as(self._tensor_spec,
                                                 normalized_tensor)
    normalized_tensor = tf.nest.map_structure(
        lambda t, spec: tf.cast(t, spec.dtype), normalized_tensor,
        self._tensor_spec)
    return normalized_tensor


class EMATensorNormalizer(TensorNormalizer):
  """TensorNormalizer with exponential moving avg. mean and var estimates."""

  def __init__(
      self,
      tensor_spec,
      scope='normalize_tensor',
      norm_update_rate=0.001):
    super(EMATensorNormalizer, self).__init__(tensor_spec, scope)
    self._norm_update_rate = norm_update_rate

  def _create_variables(self):
    """Creates the variables needed for EMATensorNormalizer."""
    self._mean_moving_avg = tf.nest.map_structure(
        lambda spec: create_variable('mean', 0, spec.shape, spec.dtype),
        self._flat_variable_spec)
    self._var_moving_avg = tf.nest.map_structure(
        lambda spec: create_variable('var', 1, spec.shape, spec.dtype),
        self._flat_variable_spec)

  @property
  def variables(self):
    """Returns a tuple of tf variables owned by this EMATensorNormalizer."""
    return (tf.nest.pack_sequence_as(self._tensor_spec, self._mean_moving_avg),
            tf.nest.pack_sequence_as(self._tensor_spec, self._var_moving_avg))

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
      mean = tf.reduce_mean(single_tensor, axis=outer_dims)
      var = tf.reduce_mean(
          tf.math.squared_difference(single_tensor, mean_var), axis=outer_dims)

      # Ops to update moving average. Make sure that all stats are computed
      #   before updates are performed.
      with tf.control_dependencies([mean, var]):
        update_ops = [
            mean_var.assign_add(self._norm_update_rate * (mean - mean_var)),
            var_var.assign_add(self._norm_update_rate * (var - var_var))
        ]
      return update_ops

    # Aggregate update ops for all parts of potentially nested tensor.
    tensor = tf.nest.flatten(tensor)
    updates = tf.nest.map_structure(_tensor_update_ops, tensor,
                                    self._mean_moving_avg, self._var_moving_avg)
    all_update_ops = tf.nest.flatten(updates)

    return all_update_ops

  def _get_mean_var_estimates(self):
    """Returns EMANormalizer's current estimates for mean & variance."""
    return self._mean_moving_avg, self._var_moving_avg


class StreamingTensorNormalizer(TensorNormalizer):
  """Normalizes mean & variance based on full history of tensor values."""

  def _create_variables(self):
    """Create all variables needed for the normalizer."""
    self._count = [
        create_variable(
            'count_%d' % i, _EPS, spec.shape, spec.dtype, trainable=False)
        for i, spec in enumerate(self._flat_variable_spec)
    ]
    self._avg = [
        create_variable(
            'avg_%d' % i, 0, spec.shape, spec.dtype, trainable=False)
        for i, spec in enumerate(self._flat_variable_spec)
    ]
    self._m2 = [
        create_variable(
            'm2_%d' % i, 0, spec.shape, spec.dtype, trainable=False)
        for i, spec in enumerate(self._flat_variable_spec)
    ]
    self._m2_carry = [
        create_variable(
            'm2_carry_%d' % i, 0, spec.shape, spec.dtype, trainable=False)
        for i, spec in enumerate(self._flat_variable_spec)
    ]

  @property
  def variables(self):
    """Returns a tuple of nested TF Variables owned by this normalizer."""
    return (tf.nest.pack_sequence_as(self._tensor_spec, self._count),
            tf.nest.pack_sequence_as(self._tensor_spec, self._avg),
            tf.nest.pack_sequence_as(self._tensor_spec, self._m2),
            tf.nest.pack_sequence_as(self._tensor_spec, self._m2_carry))

  def _update_ops(self, tensors, outer_dims):
    """Returns a list of ops which update normalizer variables for tensor.

    Args:
      tensors: The tensors of values to be normalized.
      outer_dims: Ignored.  The batch dimensions are extracted by comparing
        the associated tensor with the specs.

    Returns:
      A list of ops, which when run will update all necessary normaliztion
      variables.
    """
    del outer_dims

    outer_shape = nest_utils.get_outer_shape(tensors, self._tensor_spec)
    outer_rank_static = tf.compat.dimension_value(outer_shape.shape[0])
    outer_axes = (
        list(range(outer_rank_static)) if outer_rank_static is not None else
        tf.range(tf.size(outer_shape)))

    outer_n = tf.reduce_prod(outer_shape)

    flat_tensors = tf.nest.flatten(tensors)

    update_ops = []

    for i, t in enumerate(flat_tensors):
      n_a = tf.cast(outer_n, self._count[i].dtype)
      avg_a = tf.math.reduce_mean(t, outer_axes)
      m2_a = tf.math.reduce_sum(tf.math.squared_difference(t, avg_a),
                                outer_axes)
      n_b = self._count[i]
      avg_b = self._avg[i]
      m2_b = self._m2[i]
      m2_b_c = self._m2_carry[i]

      n_ab, avg_ab, m2_ab, m2_ab_c = parallel_variance_calculation(
          n_a, avg_a, m2_a, n_b, avg_b, m2_b, m2_b_c)
      with tf.control_dependencies([n_ab, avg_ab, m2_ab, m2_ab_c]):
        update_ops.extend([
            self._count[i].assign(n_ab),
            self._avg[i].assign(avg_ab),
            self._m2[i].assign(m2_ab),
            self._m2_carry[i].assign(m2_ab_c),
        ])

    return update_ops

  def _get_mean_var_estimates(self):
    """Returns this normalizer's current estimates for mean & var (flat)."""
    var = [m2_ab / n_ab for (m2_ab, n_ab) in zip(self._m2, self._count)]
    return (self._avg, var)

  def reset(self):
    """Reset the count, mean and variance to its initial state."""
    reset_ops = []
    for i in range(len(self._count)):
      reset_ops.extend([
          self._count[i].assign(_EPS * tf.ones_like(self._count[i])),
          self._avg[i].assign(tf.zeros_like(self._avg[i])),
          self._m2[i].assign(tf.zeros_like(self._m2[i])),
          self._m2_carry[i].assign(tf.zeros_like(self._m2_carry[i])),
      ])

    return reset_ops


def parallel_variance_calculation(
    n_a: types.Int,
    avg_a: types.Float,
    m2_a: types.Float,
    n_b: types.Int,
    avg_b: types.Float,
    m2_b: types.Float,
    m2_b_c: types.Float
) -> Tuple[types.Int, types.Float, types.Float, types.Float]:
  """Calculate the sufficient statistics (average & second moment) of two sets.

  For better precision if sets are of different sizes, `a` should be the smaller
  and `b` the bigger.

  For more details, see the parallel algorithm of Chan et al. at:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

  For stability we use `kahan_summation` to accumulate second moments.

  Takes in the sufficient statistics for sets `A` and `B` and calculates the
  variance and sufficient statistics for the union of `A` and `B`.

  If e.g. `B` is a single observation `x_b`, use `n_b=1`, `avg_b = x_b`, and
  `m2_b = 0`.

  To get `avg_a` and `m2_a` from a tensor `x` of shape `[n_a, ...]`, use:

  ```
  n_a = tf.shape(x)[0]
  avg_a = tf.math.reduce_mean(x, axis=[0])
  m2_a = tf.math.reduce_sum(tf.math.squared_difference(t, avg_a), axis=[0])

  ```

  Args:
    n_a: Number of elements in `A`.
    avg_a: The sample average of `A`.
    m2_a: The sample second moment of `A`.
    n_b: Number of elements in `B`.
    avg_b: The sample average of `B`.
    m2_b: The sample second moment of `B`.
    m2_b_c: Carry for accumulation of the sample second moment of `B`.

  Returns:
    A tuple `(n_ab, avg_ab, m2_ab, m2_ab_c)` such that `var_ab`,
    the variance of `A|B`, may be calculated via `var_ab = m2_ab / n_ab`,
    and the sample variance  as`sample_var_ab = m2_ab / (n_ab - 1)`.
  """
  n_ab = n_a + n_b
  delta = avg_b - avg_a
  s_delta = delta * n_b / n_ab
  avg_ab = avg_a + s_delta
  m2_ab, m2_ab_c = kahan_summation(m2_b, m2_b_c, m2_a + (delta * n_a * s_delta))
  return n_ab, avg_ab, m2_ab, m2_ab_c


def kahan_summation(accumulator: types.Float,
                    carry: types.Float,
                    value: types.Float
                    ) -> Tuple[types.Float, types.Float]:
  """Calculate stable acculated sum using compensation for low-bits.

  For more details:
  https://en.wikipedia.org/wiki/Kahan_summation_algorithm


  Args:
    accumulator: Accumulator.
    carry: Carry for lost low-order bits.
    value: New value to accumlate.

  Returns:
    A tuple `(accumulator, carry)` such that `accumulator` is the new
    accumulated value and `carry` is the new compensation value.
  """
  delta = value - carry
  new_accumulator = accumulator + delta
  carry = (new_accumulator - accumulator) - delta
  return new_accumulator, carry
