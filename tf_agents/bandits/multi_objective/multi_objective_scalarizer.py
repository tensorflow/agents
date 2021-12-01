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

"""Classes implementing various scalarizations of multiple objectives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import copy
from typing import Dict, NamedTuple, Sequence, Union, Callable, Optional

import numpy as np
import six
import tensorflow.compat.v2 as tf

ScalarFloat = Union[float, np.float16, np.float32, np.float64]


def _validate_scalarization_parameter_shape(
    multi_objectives: tf.Tensor, params: Dict[str, Union[Sequence[ScalarFloat],
                                                         tf.Tensor]]):
  """A private helper that validates the shapes of scalarization parameters.

  Every scalarization parameter in the input dictionary is either a 1-D tensor
  or `Sequence`, or a tensor whose shape matches the shape of the input
  `multi_objectives` tensor. This is invoked by the `Scalarizer.call` method.

  Args:
    multi_objectives: A `tf.Tensor` representing the multiple objectives to be
      scalarized.
    params: A dictionary from parameter names to parameter values (`Sequence` or
      `tf.Tensor`).

  Raises:
    tf.errors.InvalidArgumentError: if any scalarization parameter is not a 1-D
      tensor or `Sequence`, and has shape that does not match the shape of
      `multi_objectives`.
  """
  for param_name, param_value in params.items():
    param_shape = tf.convert_to_tensor(param_value).shape
    if param_shape.rank != 1 and not multi_objectives.shape.is_compatible_with(
        param_shape):
      raise ValueError(
          'The shape of multi_objectives: {} does not match the shape of '
          'scalarization parameter: {}, which is {}'.format(
              multi_objectives.shape, param_name, param_shape))


# TODO(b/202447704): Update to use public Protocol when available.
class ScalarizerTraceType:
  """Class outlining the default Tracing Protocol for Scalarizer.

  If included as an argument, corresponding tf.function will always retrace for
  each usage.

  Derived classes can override this behavior by specifying their own Tracing
  Protocol.
  """

  def is_subtype_of(self, _):
    return False

  def most_specific_common_supertype(self, _):
    return None

  def __hash__(self):
    return id(self)

  def __eq__(self, _):
    return False

  def __repr__(self):
    return 'ScalarizerTraceType()'


@six.add_metaclass(abc.ABCMeta)
class Scalarizer(tf.Module):
  """Abstract base class for different Scalarizers.

  The Scalarizer class is a callable that transforms multiple objectives into
  a single scalar reward.
  """

  def __init__(self, num_of_objectives: int):
    """Initialize the Scalarizer.

    Args:
      num_of_objectives: A non-negative integer indicating the number of
        objectives to scalarize.

    Raises:
      ValueError: if `not isinstance(num_of_objectives, int)`.
      ValueError: if `num_of_objectives < 2`.
    """
    if not isinstance(num_of_objectives, int):
      raise ValueError(
          'Scalarizer should be initialized with an integer representing the '
          'number of objectives, but the type of the input is {}.'.format(
              type(num_of_objectives)))
    if num_of_objectives < 2:
      raise ValueError(
          'Scalarizer should be used with at least two objectives, but only {}'
          ' are given.'.format(num_of_objectives))
    self._num_of_objectives = num_of_objectives

  def __call__(self, multi_objectives: tf.Tensor) -> tf.Tensor:
    """Returns a single reward by scalarizing multiple objectives.

    Args:
      multi_objectives: A `Tensor` of shape [batch_size, number_of_objectives],
        where each column represents an objective.
    Returns: A `Tensor` of shape [batch_size] representing scalarized rewards.

    Raises:
      ValueError: if `multi_objectives.shape.rank != 2`.
      ValueError: if
        `multi_objectives.shape.dims[1] != self._num_of_objectives`.
    """
    if multi_objectives.shape.rank != 2:
      raise ValueError('The rank of the input should be 2, but is {}'.format(
          multi_objectives.shape.rank))
    if multi_objectives.shape.dims[1] != self._num_of_objectives:
      raise ValueError(
          'The number of input objectives should be {}, but is {}.'.format(
              self._num_of_objectives, multi_objectives.shape.dims[1]))
    return self._scalarize(self._transform(multi_objectives))

  def _validate_scalarization_parameters(self, params: Dict[str, tf.Tensor]):
    """Validates the scalarization parameters.

    Each scalarization parameter in the input dictionary should be a rank-2
    tensor, and the last dimension size should match `self._num_of_objectives`.

    Args:
      params: A dictionary from parameter names to parameter tensors.

    Raises:
      ValueError: if any input scalarization parameter violates any of the
      required properties.
    """
    for param_name, param in params.items():
      if param.shape.rank != 2:
        raise ValueError(
            'Scalarization parameter: {} should be a rank-2 tensor with shape '
            '[batch_size, num_of_objectives], but found to be: {}'.format(
                param_name, param))
      elif param.shape.dims[-1] != self._num_of_objectives:
        raise ValueError(
            'The number of objectives in scalarization parameter: {} should '
            'be {}, but found to be {}.'.format(param_name,
                                                self._num_of_objectives,
                                                param.shape.dims[-1]))

  # Identity transform. Subclasses can override.
  def _transform(self, multiobjectives: tf.Tensor) -> tf.Tensor:
    return multiobjectives

  # Subclasses must implement these methods.
  @abc.abstractmethod
  def _scalarize(self, transformed_multi_objectives: tf.Tensor) -> tf.Tensor:
    """Implementation of scalarization logic by subclasses."""

  @abc.abstractmethod
  def set_parameters(self, **kwargs):
    """Setter method for scalarization parameters."""

  def __tf_tracing_type__(self, _):
    """Default TraceType Protocol for Scalarizaer Class."""
    return ScalarizerTraceType()


class LinearScalarizer(Scalarizer):
  """Scalarizes multple objectives by a linear combination."""

  def __init__(self,
               weights: Sequence[ScalarFloat],
               multi_objective_transform: Optional[Callable[[tf.Tensor],
                                                            tf.Tensor]] = None):
    """Initialize the LinearScalarizer.

    Args:
      weights: A `Sequence` of weights for linearly combining the objectives.
      multi_objective_transform: A `Optional` `Callable` that takes in a
        `tf.Tensor` of multiple objective values and applies an arbitrary
        transform that returns a `tf.Tensor` of transformed multiple objectives.
        This transform is applied before the linear scalarization. The transform
        should apply to each objective so that the shape of the multiobjectives
        and the transformed multiobjectives are equal. This is verified in
        _validate_scalarization_parameter_shape via __call__.
    """
    self._weights = copy.deepcopy(weights)
    self._transformer = multi_objective_transform
    super(LinearScalarizer, self).__init__(len(self._weights))

  def _transform(self, multiobjectives: tf.Tensor) -> tf.Tensor:
    return multiobjectives if self._transformer is None else self._transformer(
        multiobjectives)

  def _scalarize(self, transformed_multi_objectives: tf.Tensor) -> tf.Tensor:
    _validate_scalarization_parameter_shape(transformed_multi_objectives,
                                            {'weights': self._weights})
    return tf.reduce_sum(transformed_multi_objectives * self._weights, axis=1)

  def set_parameters(self, weights: tf.Tensor):
    """Set the scalarization parameter of the LinearScalarizer.

    Args:
      weights: A a rank-2 `tf.Tensor` of weights shaped as [batch_size,
        self._num_of_objectives], where `batch_size` should match the batch size
        of the `multi_objectives` passed to the scalarizer call.

    Raises:
      ValueError: if the weights tensor is not rank-2, or has a last dimension
      size that does not match `self._num_of_objectives`.
    """
    self._validate_scalarization_parameters({'weights': weights})
    self._weights = weights


class ChebyshevScalarizer(Scalarizer):
  """Implement the Chebyshev Scalarization.

  Given a vector of (at least two) objectives M, a weight vector W, and a
  reference point Z, all having the same dimension, the Chebyshev scalarization
  is defined as:

  ```min_{i} W_i * (M_i - Z_i).```

  Note that reference point Z is expected to be dominated by all Pareto-optimal
  objective vectors.
  """

  def __init__(self, weights: Sequence[ScalarFloat],
               reference_point: Sequence[ScalarFloat]):
    """Initialize the ChebyshevScalarizer.

    Args:
      weights: A `Sequence` of weights.
      reference_point: A `Sequence` of coordinates for the reference point.

    Raises:
      ValueError: if `len(weights) != len(reference_point)`.
    """
    if len(weights) != len(reference_point):
      raise ValueError(
          'weights has {} elements but reference_point has {}.'.format(
              len(weights), len(reference_point)))
    self._weights = copy.deepcopy(weights)
    self._reference_point = reference_point
    super(ChebyshevScalarizer, self).__init__(len(self._weights))

  def _scalarize(self, transformed_multi_objectives: tf.Tensor) -> tf.Tensor:
    _validate_scalarization_parameter_shape(transformed_multi_objectives, {
        'weights': self._weights,
        'reference_point': self._reference_point
    })
    return tf.reduce_min(
        (transformed_multi_objectives - self._reference_point) * self._weights,
        axis=-1)

  def set_parameters(self, weights: tf.Tensor, reference_point: tf.Tensor):
    """Set the scalarization parameters for the ChebyshevScalarizer.

    Args:
      weights: A rank-2 `tf.Tensor` of weights shaped as [batch_size,
        self._num_of_objectives], where `batch_size` should match the batch size
        of the `multi_objectives` passed to the scalarizer call.
      reference_point: A `tf.Tensor` of coordinates for the reference point that
        must satisfy the same rank and shape requirements as `weights`.

    Raises:
      ValueError: if any input scalarization parameter tensor is not rank-2, or
      has a last dimension size that does not match `self._num_of_objectives`.
    """
    self._validate_scalarization_parameters({
        'weights': weights,
        'reference_point': reference_point
    })
    self._weights = weights
    self._reference_point = reference_point


class HyperVolumeScalarizer(Scalarizer):
  """Implement the hypervolume scalarization.

  Given a vector of (at least two) objectives M, a unit-length vector V with
  non-negative coordinates, a slope vector A, and an offset vector B, all having
  the same dimension, the hypervolume scalarization of M is defined as:

  ```min_{i: V_i > 0} max(A_i * M_i + B_i, 0) / V_i.```

  See https://arxiv.org/abs/2006.04655 for more details.
  Note that it is recommended for the user to set A_i and B_i in such a way to
  ensure non-negativity of the transformed objectives.
  """
  DIRECTION_KEY = 'direction'
  SLOPE_KEY = 'slope'
  OFFSET_KEY = 'offset'
  PARAMS = NamedTuple('PARAMS', [(SLOPE_KEY, ScalarFloat),
                                 (OFFSET_KEY, ScalarFloat)])
  ALMOST_ZERO = 1e-16

  def __init__(self,
               direction: Sequence[ScalarFloat],
               transform_params: Sequence[PARAMS],
               multi_objective_transform: Optional[Callable[
                   [tf.Tensor, Sequence[ScalarFloat], Sequence[ScalarFloat]],
                   tf.Tensor]] = None):
    """Initialize the HyperVolumeScalarizer.

    Args:
      direction: A `Sequence` representing a directional vector, which will be
        normalized to have unit length. Coordinates of the normalized direction
        whose absolute values are less than `HyperVolumeScalarizer.ALMOST_ZERO`
        will be considered zeros.
      transform_params: A `Sequence` of namedtuples
        `HyperVolumeScalarizer.PARAMS`, each containing a slope and an offset
        for transforming an objective to be non-negative.
      multi_objective_transform: A `Optional` `Callable` that takes in a
        `tf.Tensor` of multiple objective values, a `Sequence` of slopes, and a
        `Sequence` of offsets, and returns a `tf.Tensor` of transformed multiple
        objectives. If unset, the transform is defaulted to the standard
        transform multiple_objectives * slopes + offsets.

    Raises:
      ValueError: if `any([x < 0 for x in direction])`.
      ValueError: if the 2-norm of `direction` is less than
        `HyperVolumeScalarizer.ALMOST_ZERO`.
      ValueError: if `len(transform_params) != len(self._direction)`.
    """
    if any([x < 0 for x in direction]):
      raise ValueError(
          'direction should be in the positive orthant, but has negative '
          'coordinates: [{}].'.format(', '.join(map(str, direction))))
    length = np.sqrt(sum([x * x for x in direction]))
    if length < self.ALMOST_ZERO:
      raise ValueError(
          'direction found to be a nearly-zero vector, but should not be.')
    self._direction = [x / length for x in direction]
    if len(transform_params) != len(self._direction):
      raise ValueError(
          'direction has {} elements but transform_params has {}.'.format(
              len(direction), len(transform_params)))
    self._slopes, self._offsets = zip(
        *[(p.slope, p.offset) for p in transform_params])

    if multi_objective_transform is None:
      multi_objective_transform = self._default_hv_transform

    self._transformer = multi_objective_transform
    super(HyperVolumeScalarizer, self).__init__(len(self._direction),)

  @staticmethod
  def _default_hv_transform(multi_objectives: tf.Tensor,
                            slopes: Sequence[ScalarFloat],
                            offsets: Sequence[ScalarFloat]) -> tf.Tensor:
    return multi_objectives * slopes + offsets

  def _transform(self, multi_objectives: tf.Tensor) -> tf.Tensor:
    _validate_scalarization_parameter_shape(
        multi_objectives, {
            self.DIRECTION_KEY: self._direction,
            self.SLOPE_KEY: self._slopes,
            self.OFFSET_KEY: self._offsets
        })
    return self._transformer(multi_objectives, self._slopes, self._offsets)

  def _scalarize(self, transformed_multi_objectives: tf.Tensor) -> tf.Tensor:
    transformed_multi_objectives = tf.maximum(transformed_multi_objectives, 0)
    nonzero_mask = tf.broadcast_to(
        tf.cast(tf.abs(self._direction) >= self.ALMOST_ZERO, dtype=tf.bool),
        tf.shape(transformed_multi_objectives))
    return tf.reduce_min(
        tf.where(nonzero_mask, transformed_multi_objectives / self._direction,
                 transformed_multi_objectives.dtype.max),
        axis=1)

  def set_parameters(self, direction: tf.Tensor,
                     transform_params: Dict[str, tf.Tensor]):
    """Set the scalarization parameters for the HyperVolumeScalarizer.

    Args:
      direction: A `tf.Tensor` representing a directional vector, which will be
        normalized to have unit length. Coordinates of the normalized direction
        whose absolute values are less than `HyperVolumeScalarizer.ALMOST_ZERO`
        will be considered zeros. It must be rank-2 and shaped as [batch_size,
        self._num_of_objectives], where `batch_size` should match the batch size
        of the multi objectives passed to the scalarizer call.
      transform_params: A dictionary mapping `self.SLOPE_KEY` and/or
        `self.OFFSET_KEY` to `tf.Tensor`, representing the slope and the offset
        parameters for transforming an objective to be non-negative. These
        tensors must satisfy the same rank and shape requirements as
        `direction`.

    Raises:
      ValueError: if any input scalarization parameter tensor is not rank-2, or
      has a last dimension size that does not match `self._num_of_objectives`.
    """
    self._validate_scalarization_parameters({self.DIRECTION_KEY: direction})
    self._direction = direction
    for key, param in transform_params.items():
      if key == self.SLOPE_KEY:
        self._validate_scalarization_parameters({key: param})
        self._slopes = param
      elif key == self.OFFSET_KEY:
        self._validate_scalarization_parameters({key: param})
        self._offsets = param
      else:
        raise ValueError(
            'All transform_params keys should be {} or {}, but one key is not:'
            ' {}'.format(self.SLOPE_KEY, self.OFFSET_KEY, key))
