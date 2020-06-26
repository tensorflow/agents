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

"""Classes implementing various scalarizations of multiple objectives."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
import copy
from typing import NamedTuple, Sequence, Union

import numpy as np
import six
import tensorflow.compat.v2 as tf

ScalarFloat = Union[float, np.float16, np.float32, np.float64]


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
    return self.call(multi_objectives)

  # Subclasses must implement these methods.
  @abc.abstractmethod
  def call(self, multi_objectives: tf.Tensor) -> tf.Tensor:
    """Implementation of scalarization logic by subclasses."""


class LinearScalarizer(Scalarizer):
  """Scalarizes multple objectives by a linear combination."""

  def __init__(self, weights: Sequence[ScalarFloat]):
    """Initialize the LinearScalarizer.

    Args:
      weights: A `Sequence` of weights for linearly combining the objectives.

    Raises:
      TypeError: if `not isinstance(weights, Sequence)`.
    """
    if not isinstance(weights, Sequence):
      raise TypeError(
          'weights should be a Sequence, but is {}.'.format(weights))
    self._weights = copy.deepcopy(weights)
    super(LinearScalarizer, self).__init__(len(self._weights))

  def call(self, multi_objectives: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(multi_objectives * self._weights, axis=1)


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
      TypeError: if `not isinstance(weights, Sequence)`.
      TypeError: if `not isinstance(reference_point, Sequence)`.
      ValueError: if `len(weights) != len(reference_point)`.
    """
    if not isinstance(weights, Sequence):
      raise TypeError('weights should be a Sequence, but is {}'.format(weights))
    if not isinstance(reference_point, Sequence):
      raise TypeError(
          'reference should be a Sequence, but is {}'.format(reference_point))
    if len(weights) != len(reference_point):
      raise ValueError(
          'weights has {} elements but reference_point has {}.'.format(
              len(weights), len(reference_point)))
    self._weights = copy.deepcopy(weights)
    self._reference_point = copy.deepcopy(reference_point)
    super(ChebyshevScalarizer, self).__init__(len(self._weights))

  def call(self, multi_objectives: tf.Tensor) -> tf.Tensor:
    return tf.reduce_min(
        (multi_objectives - self._reference_point) * self._weights, axis=1)


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

  PARAMS = NamedTuple('PARAMS', [('slope', ScalarFloat),
                                 ('offset', ScalarFloat)])
  ALMOST_ZERO = 1e-16

  def __init__(self, direction: Sequence[ScalarFloat],
               transform_params: Sequence[PARAMS]):
    """Initialize the HyperVolumeScalarizer.

    Args:
      direction: A `Sequence` representing a directional vector, which will be
        normalized to have unit length. Coordinates of the normalized direction
        whose absolute values are less than `HyperVolumeScalarizer.ALMOST_ZERO`
        will be considered zeros.
      transform_params: A `Sequence` of namedtuples
        `HyperVolumeScalarizer.PARAMS`, each containing a slope and an offset
        for transforming an objective to be non-negative.

    Raises:
      TypeError: if `not isinstance(direction, Sequence)`.
      ValueError: if `any([x < 0 for x in direction])`.
      ValueError: if the 2-norm of `direction` is less than
        `HyperVolumeScalarizer.ALMOST_ZERO`.
      TypeError: if `not isinstance(transform_params, Sequence)`.
      ValueError: if `len(transform_params) != len(self._direction)`.
    """
    if not isinstance(direction, Sequence):
      raise TypeError(
          'direction should be a Sequence, but is {}.'.format(direction))
    if any([x < 0 for x in direction]):
      raise ValueError(
          'direction should be in the positive orthant, but has negative '
          'coordinates: [{}].'.format(', '.join(map(str, direction))))
    length = np.sqrt(sum([x * x for x in direction]))
    if length < self.ALMOST_ZERO:
      raise ValueError(
          'direction found to be a nearly-zero vector, but should not be.')
    self._direction = [x / length for x in direction]
    if not isinstance(transform_params, Sequence):
      raise TypeError(
          'transform_params should be a Sequence, but is {}.'.format(
              transform_params))
    if len(transform_params) != len(self._direction):
      raise ValueError(
          'direction has {} elements but transform_params has {}.'.format(
              len(self._direction), len(transform_params)))
    self._slopes, self._offsets = zip(*[(p.slope, p.offset)
                                        for p in transform_params])
    super(HyperVolumeScalarizer, self).__init__(len(self._direction))

  def call(self, multi_objectives: tf.Tensor) -> tf.Tensor:
    transformed_objectives = tf.maximum(
        multi_objectives * self._slopes + self._offsets, 0)
    nonzero_mask = tf.broadcast_to(
        tf.cast(tf.abs(self._direction) >= self.ALMOST_ZERO, dtype=tf.bool),
        multi_objectives.shape)
    return tf.reduce_min(
        tf.where(nonzero_mask, transformed_objectives / self._direction,
                 multi_objectives.dtype.max),
        axis=1)
