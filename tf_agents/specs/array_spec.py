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

"""A class to describe the shape and dtype of numpy arrays."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def sample_bounded_spec(spec, rng):
  """Samples the given bounded spec.

  Args:
    spec: A BoundedSpec to sample.
    rng: A numpy RandomState to use for the sampling.

  Returns:
    An np.array sample of the requested space.
  """
  tf_dtype = tf.as_dtype(spec.dtype)
  low = spec.minimum
  high = spec.maximum

  if tf_dtype.is_floating:
    if spec.dtype == np.float64 and np.any(np.isinf(high - low)):
      # The min-max interval cannot be represented by the np.float64. This is a
      # problem only for np.float64, np.float32 works as expected.
      # Spec bounds are set to read only so we can't use argumented assignment.
      low = low / 2  # pylint: disable=g-no-augmented-assignment
      high = high / 2  # pylint: disable=g-no-augmented-assignment
    return rng.uniform(
        low,
        high,
        size=spec.shape,
    ).astype(spec.dtype)

  else:
    if spec.dtype == np.int64 and np.any(high - low < 0):
      # The min-max interval cannot be represented by the tf_dtype. This is a
      # problem only for int64.
      low = low / 2  # pylint: disable=g-no-augmented-assignment
      high = high / 2  # pylint: disable=g-no-augmented-assignment

    if np.any(high < tf_dtype.max):
      high = np.where(high < tf_dtype.max, high + 1, high)  # pylint: disable=g-no-augmented-assignment
    elif spec.dtype != np.int64 or spec.dtype != np.uint64:
      # We can still +1 the high if we cast it to the larger dtype.
      high = high.astype(np.int64) + 1

    if low.size == 1 and high.size == 1:
      return rng.randint(
          low,
          high,
          size=spec.shape,
          dtype=spec.dtype,
      )
    else:
      return np.reshape(
          np.array([
              rng.randint(low, high, size=1, dtype=spec.dtype)
              for low, high in zip(low.flatten(), high.flatten())
          ]), spec.shape)


def sample_spec_nest(structure, rng, outer_dims=()):
  """Samples the given nest of specs.

  Args:
    structure: An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
    rng: A numpy RandomState to use for the sampling.
    outer_dims: An optional list/tuple specifying outer dimensions to add to the
      spec shape before sampling.

  Returns:
    A nest of sampled values following the ArraySpec definition.
  """

  def sample_fn(spec):
    spec = BoundedArraySpec.from_spec(spec)
    spec = BoundedArraySpec(
        tuple(outer_dims) + tuple(spec.shape), spec.dtype, spec.minimum,
        spec.maximum, spec.name)
    return sample_bounded_spec(spec, rng)

  return tf.nest.map_structure(sample_fn, structure)


def check_arrays_nest(arrays, spec):
  """Check that the arrays conform to the spec.

  Args:
    arrays: A NumPy array, or a nested dict, list or tuple of arrays.
    spec: An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.

  Returns:
    True if the arrays conforms to the spec, False otherwise.
  """
  # Check that arrays and spec has the same structure.
  try:
    tf.nest.assert_same_structure(arrays, spec)
  except (TypeError, ValueError):
    return False

  def check_array(spec, array):
    if not isinstance(spec, ArraySpec):
      return False
    return spec.check_array(array)

  # Check all the elements in arrays match to their spec
  checks = tf.nest.map_structure(check_array, spec, arrays)
  # Only return True if all the checks pass.
  return all(tf.nest.flatten(checks))


def add_outer_dims_nest(structure, outer_dims):
  def add_outer_dims(spec):
    name = spec.name
    shape = outer_dims + spec.shape
    if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
      return BoundedArraySpec(shape, spec.dtype, spec.minimum,
                              spec.maximum, name)
    return ArraySpec(shape, spec.dtype, name=name)

  return tf.nest.map_structure(add_outer_dims, structure)


@gin.configurable
class ArraySpec(object):
  """Describes a numpy array or scalar shape and dtype.

  An `ArraySpec` allows an API to describe the arrays that it accepts or
  returns, before that array exists.
  The equivalent version describing a `tf.Tensor` is `TensorSpec`.
  """

  __hash__ = None
  __slots__ = ('_shape', '_dtype', '_name')

  def __init__(self, shape, dtype, name=None):
    """Initializes a new `ArraySpec`.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    self._shape = tuple(shape)
    self._dtype = np.dtype(dtype)
    self._name = name

  @property
  def shape(self):
    """Returns a `tuple` specifying the array shape."""
    return self._shape

  @property
  def dtype(self):
    """Returns a numpy dtype specifying the array dtype."""
    return self._dtype

  @property
  def name(self):
    """Returns the name of the ArraySpec."""
    return self._name

  def __repr__(self):
    return 'ArraySpec(shape={}, dtype={}, name={})'.format(
        self.shape, repr(self.dtype), repr(self.name))

  def __eq__(self, other):
    """Checks if the shape and dtype of two specs are equal."""
    if not isinstance(other, ArraySpec):
      return False
    return self.shape == other.shape and self.dtype == other.dtype

  def __ne__(self, other):
    return not self == other

  def check_array(self, array):
    """Return whether the given NumPy array conforms to the spec.

    Args:
      array: A NumPy array or a scalar. Tuples and lists will not be converted
        to a NumPy array automatically; they will cause this function to return
        false, even if a conversion to a conforming array is trivial.

    Returns:
      True if the array conforms to the spec, False otherwise.
    """
    if isinstance(array, np.ndarray):
      return self.shape == array.shape and self.dtype == array.dtype
    elif isinstance(array, numbers.Number):
      return self.shape == tuple() and self.dtype == np.dtype(type(array))
    else:
      return False

  def __reduce__(self):
    return (ArraySpec, (self.shape, self.dtype, self.name))

  @staticmethod
  def from_array(array, name=None):
    """Construct a spec from the given array or number."""
    if isinstance(array, np.ndarray):
      return ArraySpec(array.shape, array.dtype, name)
    elif isinstance(array, numbers.Number):
      return ArraySpec(tuple(), type(array), name)
    else:
      raise ValueError('Array must be a np.ndarray or number. Got %r.' % array)

  @staticmethod
  def from_spec(spec):
    """Construct a spec from the given spec."""
    return ArraySpec(spec.shape, spec.dtype, spec.name)

  def replace(self, shape=None, dtype=None, name=None):
    shape = self.shape if shape is None else shape
    dtype = self.dtype if dtype is None else dtype
    name = self.name if name is None else name
    return ArraySpec(shape, dtype, name)


@gin.configurable
class BoundedArraySpec(ArraySpec):
  """An `ArraySpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  # Specifying the same minimum and maximum for every element.
  spec = BoundedArraySpec((3, 4), np.float64, minimum=0.0, maximum=1.0)

  # Specifying a different minimum and maximum for each element.
  spec = BoundedArraySpec(
      (2,), np.float64, minimum=[0.1, 0.2], maximum=[0.9, 0.9])

  # Specifying the same minimum and a different maximum for each element.
  spec = BoundedArraySpec(
      (3,), np.float64, minimum=-10.0, maximum=[4.0, 5.0, 3.0])
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by arrays
  with values in the set {0, 1, 2}:
  ```python
  spec = BoundedArraySpec((3, 4), np.int, minimum=0, maximum=2)
  ```
  """

  __hash__ = None
  __slots__ = ('_minimum', '_maximum')

  def __init__(self, shape, dtype, minimum=None, maximum=None, name=None):
    """Initializes a new `BoundedArraySpec`.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      minimum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not broadcastable to `shape` or
        if the limits are outside of the range of the specified dtype.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    super(BoundedArraySpec, self).__init__(shape, dtype, name)

    try:
      np.broadcast_to(minimum, shape=shape)
    except ValueError as numpy_exception:
      raise ValueError('minimum is not compatible with shape. '
                       'Message: {!r}.'.format(numpy_exception))

    try:
      np.broadcast_to(maximum, shape=shape)
    except ValueError as numpy_exception:
      raise ValueError('maximum is not compatible with shape. '
                       'Message: {!r}.'.format(numpy_exception))

    tf_dtype = tf.as_dtype(self._dtype)
    low = tf_dtype.min
    high = tf_dtype.max

    if minimum is None:
      minimum = low
    if maximum is None:
      maximum = high

    self._minimum = np.array(minimum)
    self._maximum = np.array(maximum)

    if tf_dtype.is_floating:
      # Replacing infinities with extreme finite float values.
      self._minimum[self._minimum == -np.inf] = low
      self._minimum[self._minimum == np.inf] = high

      self._maximum[self._maximum == -np.inf] = low
      self._maximum[self._maximum == np.inf] = high

    if np.any(self._minimum > self._maximum):
      raise ValueError(
          'Spec bounds min has values greater than max: [{},{}]'.format(
              self._minimum, self._maximum))
    if (np.any(self._minimum < low) or np.any(self._minimum > high) or
        np.any(self._maximum < low) or np.any(self._maximum > high)):
      raise ValueError(
          'Spec bounds [{},{}] not within the range [{}, {}] of the given '
          'dtype ({})'.format(self._minimum, self._maximum, low, high,
                              self._dtype))

    self._minimum = self._minimum.astype(self._dtype)
    self._minimum.setflags(write=False)

    self._maximum = self._maximum.astype(self._dtype)
    self._maximum.setflags(write=False)

  @classmethod
  def from_spec(cls, spec, name=None):
    if name is None:
      name = spec.name

    if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
      return BoundedArraySpec(spec.shape, spec.dtype, spec.minimum,
                              spec.maximum, name)

    return BoundedArraySpec(spec.shape, spec.dtype, name=name)

  @property
  def minimum(self):
    """Returns a NumPy array specifying the minimum bounds (inclusive)."""
    return self._minimum

  @property
  def maximum(self):
    """Returns a NumPy array specifying the maximum bounds (inclusive)."""
    return self._maximum

  @property
  def num_values(self):
    """Returns the number of values for discrete BoundedArraySpec."""
    if is_discrete(self):
      return (np.broadcast_to(self.maximum, shape=self.shape) -
              np.broadcast_to(self.minimum, shape=self.shape) + 1)

  def __repr__(self):
    template = ('BoundedArraySpec(shape={}, dtype={}, name={}, '
                'minimum={}, maximum={})')
    return template.format(self.shape, repr(self.dtype), repr(self.name),
                           self._minimum, self._maximum)

  def __eq__(self, other):
    if not isinstance(other, BoundedArraySpec):
      return False
    return (super(BoundedArraySpec, self).__eq__(other) and
            (self.minimum == other.minimum).all() and
            (self.maximum == other.maximum).all())

  def check_array(self, array):
    """Return true if the given array conforms to the spec."""
    return (super(BoundedArraySpec, self).check_array(array) and
            np.all(array >= self.minimum) and np.all(array <= self.maximum))

  def replace(self, shape=None, dtype=None,
              minimum=None, maximum=None,
              name=None):
    shape = self.shape if shape is None else shape
    dtype = self.dtype if dtype is None else dtype
    minimum = self.minimum if minimum is None else minimum
    maximum = self.maximum if maximum is None else maximum
    name = self.name if name is None else name
    return BoundedArraySpec(shape, dtype, minimum, maximum, name)

  def __reduce__(self):
    return (BoundedArraySpec, (self.shape, self.dtype, self.minimum,
                               self.maximum, self.name))


def is_bounded(spec):
  return isinstance(spec, BoundedArraySpec)


def is_discrete(spec):
  return issubclass(np.dtype(spec).type, np.integer)


def is_continuous(spec):
  return issubclass(np.dtype(spec).type, np.floating)
