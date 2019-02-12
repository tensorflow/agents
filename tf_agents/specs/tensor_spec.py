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

"""Utilities related to TensorSpec class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.specs import array_spec
from tensorflow.python.framework import ops  # TF internal
from tensorflow.python.framework import tensor_spec as ts  # TF internal

tfd = tfp.distributions

TensorSpec = ts.TensorSpec
BoundedTensorSpec = ts.BoundedTensorSpec


def is_bounded(spec):
  return isinstance(
      spec, (array_spec.BoundedArraySpec, BoundedTensorSpec))


def is_discrete(spec):
  if isinstance(spec, TensorSpec):
    return spec.dtype.is_integer
  else:
    return array_spec.is_discrete(spec)


def is_continuous(spec):
  if isinstance(spec, TensorSpec):
    return spec.dtype.is_floating
  else:
    return array_spec.is_continuous(spec)


def from_spec(spec):
  """Maps the given spec into corresponding TensorSpecs keeping bounds."""

  def _convert_to_tensor_spec(s):
    # Need to check bounded first as non bounded specs are base class.
    if isinstance(s, (array_spec.BoundedArraySpec, BoundedTensorSpec)):
      return BoundedTensorSpec.from_spec(s)
    elif isinstance(s, (array_spec.ArraySpec, TensorSpec)):
      return TensorSpec.from_spec(s)
    else:
      raise ValueError(
          "No known conversion from type `%s` to a TensorSpec" % type(s))

  return tf.nest.map_structure(_convert_to_tensor_spec, spec)


def to_array_spec(tensor_spec):
  """Converts TensorSpec into ArraySpec."""
  if hasattr(tensor_spec, "minimum") and hasattr(tensor_spec, "maximum"):
    return array_spec.BoundedArraySpec(tensor_spec.shape.as_list(),
                                       tensor_spec.dtype.as_numpy_dtype,
                                       minimum=tensor_spec.minimum,
                                       maximum=tensor_spec.maximum,
                                       name=tensor_spec.name)
  else:
    return array_spec.ArraySpec(tensor_spec.shape.as_list(),
                                tensor_spec.dtype.as_numpy_dtype,
                                tensor_spec.name)


def to_nest_array_spec(nest_array_spec):
  """Converted a nest of TensorSpecs to a nest of matching ArraySpecs."""
  return tf.nest.map_structure(to_array_spec, nest_array_spec)


def to_placeholder(spec, outer_dims=()):
  """Creates a placeholder from TensorSpec.

  Args:
    spec: instance of TensorSpec
    outer_dims: optional leading dimensions of the placeholder.

  Returns:
    An instance of tf.placeholder.
  """
  ph_shape = list(outer_dims) + spec.shape.as_list()
  return tf.compat.v1.placeholder(spec.dtype, ph_shape, spec.name)


def to_placeholder_with_default(default, spec, outer_dims=()):
  """Creates a placeholder from TensorSpec.

  Args:
    default: A constant value of output type dtype.
    spec: Instance of TensorSpec
    outer_dims: Optional leading dimensions of the placeholder.

  Returns:
    An instance of tf.placeholder.
  """
  ph_shape = list(outer_dims) + spec.shape.as_list()
  return tf.compat.v1.placeholder_with_default(default, ph_shape, spec.name)


def to_nest_placeholder(nested_tensor_specs,
                        default=None,
                        name_scope="",
                        outer_dims=()):
  """Converts a nest of TensorSpecs to a nest of matching placeholders.

  Args:
    nested_tensor_specs: A nest of tensor specs.
    default: Optional constant value to set as a default for the placeholder.
    name_scope: String name for the scope to create the placeholders in.
    outer_dims: Optional leading dimensions for the placeholder.

  Returns:
    A nest of placeholders matching the given tensor spec.

  Raises:
    ValueError: If a default is provided outside of the allowed types, or if
      default is a np.array that does not match the spec shape.
  """
  if default is None:
    to_ph = lambda spec: to_placeholder(spec, outer_dims=outer_dims)
  else:
    if not isinstance(default, (int, float, np.ndarray)):
      raise ValueError("to_nest_placeholder default value must be an int, "
                       "float, or np.ndarray")
    def to_ph(spec):
      shape = list(outer_dims) + spec.shape.as_list()
      if isinstance(default, np.ndarray) and list(default.shape) != shape:
        raise ValueError("Shape mismatch between default value and spec. "
                         "Got {}, expected {}".format(default.shape, shape))
      const = tf.constant(default, shape=shape, dtype=spec.dtype)
      return to_placeholder_with_default(const, spec, outer_dims=outer_dims)

  with tf.name_scope(name_scope):
    return tf.nest.map_structure(to_ph, nested_tensor_specs)


def _random_uniform_int(shape, outer_dims, minval, maxval, dtype, seed=None):
  """Iterates over n-d tensor minval, maxval limits to sample uniformly."""
  # maxval in BoundedTensorSpec is bound inclusive.
  # tf.random_uniform is upper bound exclusive, +1 to fix the sampling
  # behavior.
  # However +1 could cause overflow, in such cases we use the original maxval.
  maxval = np.broadcast_to(maxval, minval.shape).astype(dtype.as_numpy_dtype)
  minval = np.broadcast_to(minval, maxval.shape).astype(dtype.as_numpy_dtype)

  sampling_maxval = maxval
  if dtype.is_integer:
    sampling_maxval = np.where(maxval < dtype.max, maxval + 1, maxval)

  if not np.all(shape[-len(minval.shape):] == minval.shape):
    raise ValueError("%s == shape[-%d:] != minval.shape == %s.  shape == %s."
                     % (shape[len(minval.shape):],
                        len(minval.shape),
                        minval.shape,
                        shape))

  # Example:
  #  minval = [1.0, 2.0]
  #  shape = [3, 2]
  #  outer_dims = [5]
  # Sampling becomes:
  #  sample [5, 3] for minval 1.0
  #  sample [5, 3] for minval 2.0
  #  stack on innermost axis to get [5, 3, 2]
  #  reshape to get [5, 3, 2]
  samples = []
  shape = ops.convert_to_tensor(shape, dtype=tf.int32)
  sample_shape = tf.concat((outer_dims, shape[:-len(minval.shape)]), axis=0)
  full_shape = tf.concat((outer_dims, shape), axis=0)
  for (single_min, single_max) in zip(minval.flat, sampling_maxval.flat):
    samples.append(
        tf.random.uniform(
            shape=sample_shape,
            minval=single_min,
            maxval=single_max,
            dtype=dtype,
            seed=seed))
  samples = tf.stack(samples, axis=-1)
  samples = tf.reshape(samples, full_shape)
  return samples


def sample_bounded_spec(spec, seed=None, outer_dims=None):
  """Samples uniformily the given bounded spec.

  Args:
    spec: A BoundedSpec to sample.
    seed: A seed used for sampling ops
    outer_dims: An optional `Tensor` specifying outer dimensions to add to
      the spec shape before sampling.
  Returns:
    An Tensor sample of the requested spec.
  """
  minval = spec.minimum
  maxval = spec.maximum
  dtype = tf.as_dtype(spec.dtype)

  # To sample uint8 we will use int32 and cast later. This is needed for two
  # reasons:
  #  - tf.random_uniform does not currently support uint8
  #  - if you want to sample [0, 255] range, there's no way to do this since
  #    tf.random_uniform has exclusive upper bound and 255 + 1 would overflow.
  is_uint8 = dtype == tf.uint8
  sampling_dtype = tf.int32 if is_uint8 else dtype

  if dtype in [tf.float64, tf.float32]:
    # Avoid under/over-flow as random_uniform can't sample over the full range
    # for these types.
    minval = np.maximum(dtype.min / 2, minval)
    maxval = np.minimum(dtype.max / 2, maxval)

  if outer_dims is None:
    outer_dims = tf.constant([], dtype=tf.int32)
  else:
    outer_dims = ops.convert_to_tensor(outer_dims, dtype=tf.int32)

  def _unique_vals(vals):
    if vals.size > 0:
      if vals.ndim > 0:
        return np.all(vals == vals[0])
    return True

  if (minval.ndim != 0 or maxval.ndim != 0) and not (
      _unique_vals(minval) and _unique_vals(maxval)):
    # tf.random_uniform can only handle minval/maxval 0-d tensors.
    res = _random_uniform_int(
        shape=spec.shape,
        outer_dims=outer_dims,
        minval=minval, maxval=maxval, dtype=sampling_dtype,
        seed=seed)
  else:
    minval = minval.item(0) if minval.ndim != 0 else minval
    maxval = maxval.item(0) if maxval.ndim != 0 else maxval
    # BoundedTensorSpec are bounds inclusive.
    # tf.random_uniform is upper bound exclusive, +1 to fix the sampling
    # behavior.
    # However +1 will cause overflow, in such cases we use the original maxval.
    if sampling_dtype.is_integer and maxval < sampling_dtype.max:
      maxval = maxval + 1

    shape = ops.convert_to_tensor(spec.shape, dtype=tf.int32)
    full_shape = tf.concat((outer_dims, shape), axis=0)
    res = tf.random.uniform(
        full_shape,
        minval=minval,
        maxval=maxval,
        dtype=sampling_dtype,
        seed=seed)

  if is_uint8:
    res = tf.cast(res, dtype=dtype)

  return res


def sample_spec_nest(structure, seed=None, outer_dims=()):
  """Samples the given nest of specs.

  Args:
    structure: An `TensorSpec`, or a nested dict, list or tuple of
        `TensorSpec`s.
    seed: A seed used for sampling ops
    outer_dims: An optional `Tensor` specifying outer dimensions to add to
      the spec shape before sampling.
  Returns:
    A nest of sampled values following the ArraySpec definition.
  """

  seed_stream = tfd.SeedStream(seed=seed, salt="sample_spec_nest")
  def sample_fn(spec):
    spec = BoundedTensorSpec.from_spec(spec)
    return sample_bounded_spec(spec, outer_dims=outer_dims, seed=seed_stream())

  return tf.nest.map_structure(sample_fn, structure)
