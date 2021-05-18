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

"""Utilities related to TensorSpec class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.specs import array_spec
from tf_agents.typing import types

from google.protobuf import text_format
# pylint:disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import struct_pb2  # TF internal
from tensorflow.python.framework import tensor_spec as ts  # TF internal
from tensorflow.python.saved_model import nested_structure_coder  # TF internal
# pylint:enable=g-direct-tensorflow-import

tfd = tfp.distributions

TensorSpec = tf.TensorSpec
BoundedTensorSpec = ts.BoundedTensorSpec


def is_bounded(spec):
  return isinstance(spec, (array_spec.BoundedArraySpec, BoundedTensorSpec))


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
    if isinstance(s, tf.TypeSpec):
      return s
    if isinstance(s, (array_spec.BoundedArraySpec, BoundedTensorSpec)):
      return BoundedTensorSpec.from_spec(s)
    elif isinstance(s, array_spec.ArraySpec):
      return TensorSpec.from_spec(s)
    else:
      raise ValueError(
          "No known conversion from type `%s` to a TensorSpec.  Saw:\n  %s"
          % (type(s), s))

  return tf.nest.map_structure(_convert_to_tensor_spec, spec)


def to_array_spec(
    tensor_spec: Union[types.NestedArraySpec, types.NestedTensorSpec]
) -> types.NestedArraySpec:
  """Converts TensorSpec into ArraySpec."""
  def _convert(s):
    if isinstance(s, array_spec.ArraySpec):
      return s

    if hasattr(s, "minimum") and hasattr(s, "maximum"):
      return array_spec.BoundedArraySpec(
          s.shape.as_list(),
          s.dtype.as_numpy_dtype,
          minimum=s.minimum,
          maximum=s.maximum,
          name=s.name)
    else:
      return array_spec.ArraySpec(s.shape.as_list(),
                                  s.dtype.as_numpy_dtype,
                                  s.name)

  return tf.nest.map_structure(_convert, tensor_spec)


def to_nest_array_spec(
    nest_array_spec: Union[types.NestedArraySpec, types.NestedTensorSpec]
) -> types.NestedArraySpec:
  """(Deprecated) Alias for `to_array_spec`."""
  return to_array_spec(nest_array_spec)


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
    raise ValueError(
        "%s == shape[-%d:] != minval.shape == %s.  shape == %s." %
        (shape[len(minval.shape):], len(minval.shape), minval.shape, shape))

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
  shape = tf.convert_to_tensor(shape, dtype=tf.int32)
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
    outer_dims: An optional `Tensor` specifying outer dimensions to add to the
      spec shape before sampling.

  Returns:
    A Tensor sample of the requested spec.
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
    minval = np.maximum(dtype.min / 8, minval)
    maxval = np.minimum(dtype.max / 8, maxval)

  if outer_dims is None:
    outer_dims = tf.constant([], dtype=tf.int32)
  else:
    outer_dims = tf.convert_to_tensor(outer_dims, dtype=tf.int32)

  def _unique_vals(vals):
    if vals.size > 0:
      if vals.ndim > 0:
        return np.all(vals == vals[0])
    return True

  if (minval.ndim != 0 or
      maxval.ndim != 0) and not (_unique_vals(minval) and _unique_vals(maxval)):
    # tf.random_uniform can only handle minval/maxval 0-d tensors.
    res = _random_uniform_int(
        shape=spec.shape,
        outer_dims=outer_dims,
        minval=minval,
        maxval=maxval,
        dtype=sampling_dtype,
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

    shape = tf.convert_to_tensor(spec.shape, dtype=tf.int32)
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
    structure: A nest of `TensorSpec`.
    seed: A seed used for sampling ops
    outer_dims: An optional `Tensor` specifying outer dimensions to add to the
      spec shape before sampling.

  Returns:
    A nest of sampled values following the ArraySpec definition.

  Raises:
    TypeError: If `spec` is an unknown type.
    NotImplementedError: If `outer_dims` is not statically known but nest
      contains a `SparseTensorSpec`.
  """
  seed_stream = tfp.util.SeedStream(seed=seed, salt="sample_spec_nest")

  def sample_fn(spec):
    """Return a composite tensor sample given `spec`.

    Args:
      spec: A TensorSpec, SparseTensorSpec, etc.

    Returns:
      A tensor or SparseTensor.

    Raises:
      NotImplementedError: If `outer_dims` is not statically known and a
        SparseTensor is requested.
    """
    if isinstance(spec, tf.SparseTensorSpec):
      outer_shape = tf.get_static_value(outer_dims)
      if outer_dims is not None and outer_shape is None:
        raise NotImplementedError(
            "outer_dims must be statically known, got: {}".format(outer_dims))
      shape = tf.TensorShape(outer_shape or []).concatenate(spec.shape)

      if shape.num_elements() == 0 or tf.compat.dimension_value(shape[0]) == 0:
        return tf.SparseTensor(
            indices=tf.zeros([0, shape.rank], dtype=tf.int64),
            values=tf.zeros([0], dtype=spec.dtype),
            dense_shape=shape)

      indices_spec = BoundedTensorSpec(
          dtype=tf.int64,
          shape=[7, shape.rank],
          minimum=[0] * shape.rank,
          maximum=[x - 1 for x in shape.as_list()])
      values_dtype = tf.int32 if spec.dtype == tf.string else spec.dtype
      values_spec = BoundedTensorSpec(
          dtype=values_dtype,
          shape=[7],
          minimum=0,
          maximum=shape.as_list()[-1] - 1)
      values_sample = sample_bounded_spec(values_spec, seed=seed_stream())
      if spec.dtype == tf.string:
        values_sample = tf.as_string(values_sample)
      return tf.sparse.reorder(
          tf.SparseTensor(
              indices=sample_bounded_spec(indices_spec, seed=seed_stream()),
              values=values_sample,
              dense_shape=shape))
    elif isinstance(spec, (TensorSpec, BoundedTensorSpec)):
      if spec.dtype == tf.string:
        sample_spec = BoundedTensorSpec(
            spec.shape, tf.int32, minimum=0, maximum=10)
        return tf.as_string(
            sample_bounded_spec(
                sample_spec, outer_dims=outer_dims, seed=seed_stream()))
      else:
        return sample_bounded_spec(
            BoundedTensorSpec.from_spec(spec),
            outer_dims=outer_dims,
            seed=seed_stream())
    else:
      raise TypeError("Spec type not supported: '{}'".format(spec))

  return tf.nest.map_structure(sample_fn, structure)


def zero_spec_nest(specs, outer_dims=None):
  """Create zero tensors for a given spec.

  Args:
    specs: A nest of `TensorSpec`.
    outer_dims: An optional list of constants or `Tensor` specifying outer
      dimensions to add to the spec shape before sampling.

  Returns:
    A nest of zero tensors matching `specs`, with the optional outer
    dimensions added.

  Raises:
    TypeError: If `specs` is an unknown type.
    NotImplementedError: If `specs` contains non-dense tensor specs.
  """

  def make_zero(spec):
    if not isinstance(spec, TensorSpec):
      raise NotImplementedError("Spec type not supported: '{}'".format(spec))
    if outer_dims is None:
      shape = spec.shape
    else:
      spec_shape = tf.convert_to_tensor(value=spec.shape, dtype=tf.int32)
      shape = tf.concat((outer_dims, spec_shape), axis=0)
    return tf.zeros(shape, spec.dtype)

  if specs:
    if outer_dims is None:
      outer_dims = tf.constant([], dtype=tf.int32)
    else:
      outer_dims = tf.convert_to_tensor(outer_dims, dtype=tf.int32)

  return tf.nest.map_structure(make_zero, specs)


def add_outer_dims_nest(specs, outer_dims):
  """Adds outer dimensions to the shape of input specs.

  Args:
    specs: Nested list/tuple/dict of TensorSpecs/ArraySpecs, describing the
      shape of tensors.
    outer_dims: a list or tuple, representing the outer shape to be added to the
      TensorSpecs in specs.

  Returns:
    Nested TensorSpecs with outer dimensions added to the shape of input specs.

  Raises:
    ValueError: if any outer_dims is neither a list nor tuple.
  """
  if not isinstance(outer_dims, (tuple, list)):
    raise ValueError("outer_dims must be a tuple or list of dimensions")

  def add_outer_dims(spec):
    # TODO(b/187478998) Use spec.name when tf.SparseTensorSpec supports it.
    name = getattr(spec, "name", None)
    shape = outer_dims + spec.shape
    if hasattr(spec, "minimum") and hasattr(spec, "maximum"):
      return BoundedTensorSpec(shape, spec.dtype, spec.minimum, spec.maximum,
                               name)
    elif isinstance(specs, tf.SparseTensorSpec):
      # TODO(b/187478998) Add name when tf.SparseTensorSpec supports it.
      return tf.SparseTensorSpec(shape, spec.dtype)
    return TensorSpec(shape, spec.dtype, name=name)

  return tf.nest.map_structure(add_outer_dims, specs)


def add_outer_dim(specs, dim=None):
  """Adds an outer dimension to the shape of input specs.

  Args:
    specs: Nested list/tuple/dict of TensorSpecs/ArraySpecs, describing the
      shape of tensors.
    dim: Int, representing the outer dimension to be added to the TensorSpecs in
      specs.

  Returns:
    Nested TensorSpecs with outer dimensions added to the shape of input specs.

  """
  return add_outer_dims_nest(specs, outer_dims=(dim,))


def with_dtype(specs, dtype):
  """Updates dtypes of all specs in the input spec.

  Args:
    specs: Nested list/tuple/dict of TensorSpecs/ArraySpecs, describing the
      shape of tensors.
    dtype: dtype to update the specs to.

  Returns:
    Nested TensorSpecs with the udpated dtype.
  """

  def update_dtype(spec):
    if hasattr(spec, "minimum") and hasattr(spec, "maximum"):
      return BoundedTensorSpec(spec.shape, dtype, spec.minimum, spec.maximum,
                               spec.name)
    return TensorSpec(spec.shape, dtype, name=spec.name)

  return tf.nest.map_structure(update_dtype, specs)


def remove_outer_dims_nest(specs, num_outer_dims):
  """Removes the specified number of outer dimensions from the input spec nest.

  Args:
    specs: Nested list/tuple/dict of TensorSpecs/ArraySpecs, describing the
      shape of tensors.
    num_outer_dims: (int) Number of outer dimensions to remove.

  Returns:
    Nested TensorSpecs with outer dimensions removed from the input specs.

  Raises:
    Value error if a spec in the nest has shape rank less than `num_outer_dims`.
  """

  def remove_outer_dims(spec):
    """Removes the num_outer_dims of a tensor spec."""
    # TODO(b/187478998) Use spec.name when tf.SparseTensorSpec supports it.
    name = getattr(spec, "name", None)
    if len(spec.shape) < num_outer_dims:
      raise ValueError("The shape of spec {} has rank lower than the specified "
                       "num_outer_dims {}".format(spec, num_outer_dims))
    shape = list(spec.shape)[num_outer_dims:]
    if hasattr(spec, "minimum") and hasattr(spec, "maximum"):
      if isinstance(spec.minimum,
                    (tuple, list)) and len(spec.minimum) == len(spec.shape):
        minimum = spec.minimum[num_outer_dims:]
      else:
        minimum = spec.minimum
      if isinstance(spec.maximum,
                    (tuple, list)) and len(spec.maximum) == len(spec.shape):
        maximum = spec.maximum[num_outer_dims:]
      else:
        maximum = spec.maximum

      return BoundedTensorSpec(shape, spec.dtype, minimum, maximum, name)
    elif isinstance(spec, tf.SparseTensorSpec):
      # TODO(b/187478998) Add name when tf.SparseTensorSpec supports it.
      return tf.SparseTensorSpec(shape, spec.dtype)
    return TensorSpec(shape, spec.dtype, name=name)

  return tf.nest.map_structure(remove_outer_dims, specs)


def to_proto(spec):
  """Encodes a nested spec into a struct_pb2.StructuredValue proto.

  Args:
    spec: Nested list/tuple or dict of TensorSpecs, describing the
      shape of the non-batched Tensors.
  Returns:
    A `struct_pb2.StructuredValue` proto.
  """
  # Make sure spec is a tensor_spec.
  spec = from_spec(spec)
  signature_encoder = nested_structure_coder.StructureCoder()
  return signature_encoder.encode_structure(spec)


def from_proto(spec_proto):
  """Decodes a struct_pb2.StructuredValue proto into a nested spec."""
  signature_encoder = nested_structure_coder.StructureCoder()
  return signature_encoder.decode_proto(spec_proto)


def from_packed_proto(spec_packed_proto):
  """Decodes a packed Any proto containing the structured value for the spec."""
  spec_proto = struct_pb2.StructuredValue()
  spec_packed_proto.Unpack(spec_proto)
  return from_proto(spec_proto)


def to_pbtxt_file(output_path, spec):
  """Saves a spec encoded as a struct_pb2.StructuredValue in a pbtxt file."""
  spec_proto = to_proto(spec)
  with tf.io.gfile.GFile(output_path, "wb") as f:
    f.write(text_format.MessageToString(spec_proto))


def from_pbtxt_file(spec_path):
  """Loads a spec encoded as a struct_pb2.StructuredValue from a pbtxt file."""
  spec_proto = struct_pb2.StructuredValue()
  with tf.io.gfile.GFile(spec_path, "rb") as f:
    text_format.MergeLines(f, spec_proto)
  return from_proto(spec_proto)
