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

"""Utilities for handling nested tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# TODO(b/128613858): Update to a public facing API.
from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal

flatten_with_tuple_paths = nest.flatten_with_tuple_paths


def flatten_with_joined_paths(structure):
  flattened = flatten_with_tuple_paths(structure)

  def stringify_and_join(path_elements):
    return '/'.join(str(path_element) for path_element in path_elements)

  return [(stringify_and_join(path), value) for (path, value) in flattened]


def fast_map_structure_flatten(func, structure, *flat_structure):
  entries = zip(*flat_structure)
  return tf.nest.pack_sequence_as(structure, [func(*x) for x in entries])


def fast_map_structure(func, *structure):
  flat_structure = [tf.nest.flatten(s) for s in structure]
  entries = zip(*flat_structure)

  return tf.nest.pack_sequence_as(structure[0], [func(*x) for x in entries])


def has_tensors(*x):
  return np.any([tf.is_tensor(t) for t in tf.nest.flatten(x)])


def is_batched_nested_tensors(tensors, specs, num_outer_dims=1):
  """Compares tensors to specs to determine if all tensors are batched or not.

  For each tensor, it checks the dimensions with respect to specs and returns
  True if all tensors are batched and False if all tensors are unbatched, and
  raises a ValueError if the shapes are incompatible or a mix of batched and
  unbatched tensors are provided.

  Args:
    tensors: Nested list/tuple/dict of Tensors.
    specs: Nested list/tuple/dict of Tensors describing the shape of unbatched
      tensors.
    num_outer_dims: The integer number of dimensions that are considered batch
      dimensions.  Default 1.

  Returns:
    True if all Tensors are batched and False if all Tensors are unbatched.
  Raises:
    ValueError: If
      1. Any of the tensors or specs have shapes with ndims == None, or
      2. The shape of Tensors are not compatible with specs, or
      3. A mix of batched and unbatched tensors are provided.
      4. The tensors are batched but have an incorrect number of outer dims.
  """
  tf.nest.assert_same_structure(tensors, specs)
  tensor_shapes = [t.shape for t in tf.nest.flatten(tensors)]
  spec_shapes = [s.shape for s in tf.nest.flatten(specs)]

  if any(spec_shape.ndims is None for spec_shape in spec_shapes):
    raise ValueError('All specs should have ndims defined.  Saw shapes: %s' %
                     (tf.nest.pack_sequence_as(specs, spec_shapes),))

  if any(tensor_shape.ndims is None for tensor_shape in tensor_shapes):
    raise ValueError('All tensors should have ndims defined.  Saw shapes: %s' %
                     (tf.nest.pack_sequence_as(tensors, tensor_shapes),))

  is_unbatched = [
      spec_shape.is_compatible_with(tensor_shape)
      for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
  ]
  if all(is_unbatched):
    return False

  tensor_ndims_discrepancy = [
      tensor_shape.ndims - spec_shape.ndims
      for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
  ]

  tensor_matches_spec = [
      spec_shape.is_compatible_with(tensor_shape[discrepancy:])
      for discrepancy, spec_shape, tensor_shape in zip(
          tensor_ndims_discrepancy, spec_shapes, tensor_shapes)
  ]

  # Check if all tensors match and have correct number of outer_dims.
  is_batched = (
      all(discrepancy == num_outer_dims
          for discrepancy in tensor_ndims_discrepancy) and
      all(tensor_matches_spec))

  if is_batched:
    return True

  # Check if tensors match but have incorrect number of batch dimensions.
  if all(
      discrepancy == tensor_ndims_discrepancy[0]
      for discrepancy in tensor_ndims_discrepancy) and all(tensor_matches_spec):
    return False

  raise ValueError(
      'Received a mix of batched and unbatched Tensors, or Tensors'
      ' are not compatible with Specs.  num_outer_dims: %d.\n'
      'Saw tensor_shapes:\n   %s\n'
      'And spec_shapes:\n   %s' %
      (num_outer_dims, tf.nest.pack_sequence_as(tensors, tensor_shapes),
       tf.nest.pack_sequence_as(specs, spec_shapes)))


def batch_nested_tensors(tensors, specs=None):
  """Add batch dimension if needed to nested tensors while checking their specs.

  If specs is None, a batch dimension is added to each tensor.
  If specs are provided, each tensor is compared to the corresponding spec,
  and a batch dimension is added only if the tensor doesn't already have it.

  For each tensor, it checks the dimensions with respect to specs, and adds an
  extra batch dimension if it doesn't already have it.

  Args:
    tensors: Nested list/tuple or dict of Tensors.
    specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
      non-batched Tensors.

  Returns:
    A nested batched version of each tensor.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
  """
  if specs is None:
    return tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), tensors)

  tf.nest.assert_same_structure(tensors, specs)

  flat_tensors = tf.nest.flatten(tensors)
  flat_shapes = [s.shape for s in tf.nest.flatten(specs)]
  batched_tensors = []

  tensor_shape_fn = lambda tensor: tensor.shape.ndims
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor_shape_fn(tensor) == shape.ndims:
      tensor.shape.assert_is_compatible_with(shape)
      tensor = tf.expand_dims(tensor, tf.constant(0))
    elif tensor_shape_fn(tensor) == shape.ndims + 1:
      tensor.shape[1:].assert_is_compatible_with(shape)
    else:
      raise ValueError('Tensor does not have the correct dimensions. '
                       'tensor.shape {} expected shape {}'.format(
                           tensor.shape, shape))
    batched_tensors.append(tensor)
  return tf.nest.pack_sequence_as(tensors, batched_tensors)


def _flatten_and_check_shape_nested_tensors(tensors, specs, num_outer_dims=1):
  """Flatten nested tensors and check their shape for use in other functions."""
  tf.nest.assert_same_structure(tensors, specs)
  flat_tensors = tf.nest.flatten(tensors)
  flat_shapes = [s.shape for s in tf.nest.flatten(specs)]
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.ndims == shape.ndims:
      tensor.shape.assert_is_compatible_with(shape)
    elif tensor.shape.ndims == shape.ndims + num_outer_dims:
      tensor.shape[num_outer_dims:].assert_is_compatible_with(shape)
    else:
      raise ValueError('Tensor does not have the correct dimensions. '
                       'tensor.shape {} expected shape {}'.format(
                           tensor.shape, [None] + shape.as_list()))
  return flat_tensors, flat_shapes


def unbatch_nested_tensors(tensors, specs=None):
  """Remove the batch dimension if needed from nested tensors using their specs.

  If specs is None, the first dimension of each tensor will be removed.
  If specs are provided, each tensor is compared to the corresponding spec,
  and the first dimension is removed only if the tensor was batched.

  Args:
    tensors: Nested list/tuple or dict of batched Tensors.
    specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
      non-batched Tensors.

  Returns:
    A nested non-batched version of each tensor.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
  """
  if specs is None:
    return tf.nest.map_structure(lambda x: tf.squeeze(x, [0]), tensors)

  unbatched_tensors = []
  flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(
      tensors, specs)
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.ndims == shape.ndims + 1:
      tensor = tf.squeeze(tensor, [0])
    unbatched_tensors.append(tensor)
  return tf.nest.pack_sequence_as(tensors, unbatched_tensors)


def split_nested_tensors(tensors, specs, num_or_size_splits):
  """Split batched nested tensors, on batch dim (outer dim), into a list.

  Args:
    tensors: Nested list/tuple or dict of batched Tensors.
    specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
      non-batched Tensors.
    num_or_size_splits: Same as argument for tf.split. Either a 0-D integer
      Tensor indicating the number of splits along batch_dim or a 1-D integer
      Tensor containing the sizes of each output tensor along batch_dim. If a
      scalar then it must evenly divide value.shape[axis]; otherwise the sum of
      sizes along the split dimension must match that of the value.

  Returns:
    A list of nested non-batched version of each tensor, where each list item
      corresponds to one batch item.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
  """
  split_tensor_lists = []
  flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(
      tensors, specs)
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.ndims == shape.ndims:
      raise ValueError('Can only split tensors with a batch dimension.')
    if tensor.shape.ndims == shape.ndims + 1:
      split_tensors = tf.split(tensor, num_or_size_splits)
    split_tensor_lists.append(split_tensors)
  split_tensors_zipped = zip(*split_tensor_lists)
  return [
      tf.nest.pack_sequence_as(tensors, zipped)
      for zipped in split_tensors_zipped
  ]


def unstack_nested_tensors(tensors, specs):
  """Make list of unstacked nested tensors.

  Args:
    tensors: Nested tensors whose first dimension is to be unstacked.
    specs: Tensor specs for tensors.

  Returns:
    A list of the unstacked nested tensors.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
  """
  unstacked_tensor_lists = []
  flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(
      tensors, specs)
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.ndims == shape.ndims:
      raise ValueError('Can only unstack tensors with a batch dimension.')
    if tensor.shape.ndims == shape.ndims + 1:
      unstacked_tensors = tf.unstack(tensor)
    unstacked_tensor_lists.append(unstacked_tensors)
  unstacked_tensors_zipped = zip(*unstacked_tensor_lists)
  return [
      tf.nest.pack_sequence_as(tensors, zipped)
      for zipped in unstacked_tensors_zipped
  ]


def stack_nested_tensors(tensors):
  """Stacks a list of nested tensors along the first dimension.

  Args:
    tensors: A list of nested tensors to be stacked along the first dimension.

  Returns:
    A stacked nested tensor.
  """
  return tf.nest.map_structure(lambda *tensors: tf.stack(tensors), *tensors)


def flatten_multi_batched_nested_tensors(tensors, specs):
  """Reshape tensors to contain only one batch dimension.

  For each tensor, it checks the number of extra dimensions beyond those in
  the spec, and reshapes tensor to have only one batch dimension.
  NOTE: Each tensor's batch dimensions must be the same.

  Args:
    tensors: Nested list/tuple or dict of batched Tensors.
    specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
      non-batched Tensors.

  Returns:
    A nested version of each tensor with a single batch dimension.
    A list of the batch dimensions which were flattened.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
  """
  tf.nest.assert_same_structure(tensors, specs)
  flat_tensors = tf.nest.flatten(tensors)
  flat_shapes = [s.shape for s in tf.nest.flatten(specs)]
  out_tensors = []
  batch_dims = []
  for i, (tensor, shape) in enumerate(zip(flat_tensors, flat_shapes)):
    if i == 0:  # Set batch_dims based on first tensor.
      tensor_shape = tf.shape(input=tensor)
      batch_dims = tensor_shape[:tensor.shape.ndims - shape.ndims]
    reshaped_dims = [tf.reduce_prod(input_tensor=batch_dims)] + shape.as_list()
    out_tensors.append(tf.reshape(tensor, reshaped_dims))
  return tf.nest.pack_sequence_as(tensors, out_tensors), batch_dims


def get_outer_shape(nested_tensor, spec):
  """Runtime batch dims of tensor's batch dimension `dim`."""
  tf.nest.assert_same_structure(nested_tensor, spec)
  first_tensor = tf.nest.flatten(nested_tensor)[0]
  first_spec = tf.nest.flatten(spec)[0]

  # Check tensors have same batch shape.
  num_outer_dims = (len(first_tensor.shape) - len(first_spec.shape))
  if not is_batched_nested_tensors(
      nested_tensor, spec, num_outer_dims=num_outer_dims):
    return []

  return tf.shape(input=first_tensor)[:num_outer_dims]


def get_outer_rank(tensors, specs):
  """Compares tensors to specs to determine the number of batch dimensions.

  For each tensor, it checks the dimensions with respect to specs and
  returns the number of batch dimensions if all nested tensors and
  specs agree with each other.

  Args:
    tensors: Nested list/tuple/dict of Tensors.
    specs: Nested list/tuple/dict of TensorSpecs, describing the shape of
      unbatched tensors.

  Returns:
    The number of outer dimensions for all Tensors (zero if all are
      unbatched or empty).
  Raises:
    ValueError: If
      1. Any of the tensors or specs have shapes with ndims == None, or
      2. The shape of Tensors are not compatible with specs, or
      3. A mix of batched and unbatched tensors are provided.
      4. The tensors are batched but have an incorrect number of outer dims.
  """
  tf.nest.assert_same_structure(tensors, specs)
  tensor_shapes = [t.shape for t in tf.nest.flatten(tensors)]
  spec_shapes = [s.shape for s in tf.nest.flatten(specs)]

  if any(spec_shape.ndims is None for spec_shape in spec_shapes):
    raise ValueError('All specs should have ndims defined.  Saw shapes: %s' %
                     spec_shapes)

  if any(tensor_shape.ndims is None for tensor_shape in tensor_shapes):
    raise ValueError('All tensors should have ndims defined.  Saw shapes: %s' %
                     tensor_shapes)

  is_unbatched = [
      spec_shape.is_compatible_with(tensor_shape)
      for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
  ]
  if all(is_unbatched):
    return 0

  tensor_ndims_discrepancy = [
      tensor_shape.ndims - spec_shape.ndims
      for spec_shape, tensor_shape in zip(spec_shapes, tensor_shapes)
  ]

  tensor_matches_spec = [
      spec_shape.is_compatible_with(tensor_shape[discrepancy:])
      for discrepancy, spec_shape, tensor_shape in zip(
          tensor_ndims_discrepancy, spec_shapes, tensor_shapes)
  ]

  # At this point we are guaranteed to have at least one tensor/spec.
  num_outer_dims = tensor_ndims_discrepancy[0]

  # Check if all tensors match and have correct number of batch dimensions.
  is_batched = (
      all(discrepancy == num_outer_dims
          for discrepancy in tensor_ndims_discrepancy) and
      all(tensor_matches_spec))

  if is_batched:
    return num_outer_dims

  # Check if tensors match but have incorrect number of batch dimensions.
  incorrect_batch_dims = (
      tensor_ndims_discrepancy and
      all(discrepancy == tensor_ndims_discrepancy[0] and discrepancy >= 0
          for discrepancy in tensor_ndims_discrepancy) and
      all(tensor_matches_spec))

  if incorrect_batch_dims:
    raise ValueError('Received tensors with %d outer dimensions. '
                     'Expected %d.' %
                     (tensor_ndims_discrepancy[0], num_outer_dims))

  raise ValueError('Received a mix of batched and unbatched Tensors, or Tensors'
                   ' are not compatible with Specs.  num_outer_dims: %d.\n'
                   'Saw tensor_shapes:\n   %s\n'
                   'And spec_shapes:\n   %s' %
                   (num_outer_dims, tensor_shapes, spec_shapes))


def batch_nested_array(nested_array):
  return tf.nest.map_structure(lambda x: np.expand_dims(x, 0), nested_array)


def unbatch_nested_array(nested_array):
  return tf.nest.map_structure(lambda x: np.squeeze(x, 0), nested_array)


def unstack_nested_arrays(nested_array):
  """Unstack/unbatch a nest of numpy arrays.

  Args:
    nested_array: Nest of numpy arrays where each array has shape [batch_size,
      ...].

  Returns:
    A list of length batch_size where each item in the list is a nest
      having the same structure as `nested_array`.
  """

  def _unstack(array):
    if array.shape[0] == 1:
      arrays = [array]
    else:
      arrays = np.split(array, array.shape[0])
    return [np.reshape(a, a.shape[1:]) for a in arrays]

  unstacked_arrays_zipped = zip(
      *[_unstack(array) for array in tf.nest.flatten(nested_array)])
  return [
      tf.nest.pack_sequence_as(nested_array, zipped)
      for zipped in unstacked_arrays_zipped
  ]


def stack_nested_arrays(nested_arrays):
  """Stack/batch a list of nested numpy arrays.

  Args:
    nested_arrays: A list of nested numpy arrays of the same shape/structure.

  Returns:
    A nested array containing batched items, where each batched item is obtained
      by stacking corresponding items from the list of nested_arrays.
  """
  nested_arrays_flattened = [tf.nest.flatten(a) for a in nested_arrays]
  batched_nested_array_flattened = [
      np.stack(a) for a in zip(*nested_arrays_flattened)
  ]
  return tf.nest.pack_sequence_as(nested_arrays[0],
                                  batched_nested_array_flattened)


def get_outer_array_shape(nested_array, spec):
  """Batch dims of array's batch dimension `dim`."""
  first_array = tf.nest.flatten(nested_array)[0]
  first_spec = tf.nest.flatten(spec)[0]
  num_outer_dims = len(first_array.shape) - len(first_spec.shape)
  return first_array.shape[:num_outer_dims]


def where(condition, true_outputs, false_outputs):
  """Generalization of tf.compat.v1.where supporting nests as the outputs.


  Args:
    condition: A boolean Tensor of shape [B,].
    true_outputs: Tensor or nested tuple of Tensors of any dtype, each with
      shape [B, ...], to be split based on `condition`.
    false_outputs: Tensor or nested tuple of Tensors of any dtype, each with
      shape [B, ...], to be split based on `condition`.

  Returns:
    Interleaved output from `true_outputs` and `false_outputs` based on
    `condition`.
  """
  return tf.nest.map_structure(lambda t, f: tf.compat.v1.where(condition, t, f),
                               true_outputs, false_outputs)
