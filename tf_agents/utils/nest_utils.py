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

"""Utilities for handling nested tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numbers

from typing import Optional, Text

from absl import logging
import numpy as np
from six.moves import zip
import tensorflow as tf
from tf_agents.typing import types
from tf_agents.utils import composite
import wrapt


# TODO(b/128613858): Update to a public facing API.
from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


try:
  # Python 3.3 and above.
  collections_abc = collections.abc
except AttributeError:
  collections_abc = collections


flatten_up_to = nest.flatten_up_to
flatten_with_tuple_paths = nest.flatten_with_tuple_paths
map_structure_up_to = nest.map_structure_up_to
map_structure_with_paths = nest.map_structure_with_paths


class _Dot(object):
  """An object whose representation is a simple '.'."""

  def __repr__(self):
    return '.'

  def __str__(self):
    return '.'


_DOT = _Dot()


def assert_same_structure(nest1,
                          nest2,
                          check_types: bool = True,
                          expand_composites: bool = False,
                          allow_shallow_nest1: bool = False,
                          message: Optional[Text] = None) -> None:
  """Same as tf.nest.assert_same_structure but with cleaner error messages.

  Args:
    nest1: an arbitrarily nested structure.
    nest2: an arbitrarily nested structure.
    check_types: if `True` (default) types of sequences are checked as well,
      including the keys of dictionaries. If set to `False`, for example a list
      and a tuple of objects will look the same if they have the same size. Note
      that namedtuples with identical name and fields are always considered to
      have the same shallow structure. Two types will also be considered the
      same if they are both list subtypes (which allows "list" and
      "_ListWrapper" from trackable dependency tracking to compare equal).
    expand_composites: If true, then composite tensors such as `tf.SparseTensor`
      and `tf.RaggedTensor` are expanded into their component tensors.
    allow_shallow_nest1: If `True`, `nest1` is allowed to be more shallow
      than `nest2`.
    message: Optional error message to provide in case of failure.

  Raises:
    ValueError: If the two structures do not have the same number of elements
      or if the two structures are not nested in the same way.
    TypeError: If the two structures differ in the type of sequence in any
      of their substructures. Only possible if `check_types is True`.
  """
  if not isinstance(check_types, bool):
    raise TypeError(
        'check_types must be a bool but saw: \'{}\''.format(check_types))
  if not isinstance(expand_composites, bool):
    raise TypeError('expand_composites must be a bool but saw: \'{}\''.format(
        expand_composites))
  message = message or 'The two structures do not match'
  exception = None

  if allow_shallow_nest1:
    check_fn = nest.assert_shallow_structure
  else:
    check_fn = tf.nest.assert_same_structure

  try:
    check_fn(
        nest1,
        nest2,
        check_types=check_types,
        expand_composites=expand_composites)
  except (TypeError, ValueError) as e:
    exception = type(e)

  if exception:
    str1 = tf.nest.map_structure(
        lambda _: _DOT, nest1, expand_composites=expand_composites)
    str2 = tf.nest.map_structure(
        lambda _: _DOT, nest2, expand_composites=expand_composites)
    raise exception('{}:\n  {}\nvs.\n  {}\nValues:\n  {}\nvs.\n  {}.'
                    .format(message, str1, str2, nest1, nest2))


def flatten_with_joined_paths(structure, expand_composites=False):
  flattened = flatten_with_tuple_paths(
      structure, expand_composites=expand_composites)

  def stringify_and_join(path_elements):
    return '/'.join(str(path_element) for path_element in path_elements)

  return [(stringify_and_join(path), value) for (path, value) in flattened]


def fast_map_structure_flatten(func, structure, *flat_structure, **kwargs):
  expand_composites = kwargs.get('expand_composites', False)
  entries = zip(*flat_structure)
  return tf.nest.pack_sequence_as(
      structure, [func(*x) for x in entries],
      expand_composites=expand_composites)


def fast_map_structure(func, *structure, **kwargs):
  expand_composites = kwargs.get('expand_composites', False)
  flat_structure = [
      tf.nest.flatten(s, expand_composites=expand_composites) for s in structure
  ]
  entries = zip(*flat_structure)

  return tf.nest.pack_sequence_as(
      structure[0], [func(*x) for x in entries],
      expand_composites=expand_composites)


def has_tensors(*x):
  return np.any(
      [tf.is_tensor(t) for t in tf.nest.flatten(x, expand_composites=True)])


def _is_namedtuple(x):
  return (isinstance(x, tuple)
          and isinstance(getattr(x, '_fields', None), collections_abc.Sequence))


def _is_attrs(x):
  return getattr(type(x), '__attrs_attrs__', None) is not None


def _attr_items(x):
  attrs = getattr(type(x), '__attrs_attrs__')
  attr_names = [a.name for a in attrs]
  return [(attr_name, getattr(x, attr_name)) for attr_name in attr_names]


def prune_extra_keys(narrow, wide):
  """Recursively prunes keys from `wide` if they don't appear in `narrow`.

  Often used as preprocessing prior to calling `tf.nest.flatten`
  or `tf.nest.map_structure`.

  This function is more forgiving than the ones in `nest`; if two substructures'
  types or structures don't agree, we consider it invalid and `prune_extra_keys`
  will return the `wide` substructure as is.  Typically, additional checking is
  needed: you will also want to use
  `nest.assert_same_structure(narrow, prune_extra_keys(narrow, wide))`
  to ensure the result of pruning is still a correct structure.

  Examples:
  ```python
  wide = [{"a": "a", "b": "b"}]
  # Narrows 'wide'
  assert prune_extra_keys([{"a": 1}], wide) == [{"a": "a"}]
  # 'wide' lacks "c", is considered invalid.
  assert prune_extra_keys([{"c": 1}], wide) == wide
  # 'wide' contains a different type from 'narrow', is considered invalid
  assert prune_extra_keys("scalar", wide) == wide
  # 'wide' substructure for key "d" does not match the one in 'narrow' and
  # therefore is returned unmodified.
  assert (prune_extra_keys({"a": {"b": 1}, "d": None},
                           {"a": {"b": "b", "c": "c"}, "d": [1, 2]})
          == {"a": {"b": "b"}, "d": [1, 2]})
  # assert prune_extra_keys((), wide) == ()
  # assert prune_extra_keys({"a": ()}, wide) == {"a": ()}
  ```

  Args:
    narrow: A nested structure.
    wide: A nested structure that may contain dicts with more fields than
      `narrow`.

  Returns:
    A structure with the same nested substructures as `wide`, but with
    dicts whose entries are limited to the keys found in the associated
    substructures of `narrow`.

    In case of substructure or size mismatches, the returned substructures
    will be returned as is.  Note that ObjectProxy-wrapped objects are
    considered equivalent to their non-ObjectProxy types.
  """
  #  If `narrow` is `()`, then `()` is returned.  That is, we narrow any
  #  object w.r.t. an empty tuple to to an empty tuple.  We use `id()`
  #  here because the emtpy tuple is a singleton in cpython and
  #  because using "x is ()" or "x == ()" gives syntax warnings for
  #  numpy arrays.
  narrow_raw = (narrow.__wrapped__ if isinstance(narrow, wrapt.ObjectProxy)
                else narrow)

  if id(narrow_raw) == id(()):
    return narrow

  if isinstance(wide, wrapt.ObjectProxy):
    return type(wide)(prune_extra_keys(narrow, wide.__wrapped__))

  wide_raw = (wide.__wrapped__ if isinstance(wide, wrapt.ObjectProxy) else wide)

  if ((type(narrow_raw) != type(wide_raw))  # pylint: disable=unidiomatic-typecheck
      and not (isinstance(narrow_raw, list) and isinstance(wide_raw, list))
      and not (isinstance(narrow_raw, collections_abc.Mapping)
               and isinstance(wide_raw, collections_abc.Mapping))):
    # We return early if the types are different; but we make some exceptions:
    #  list subtypes are considered the same (e.g. ListWrapper and list())
    #  Mapping subtypes are considered the same (e.g. DictWrapper and dict())
    #  (TupleWrapper subtypes are handled by unwrapping ObjectProxy above).
    return wide

  if isinstance(narrow, collections_abc.Mapping):
    if len(narrow) > len(wide):
      # wide lacks a required key from narrow; return early.
      return wide

    narrow_keys = set(narrow.keys())
    wide_keys = set(wide.keys())
    if not wide_keys.issuperset(narrow_keys):
      # wide lacks a required key from narrow; return early.
      return wide
    ordered_items = [
        (k, prune_extra_keys(v, wide[k]))
        for k, v in narrow.items()]
    if isinstance(wide, collections.defaultdict):
      subset = type(wide)(wide.default_factory, ordered_items)
    else:
      subset = type(wide)(ordered_items)
    return subset

  if nest.is_sequence(narrow):
    if _is_attrs(wide):
      items = [prune_extra_keys(n, w)
               for n, w in zip(_attr_items(narrow), _attr_items(wide))]
      return type(wide)(*items)

    # Not an attrs, so can treat as lists or tuples from here on.
    if len(narrow) != len(wide):
      # wide's size is different than narrow; return early.
      return wide

    items = [prune_extra_keys(n, w) for n, w in zip(narrow, wide)]
    if _is_namedtuple(wide):
      return type(wide)(*items)
    elif _is_attrs(wide):
      return type(wide)
    return type(wide)(items)

  # narrow is a leaf, just return wide
  return wide


def assert_tensors_matching_dtypes_and_shapes(tensors_1, tensors_2, caller,
                                              tensors_1_name, tensors_2_name):
  """Checks if tensors have matching dtypes and shapes.

  Args:
    tensors_1: A nest of tensor objects.
    tensors_2: A nest of tensor objects.
    caller: The object calling `assert...`.
    tensors_1_name: (str) Name to use for tensors_1 in case of an error.
    tensors_2_name: (str) Name to use for tensors_2 in case of an error.

  Raises:
    ValueError: If the tensors do not match dtypes or shapes.
  """
  assert_same_structure(
      tensors_1,
      tensors_2,
      message=('{}: {} and {} do not have matching structures'.format(
          caller, tensors_1_name, tensors_2_name)))

  def convert_to_tensor(t):
    return tf.convert_to_tensor(t) if not tf.is_tensor(t) else t

  flat_t1 = tf.nest.map_structure(convert_to_tensor, tf.nest.flatten(tensors_1))
  flat_t2 = tf.nest.map_structure(convert_to_tensor, tf.nest.flatten(tensors_2))

  t1_shapes = [t.shape for t in flat_t1]
  t1_dtypes = [t.dtype for t in flat_t1]
  t2_shapes = [t.shape for t in flat_t2]
  t2_dtypes = [t.dtype for t in flat_t2]

  compatible = True

  if any(
      t1_dtype != t2_dtype for t1_dtype, t2_dtype in zip(t1_dtypes, t2_dtypes)):
    compatible = False
  else:
    for t1_shape, t2_shape in zip(t1_shapes, t2_shapes):
      if t1_shape.ndims != t2_shape.ndims:
        compatible = False
        break

  if not compatible:
    get_dtypes = lambda v: tf.nest.map_structure(lambda x: x.dtype, v)
    get_shapes = lambda v: tf.nest.map_structure(lambda x: x.shape, v)
    raise ValueError('{}: Inconsistent dtypes or shapes between {} and {}.\n'
                     'dtypes:\n{}\nvs.\n{}.\n'
                     'shapes:\n{}\nvs.\n{}.'.format(caller, tensors_1_name,
                                                    tensors_2_name,
                                                    get_dtypes(tensors_1),
                                                    get_dtypes(tensors_2),
                                                    get_shapes(tensors_1),
                                                    get_shapes(tensors_2)))


def assert_matching_dtypes_and_inner_shapes(tensors_or_specs,
                                            specs,
                                            caller,
                                            tensors_name,
                                            specs_name,
                                            allow_extra_fields=False):
  """Returns `True` if tensors and specs have matching dtypes and inner shapes.

  Args:
    tensors_or_specs: A nest of `Tensor` like or `tf.TypeSpec` objects.
    specs: A nest of `tf.TypeSpec` objects.
    caller: The object calling `assert...`.
    tensors_name: (str) Name to use for the tensors in case of an error.
    specs_name: (str) Name to use for the specs in case of an error.
    allow_extra_fields: If `True`, then `tensors` may contain more keys or list
      fields than strictly required by `specs`.

  Raises:
    ValueError: If the tensors do not match the specs' dtypes or their inner
      shapes do not match the specs' shapes.
  """
  if allow_extra_fields:
    tensors_or_specs = prune_extra_keys(specs, tensors_or_specs)
  assert_same_structure(
      tensors_or_specs,
      specs,
      message=('{}: {} and {} do not have matching structures'.format(
          caller, tensors_name, specs_name)))

  flat_tensors = nest.flatten(tensors_or_specs)
  flat_specs = tf.nest.flatten(specs)
  def _convert(t, s):
    if not isinstance(t, tf.TypeSpec) and not tf.is_tensor(t):
      t = tf.convert_to_tensor(t, dtype_hint=s.dtype)
    return t

  flat_tensors = [_convert(t, s) for (t, s) in zip(flat_tensors, flat_specs)]

  tensor_shapes = [t.shape for t in flat_tensors]
  tensor_dtypes = [t.dtype for t in flat_tensors]
  spec_shapes = [spec_shape(s) for s in flat_specs]
  spec_dtypes = [t.dtype for t in flat_specs]

  compatible = True

  if any(s_dtype != t_dtype
         for s_dtype, t_dtype in zip(spec_dtypes, tensor_dtypes)):
    compatible = False
  else:
    for s_shape, t_shape in zip(spec_shapes, tensor_shapes):
      if s_shape.ndims in (0, None) or t_shape.ndims is None:
        continue
      if s_shape.ndims > t_shape.ndims:
        compatible = False
        break
      if not s_shape.is_compatible_with(t_shape[-s_shape.ndims:]):
        compatible = False
        break

  if not compatible:
    get_dtypes = lambda v: tf.nest.map_structure(lambda x: x.dtype, v)
    get_shapes = lambda v: tf.nest.map_structure(spec_shape, v)
    raise ValueError('{}: Inconsistent dtypes or shapes between {} and {}.\n'
                     'dtypes:\n{}\nvs.\n{}.\n'
                     'shapes:\n{}\nvs.\n{}.'.format(
                         caller,
                         tensors_name,
                         specs_name,
                         get_dtypes(tensors_or_specs),
                         get_dtypes(specs),
                         get_shapes(tensors_or_specs),
                         get_shapes(specs)))


def is_batched_nested_tensors(tensors,
                              specs,
                              num_outer_dims=1,
                              allow_extra_fields=False,
                              check_dtypes=True):
  """Compares tensors to specs to determine if all tensors are batched or not.

  For each tensor, it checks the dimensions and dtypes with respect to specs.

  Returns `True` if all tensors are batched and `False` if all tensors are
  unbatched.

  Raises a `ValueError` if the shapes are incompatible or a mix of batched and
  unbatched tensors are provided.

  Raises a `TypeError` if tensors' dtypes do not match specs.

  Args:
    tensors: Nested list/tuple/dict of Tensors.
    specs: Nested list/tuple/dict of Tensors or CompositeTensors describing the
      shape of unbatched tensors.
    num_outer_dims: The integer number of dimensions that are considered batch
      dimensions.  Default 1.
    allow_extra_fields: If `True`, then `tensors` may have extra subfields which
      are not in specs.  In this case, the extra subfields
      will not be checked.  For example:  ```python
      tensors = {"a": tf.zeros((3, 4), dtype=tf.float32),
                 "b": tf.zeros((5, 6), dtype=tf.float32)}
      specs = {"a": tf.TensorSpec(shape=(4,), dtype=tf.float32)} assert
        is_batched_nested_tensors(tensors, specs, allow_extra_fields=True) ```
        The above example would raise a ValueError if `allow_extra_fields` was
        False.
    check_dtypes: If `True` will validate that tensors and specs have the same
      dtypes.

  Returns:
    True if all Tensors are batched and False if all Tensors are unbatched.

  Raises:
    ValueError: If
      1. Any of the tensors or specs have shapes with ndims == None, or
      2. The shape of Tensors are not compatible with specs, or
      3. A mix of batched and unbatched tensors are provided.
      4. The tensors are batched but have an incorrect number of outer dims.
    TypeError: If `dtypes` between tensors and specs are not compatible.
  """
  if allow_extra_fields:
    tensors = prune_extra_keys(specs, tensors)

  assert_same_structure(
      tensors,
      specs,
      message='Tensors and specs do not have matching structures')
  flat_tensors = nest.flatten(tensors)
  flat_specs = tf.nest.flatten(specs)

  tensor_shapes = [t.shape for t in flat_tensors]
  tensor_dtypes = [t.dtype for t in flat_tensors]
  spec_shapes = [spec_shape(s) for s in flat_specs]
  spec_dtypes = [t.dtype for t in flat_specs]

  if any(s_shape.rank is None for s_shape in spec_shapes):
    raise ValueError('All specs should have ndims defined.  Saw shapes: %s' %
                     (tf.nest.pack_sequence_as(specs, spec_shapes),))

  if any(t_shape.rank is None for t_shape in tensor_shapes):
    raise ValueError('All tensors should have ndims defined.  Saw shapes: %s' %
                     (tf.nest.pack_sequence_as(specs, tensor_shapes),))

  if (check_dtypes and
      any(s_dtype != t_dtype
          for s_dtype, t_dtype in zip(spec_dtypes, tensor_dtypes))):
    raise TypeError('Tensor dtypes do not match spec dtypes:\n{}\nvs.\n{}'
                    .format(tf.nest.pack_sequence_as(specs, tensor_dtypes),
                            tf.nest.pack_sequence_as(specs, spec_dtypes)))
  is_unbatched = [
      s_shape.is_compatible_with(t_shape)
      for s_shape, t_shape in zip(spec_shapes, tensor_shapes)
  ]

  if all(is_unbatched):
    return False

  tensor_ndims_discrepancy = [
      t_shape.rank - s_shape.rank
      for s_shape, t_shape in zip(spec_shapes, tensor_shapes)
  ]

  tensor_matches_spec = [
      s_shape.is_compatible_with(t_shape[discrepancy:])
      for discrepancy, s_shape, t_shape in zip(
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
      (num_outer_dims, tf.nest.pack_sequence_as(specs, tensor_shapes),
       tf.nest.pack_sequence_as(specs, spec_shapes)))


def spec_shape(t):
  if isinstance(t, tf.SparseTensor):
    rank = tf.compat.dimension_value(t.dense_shape.shape[0])
    return tf.TensorShape([None] * rank)
  else:
    return t.shape


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
    return tf.nest.map_structure(lambda x: composite.expand_dims(x, 0), tensors)

  assert_same_structure(
      tensors,
      specs,
      message='Tensors and specs do not have matching structures')

  flat_tensors = tf.nest.flatten(tensors)
  flat_shapes = [spec_shape(s) for s in tf.nest.flatten(specs)]
  batched_tensors = []

  tensor_rank = lambda tensor: tensor.shape.rank
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor_rank(tensor) == shape.rank:
      tensor.shape.assert_is_compatible_with(shape)
      tensor = composite.expand_dims(tensor, 0)
    elif tensor_rank(tensor) == shape.rank + 1:
      tensor.shape[1:].assert_is_compatible_with(shape)
    else:
      raise ValueError('Tensor does not have the correct dimensions. '
                       'tensor.shape {} expected shape {}'.format(
                           tensor.shape, shape))
    batched_tensors.append(tensor)
  return tf.nest.pack_sequence_as(tensors, batched_tensors)


def _flatten_and_check_shape_nested_tensors(tensors, specs, num_outer_dims=1):
  """Flatten nested tensors and check their shape for use in other functions."""
  assert_same_structure(
      tensors,
      specs,
      message='Tensors and specs do not have matching structures')
  flat_tensors = tf.nest.flatten(tensors)
  flat_shapes = [spec_shape(s) for s in tf.nest.flatten(specs)]
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.rank == shape.rank:
      tensor.shape.assert_is_compatible_with(shape)
    elif tensor.shape.rank == shape.rank + num_outer_dims:
      tensor.shape[num_outer_dims:].assert_is_compatible_with(shape)
    else:
      raise ValueError('Tensor does not have the correct dimensions. '
                       'tensor.shape {} expected shape {}'.format(
                           tensor.shape, [None] + shape.as_list()))
  return flat_tensors, flat_shapes


def flatten_and_check_shape_nested_specs(specs, reference_specs):
  """Flatten nested specs and check their shape for use in other functions."""
  try:
    flat_specs, flat_shapes = _flatten_and_check_shape_nested_tensors(
        specs, reference_specs, num_outer_dims=0)
  except ValueError:
    raise ValueError('specs must be compatible with reference_specs'
                     '; instead got specs=%s, reference_specs=%s' %
                     (specs, reference_specs))
  return flat_specs, flat_shapes


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
    return tf.nest.map_structure(lambda x: composite.squeeze(x, 0), tensors)

  unbatched_tensors = []
  flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(
      tensors, specs)
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.rank == shape.rank + 1:
      tensor = composite.squeeze(tensor, 0)
    unbatched_tensors.append(tensor)
  return tf.nest.pack_sequence_as(tensors, unbatched_tensors)


def split_nested_tensors(tensors, specs, num_or_size_splits):
  """Split batched nested tensors, on batch dim (outer dim), into a list.

  Args:
    tensors: Nested list/tuple or dict of batched Tensors.
    specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
      non-batched Tensors.
    num_or_size_splits: Same as argument for tf.split. Either a python integer
      indicating the number of splits along batch_dim or a list of integer
      Tensors containing the sizes of each output tensor along batch_dim. If a
      scalar then it must evenly divide value.shape[axis]; otherwise the sum of
      sizes along the split dimension must match that of the value. For
      `SparseTensor` inputs, `num_or_size_splits` must be the scalar `num_split`
      (see documentation of `tf.sparse.split` for more details).

  Returns:
    A list of nested non-batched version of each tensor, where each list item
      corresponds to one batch item.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
    ValueError: if a non-scalar is passed and there are SparseTensors in the
      structure.
  """
  split_tensor_lists = []
  flat_tensors, flat_shapes = _flatten_and_check_shape_nested_tensors(
      tensors, specs)
  for tensor, shape in zip(flat_tensors, flat_shapes):
    if tensor.shape.rank == shape.rank:
      raise ValueError('Can only split tensors with a batch dimension.')
    if tensor.shape.rank == shape.rank + 1:
      if isinstance(tensor, tf.SparseTensor):
        if not isinstance(num_or_size_splits, numbers.Number):
          raise ValueError(
              'Saw a SparseTensor, for which num_or_size_splits must be a '
              'scalar.  But it is not: {}'.format(num_or_size_splits))
        split_tensors = tf.sparse.split(
            sp_input=tensor, num_split=num_or_size_splits, axis=0)
      else:
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
    if tensor.shape.rank == shape.rank:
      raise ValueError('Can only unstack tensors with a batch dimension.')
    if tensor.shape.rank == shape.rank + 1:
      unstacked_tensors = tf.unstack(tensor)
    unstacked_tensor_lists.append(unstacked_tensors)
  unstacked_tensors_zipped = zip(*unstacked_tensor_lists)
  return [
      tf.nest.pack_sequence_as(tensors, zipped)
      for zipped in unstacked_tensors_zipped
  ]


def stack_nested_tensors(tensors, axis=0):
  """Stacks a list of nested tensors along the dimension specified.

  Args:
    tensors: A list of nested tensors to be stacked.
    axis: the axis along which the stack operation is applied.

  Returns:
    A stacked nested tensor.
  """
  return tf.nest.map_structure(lambda *tensors: tf.stack(tensors, axis=axis),
                               *tensors)


def flatten_multi_batched_nested_tensors(tensors, specs):
  """Reshape tensors to contain only one batch dimension.

  For each tensor, it checks the number of extra dimensions beyond those in
  the spec, and reshapes tensor to have only one batch dimension.
  NOTE: Each tensor's batch dimensions must be the same.

  Args:
    tensors: Nested list/tuple or dict of batched Tensors or SparseTensors.
    specs: Nested list/tuple or dict of TensorSpecs, describing the shape of the
      non-batched Tensors.

  Returns:
    A nested version of each tensor with a single batch dimension.
    A list of the batch dimensions which were flattened.
  Raises:
    ValueError: if the tensors and specs have incompatible dimensions or shapes.
  """
  assert_same_structure(
      tensors,
      specs,
      message='Tensors and specs do not have matching structures')
  flat_tensors = tf.nest.flatten(tensors)
  flat_spec_shapes = [spec_shape(s) for s in tf.nest.flatten(specs)]
  out_tensors = []
  batch_dims = []
  for i, (tensor, sp_shape) in enumerate(zip(flat_tensors, flat_spec_shapes)):
    if i == 0:  # Set batch_dims based on first tensor.
      batch_dims = tensor.shape[:tensor.shape.rank - sp_shape.rank]
      if batch_dims.is_fully_defined():
        batch_dims = batch_dims.as_list()
        batch_prod = np.prod(batch_dims)
        batch_dims = tf.constant(batch_dims, dtype=tf.int64)
      else:
        batch_dims = tf.shape(tensor)[:tensor.shape.rank - sp_shape.rank]
        batch_prod = tf.reduce_prod(batch_dims)
    if not sp_shape.is_fully_defined():
      # When shape of spec is not fully defined, we do not rely on it to
      # reshape the tensor but retain the original non-batch dims of tensors.
      non_batch_dims = tf.shape(tensor)[tensor.shape.rank - sp_shape.rank:]
      reshaped_dims = tf.concat([[batch_prod], non_batch_dims], 0)
    else:
      reshaped_dims = [batch_prod] + sp_shape.as_list()
    out_tensors.append(composite.reshape(tensor, reshaped_dims))
  return tf.nest.pack_sequence_as(tensors, out_tensors), batch_dims


def get_outer_shape(nested_tensor, spec):
  """Runtime batch dims of tensor's batch dimension `dim`.

  Args:
    nested_tensor: Nest of tensors.
    spec: The nested spec.

  Returns:
    A `Tensor` containing the outer shape.

  Raises:
    ValueError: If `nested_tensor` and `spec` have different structures.
    TypeError: If `nested_tensor` and `spec` structures have differing types.
  """
  assert_same_structure(
      nested_tensor,
      spec,
      message='Tensors and specs do not have matching structures')
  first_tensor = tf.nest.flatten(nested_tensor)[0]
  first_spec = tf.nest.flatten(spec)[0]

  # Check tensors have same batch shape.
  num_outer_dims = (len(first_tensor.shape) - len(first_spec.shape))
  if not is_batched_nested_tensors(
      nested_tensor, spec, num_outer_dims=num_outer_dims, check_dtypes=False):
    return tf.constant([], dtype=tf.int32)

  return tf.shape(input=first_tensor)[:num_outer_dims]


def get_outer_rank(tensors, specs):
  """Compares tensors to specs to determine the number of batch dimensions.

  For each tensor, it checks the dimensions with respect to specs and
  returns the number of batch dimensions if all nested tensors and
  specs agree with each other.

  Args:
    tensors: Nested list/tuple/dict of Tensors or SparseTensors.
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
  assert_same_structure(
      tensors,
      specs,
      message='Tensors and specs do not have matching structures')
  tensor_shapes = [t.shape for t in tf.nest.flatten(tensors)]
  spec_shapes = [spec_shape(s) for s in tf.nest.flatten(specs)]

  if any(s_shape.rank is None for s_shape in spec_shapes):
    raise ValueError('All specs should have ndims defined.  Saw shapes: %s' %
                     spec_shapes)

  if any(t_shape.rank is None for t_shape in tensor_shapes):
    raise ValueError('All tensors should have ndims defined.  Saw shapes: %s' %
                     tensor_shapes)

  is_unbatched = [
      s_shape.is_compatible_with(t_shape)
      for s_shape, t_shape in zip(spec_shapes, tensor_shapes)
  ]
  if all(is_unbatched):
    return 0

  tensor_ndims_discrepancy = [
      t_shape.rank - s_shape.rank
      for s_shape, t_shape in zip(spec_shapes, tensor_shapes)
  ]

  tensor_matches_spec = [
      s_shape.is_compatible_with(t_shape[discrepancy:])
      for discrepancy, s_shape, t_shape in zip(
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


def unbatch_nested_tensors_to_arrays(nested_tensors):

  def _to_unbatched_numpy(tensor):
    return np.squeeze(tensor.numpy(), 0)

  return tf.nest.map_structure(_to_unbatched_numpy, nested_tensors)


def _unstack_nested_arrays_into_flat_item_iterator(nested_array):

  def _unstack(array):
    # Use numpy views instead of np.split, it's 5x+ faster.
    return [array[i] for i in range(len(array))]

  return zip(*[_unstack(array) for array in tf.nest.flatten(nested_array)])


def unstack_nested_arrays(nested_array):
  """Unstack/unbatch a nest of numpy arrays.

  Args:
    nested_array: Nest of numpy arrays where each array has shape [batch_size,
      ...].

  Returns:
    A list of length batch_size where each item in the list is a nest
      having the same structure as `nested_array`.
  """

  return [
      tf.nest.pack_sequence_as(nested_array, zipped)
      for zipped in _unstack_nested_arrays_into_flat_item_iterator(nested_array)
  ]


def unstack_nested_arrays_into_flat_items(nested_array):
  """Unstack/unbatch a nest of numpy arrays into flat items.

  Rebuild the nested structure of the unbatched elements is expensive. On the
  other hand it is sometimes unnecessary (e.g. if the downstream processing
  requires flattened structure, e.g. some replay buffer writers which flattens
  the items anyway).

  Args:
    nested_array: Nest of numpy arrays where each array has shape [batch_size,
      ...].

  Returns:
    A list of length batch_size where each item in the list is the flattened
      version of the corresponding item of the input.
  """

  return list(_unstack_nested_arrays_into_flat_item_iterator(nested_array))


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
  """Generalization of tf.where for nested structures.

  This generalization handles applying where across nested structures and the
  special case where the rank of the condition is smaller than the rank of the
  true and false cases.

  Args:
    condition: A boolean Tensor of shape [B, ...]. The shape of condition must
      be equal to or a prefix of the shape of true_outputs and false_outputs. If
      condition's rank is smaller than the rank of true_outputs and
      false_outputs, dimensions of size 1 are added to condition to make its
      rank match that of true_outputs and false_outputs in order to satisfy the
      requirements of tf.where.
    true_outputs: Tensor or nested tuple of Tensors of any dtype, each with
      shape [B, ...], to be split based on `condition`.
    false_outputs: Tensor or nested tuple of Tensors of any dtype, each with
      shape [B, ...], to be split based on `condition`.

  Returns:
    Interleaved output from `true_outputs` and `false_outputs` based on
    `condition`.
  """
  assert_same_structure(
      true_outputs,
      false_outputs,
      message='"true_outputs" and "false_outputs" structures do not match')
  if tf.nest.flatten(true_outputs):
    case_rank = tf.rank(tf.nest.flatten(true_outputs)[0])
    rank_difference = case_rank - tf.rank(condition)
    condition_shape = tf.concat(
        [tf.shape(condition),
         tf.ones(rank_difference, dtype=tf.int32)], axis=0)
    condition = tf.reshape(condition, condition_shape)

  return tf.nest.map_structure(
      lambda t, f: tf.compat.v2.where(condition, t, f), true_outputs,
      false_outputs)


def remove_singleton_batch_spec_dim(spec: tf.TypeSpec,
                                    outer_ndim: int) -> tf.TypeSpec:
  """Look for `spec`'s shape, check that outer dim is 1, and remove it.

  If `spec.shape[i] != 1` for any `i in range(outer_ndim)`, we stop removing
  singleton batch dimensions at `i` and return what's left.  This is necessary
  to handle the outputs of inconsistent layers like `tf.keras.layers.LSTM()`
  which may take as input `(batch, time, dim) = (1, 1, Nin)` and emits only the
  batch entry if `time == 1`: output shape is `(1, Nout)`.  We log an error
  in these cases.

  Args:
    spec: A `tf.TypeSpec`.
    outer_ndim: The maximum number of outer singleton dims to remove.

  Returns:
    A `tf.TypeSpec`, the spec without its outer batch dimension(s).

  Raises:
    ValueError: If `spec` lacks a `shape` property.
  """
  shape = getattr(spec, 'shape', None)
  if shape is None:
    shape = getattr(spec, '_shape', None)
  if shape is None:
    raise ValueError(
        'Could not remove singleton batch dim from spec; it lacks a shape: {}'
        .format(spec))
  for i in range(outer_ndim):
    if len(shape) <= i:
      logging.error(
          'Could not remove singleton batch dim from spec; len(shape) < %d.  '
          'Shape: %s.  Skipping.', i + 1, shape)
      break
    if tf.compat.dimension_value(shape[i]) != 1:
      logging.error(
          'Could not remove singleton batch dim from spec; shape[%d] != 1: %s '
          '(shape: %s).  Skipping.', i, spec, shape)
      break
    spec = spec._unbatch()  # pylint: disable=protected-access
  return spec


def _tile_batch(t, multiplier, ensure_shape=True):
  """Core single-tensor implementation of tile_batch."""
  t = tf.convert_to_tensor(t, name='t')
  shape_t = tf.shape(t)
  if t.shape.ndims is None or t.shape.ndims < 1:
    raise ValueError('t must have statically known rank')
  tiling = [1] * (t.shape.ndims + 1)
  tiling[1] = multiplier
  num_batch_dims = tf.compat.dimension_value(t.shape.dims[0])
  tiled_static_batch_size = (
      num_batch_dims * multiplier if num_batch_dims is not None else None)
  tiled = tf.tile(tf.expand_dims(t, 1), tiling)
  tiled = tf.reshape(tiled,
                     tf.concat(([shape_t[0] * multiplier], shape_t[1:]), 0))

  if ensure_shape:
    tiled = tf.ensure_shape(
        tiled,
        tf.TensorShape([tiled_static_batch_size]).concatenate(t.shape[1:]))
  return tiled


def tile_batch(tensors: types.NestedTensor, multiplier: types.Int):
  """Tile the batch dimension of a (possibly nested structure of) tensor(s).

  Copied from tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py

  For each tensor t in a (possibly nested structure) of tensors,
  this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed of
  minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
  `[batch_size * multiplier, s0, s1, ...]` composed of minibatch entries
  `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
  `multiplier` times.

  Args:
    tensors: A nested structure of `Tensor` shaped `[batch_size, ...]`.
    multiplier: Python int or a Tensor. Note that if the multiplier is a tensor
      the shape can not be ensured.

  Returns:
    A (possibly nested structure of) `Tensor` shaped
    `[batch_size * multiplier, ...]`.

  Raises:
    ValueError: if tensor(s) `t` do not have a statically known rank or
    the rank is < 1.
  """
  ensure_shape = False if tf.is_tensor(multiplier) else True
  return tf.nest.map_structure(
      lambda t: _tile_batch(t, multiplier, ensure_shape=ensure_shape), tensors)


def assert_value_spec(
    output_spec: types.NestedTensorSpec,
    network_name: str):
  """Checks that `output_spec` is a nest of "value" type values.

  "value" type values correspond to floating point tensors with spec shape
  `()` or `(1,)`.

  Args:
    output_spec: The output spec returned by `network.create_variables`.
    network_name: The string name of the network for error messages.

  Raises:
    ValueError: If `output_spec` is not a nest of value-type tensors.
  """
  def check_value_spec(v):
    if not isinstance(v, tf.TensorSpec):
      raise ValueError(
          '{} emits outputs that are not tensors; spec: {}'
          .format(network_name, output_spec))
    if v.shape not in ((), (1,)):
      raise ValueError(
          '{} emits multiple values; spec: {}'
          .format(network_name, output_spec))
    if not v.dtype.is_floating:
      raise ValueError(
          '{} emits outputs that are not real numbers; spec: {}'
          .format(network_name, output_spec))

  tf.nest.map_structure(check_value_spec, output_spec)
