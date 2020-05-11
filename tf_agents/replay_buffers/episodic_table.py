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

"""A tensorflow table stored in tf.Variables.

The row is the index or location at which the value is saved, and the value is
a nest of Tensors.

This class is not threadsafe.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.utils import nest_utils

from tensorflow.python.ops import list_ops  # TF internal


def _empty_slot(spec):
  shape = [s if s is not None else -1 for s in spec.shape.as_list()]
  shape = tf.convert_to_tensor(value=shape, dtype=tf.int64, name='shape')
  return list_ops.empty_tensor_list(shape, spec.dtype)


class EpisodicTable(tf.Module):
  """A table that can store Episodes of variable length."""

  def __init__(self, tensor_spec, capacity, name_prefix='EpisodicTable'):
    """Creates a table.

    Args:
      tensor_spec: A nest of TensorSpec representing each value that can be
        stored in the table.
      capacity: Maximum number of values the table can store.
      name_prefix: optional name prefix for variable names.

    Raises:
      ValueError: If the names in tensor_spec are empty or not unique.
    """
    self._tensor_spec = tensor_spec
    self._capacity = capacity
    self._spec_names = []

    def _create_unique_slot_name(spec, count=0):
      name = spec.name or 'slot'
      name = name + '_' + str(count)
      if name not in self._spec_names:
        self._spec_names.append(name)
        return name_prefix + '.' + name
      else:
        return _create_unique_slot_name(spec, count + 1)

    self._slots = tf.nest.map_structure(_create_unique_slot_name,
                                        self._tensor_spec)
    self._flattened_slots = tf.nest.flatten(self._slots)
    self._flattened_specs = tf.nest.flatten(self._tensor_spec)

    def _create_storage(spec, slot_name):
      return tf.lookup.experimental.DenseHashTable(
          key_dtype=tf.int64,
          value_dtype=tf.variant,
          empty_key=-1,
          deleted_key=-2,
          name=slot_name,
          default_value=_empty_slot(spec))

    self._storage = tf.nest.map_structure(_create_storage, self._tensor_spec,
                                          self._slots)
    self._variables = tf.nest.flatten(self._storage)
    self._slot2variable_map = dict(zip(self._flattened_slots, self._variables))
    self._slot2spec_map = dict(
        zip(self._flattened_slots, self._flattened_specs))

  @property
  def slots(self):
    return self._slots

  def variables(self):
    return self._variables

  def _stack_tensor_list(self, slot, tensor_list):
    """Stacks a slot list, restoring dtype and shape information.

    Going through the Variable loses all TensorList dtype and
    static element shape info.  Setting dtype, shape, and adding batch
    dimension for the length of the list.

    Args:
      slot: The slot corresponding to tensor_list
      tensor_list: TensorList of values stored in a scalar Tensor.

    Returns:
      A Tensor with first dimension equal to the length of tensor_list.
    """
    tensor_list.shape.assert_has_rank(0)
    value = list_ops.tensor_list_stack(tensor_list,
                                       self._slot2spec_map[slot].dtype)
    value.set_shape([None] + self._slot2spec_map[slot].shape.as_list())
    return value

  def get_episode_lists(self, rows=None):
    """Returns episodes as TensorLists.

    Args:
      rows: A list/tensor of location(s) to retrieve. If not specified, all
        episodes are returned.

    Returns:
      Episodes as TensorLists, stored in nested Tensors.
    """
    if rows is None:
      rows = tf.range(self._capacity, dtype=tf.int64)
    else:
      rows = tf.convert_to_tensor(value=rows, dtype=tf.int64)
    values = [
        self._slot2variable_map[slot].lookup(rows)
        for slot in self._flattened_slots
    ]
    return tf.nest.pack_sequence_as(self.slots, values)

  def get_episode_values(self, row):
    """Returns all values for the given row.

    Args:
      row: A scalar tensor of location to read values from. A batch of values
        will be returned with each Tensor having an extra first dimension equal
        to the length of rows.

    Returns:
      Stacked values at given row.
    """
    row = tf.convert_to_tensor(value=row, dtype=tf.int64)
    row.shape.assert_has_rank(0)
    return tf.nest.map_structure(self._stack_tensor_list, self.slots,
                                 self.get_episode_lists(row))

  def append(self, row, values):
    """Returns ops for appending multiple time values at the given row.

    Args:
      row: A scalar location at which to append values.
      values: A nest of Tensors to append.  The outermost dimension of each
        tensor is treated as a time axis, and these must all be equal.

    Returns:
      Ops for appending values at the given row.
    """
    row = tf.convert_to_tensor(value=row, dtype=tf.int64)
    flattened_values = tf.nest.flatten(values)
    append_ops = []
    for spec, slot, value in zip(self._flattened_specs, self._flattened_slots,
                                 flattened_values):
      var_slot = self._slot2variable_map[slot].lookup(row)
      value_as_tl = list_ops.tensor_list_from_tensor(
          value, element_shape=tf.cast(spec.shape.as_list(), dtype=tf.int64))
      new_value = list_ops.tensor_list_concat_lists(
          var_slot, value_as_tl, element_dtype=spec.dtype)
      append_ops.append(
          self._slot2variable_map[slot].insert_or_assign(row, new_value))
    return tf.group(*append_ops)

  def add(self, rows, values):
    """Returns ops for appending a single frame value to the given rows.

    This operation is batch-aware.

    Args:
      rows: A list/tensor of location(s) to write values at.
      values: A nest of Tensors to write. If rows has more than one element,
        values can have an extra first dimension representing the batch size.
        Values must have the same structure as the tensor_spec of this class
        Must have batch dimension matching the number of rows.

    Returns:
      Ops for appending values at rows.
    """
    rows = tf.convert_to_tensor(value=rows, dtype=tf.int64)
    flattened_values = tf.nest.flatten(values)
    write_ops = []
    for slot, value in zip(self._flattened_slots, flattened_values):
      var_slots = self._slot2variable_map[slot].lookup(rows)
      new_value = list_ops.tensor_list_push_back_batch(var_slots, value)
      write_ops.append(
          self._slot2variable_map[slot].insert_or_assign(rows, new_value))
    return tf.group(*write_ops)

  def extend(self, rows, episode_lists):
    """Returns ops for extending a set of rows by the given TensorLists.

    Args:
      rows: A batch of row locations to extend.
      episode_lists: Nested batch of TensorLists, must have the same batch
        dimension as rows.

    Returns:
      Ops for extending the table.
    """
    nest_utils.assert_same_structure(self.slots, episode_lists)

    rows = tf.convert_to_tensor(value=rows, dtype=tf.int64)
    existing_lists = self.get_episode_lists(rows)
    flat_existing_lists = tf.nest.flatten(existing_lists)
    flat_episode_lists = tf.nest.flatten(episode_lists)

    write_ops = []
    for spec, slot, existing_list, episode_list in zip(
        self._flattened_specs, self._flattened_slots, flat_existing_lists,
        flat_episode_lists):
      extended_list = list_ops.tensor_list_concat_lists(
          existing_list, episode_list, element_dtype=spec.dtype)
      slot_variable = self._slot2variable_map[slot]
      write_ops.append(
          slot_variable.insert_or_assign(rows, extended_list))
    return tf.group(*write_ops)

  def clear(self):
    """Returns op for clearing the table and removing all the episodes.

    Returns:
      Op for clearing the table.
    """
    clear_ops = []
    for slot in self._flattened_slots:
      clear_ops.append(
          self._slot2variable_map[slot].erase(
              tf.range(self._capacity, dtype=tf.int64)))
    return tf.group(*clear_ops)

  def clear_rows(self, rows):
    """Returns ops for clearing all the values at the given rows.

    Args:
      rows: A list/tensor of location(s) to clear values.
    Returns:
      Ops for clearing the values at rows.
    """
    rows = tf.convert_to_tensor(value=rows, dtype=tf.int64)
    clear_ops = []
    for spec, slot in zip(self._flattened_specs, self._flattened_slots):
      new_value = tf.fill([tf.size(input=rows)], _empty_slot(spec))
      clear_ops.append(
          self._slot2variable_map[slot].insert_or_assign(rows, new_value))
    return tf.group(*clear_ops)
