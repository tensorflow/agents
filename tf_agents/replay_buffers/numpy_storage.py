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

"""NumpyStorage stores nested objects across multiple numpy arrays."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.specs import array_spec


class NumpyStorage(tf.contrib.checkpoint.Checkpointable):
  """A class to store nested objects in a collection of numpy arrays.

  If a data_spec of `{'foo': ArraySpec(shape=(4,), dtype=np.uint8), 'bar':
  ArraySpec(shape=(3, 7), dtype=np.float32)}` were used, then this would create
  two arrays, one for the 'foo' key and one for the 'bar' key. The .get and
  .set methods would return/take Python dictionaries, but break down the
  component arrays before storing them.
  """

  def __init__(self, data_spec, capacity):
    """Creates a NumpyStorage object.

    Args:
      data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
        single item that can be stored in this table.
      capacity: The maximum number of items that can be stored in the buffer.

    Raises:
      ValueError: If data_spec is not an instance or nest of ArraySpecs.
    """
    self._capacity = capacity
    if not all([
        isinstance(spec, array_spec.ArraySpec)
        for spec in tf.nest.flatten(data_spec)
    ]):
      raise ValueError('The data_spec parameter must be an instance or nest of '
                       'array_spec.ArraySpec. Got: {}'.format(data_spec))
    self._data_spec = data_spec
    self._flat_specs = tf.nest.flatten(data_spec)
    self._np_state = tf.contrib.checkpoint.NumpyState()

    self._buf_names = tf.contrib.checkpoint.NoDependency([])
    for idx in range(len(self._flat_specs)):
      self._buf_names.append('buffer{}'.format(idx))
      # Set each buffer to a sentinel value (real buffers will never be
      # scalars) rather than a real value so that if they are restored from
      # checkpoint, we don't end up double-initializing. We don't leave them
      # as unset because setting them to a numpy value tells the checkpointer
      # this will be a checkpointed attribute and it creates TF ops for it.
      setattr(self._np_state, self._buf_names[idx], np.int64(0))

  def _array(self, index):
    """Creates or retrieves one of the numpy arrays backing the storage."""
    array = getattr(self._np_state, self._buf_names[index])
    if np.isscalar(array) or array.ndim == 0:
      spec = self._flat_specs[index]
      shape = (self._capacity,) + spec.shape
      array = np.zeros(shape=shape, dtype=spec.dtype)
      setattr(self._np_state, self._buf_names[index], array)
    return array

  def get(self, idx):
    """Get value stored at idx."""
    encoded_item = []
    for buf_idx in range(len(self._flat_specs)):
      encoded_item.append(self._array(buf_idx)[idx])
    return tf.nest.pack_sequence_as(self._data_spec, encoded_item)

  def set(self, table_idx, value):
    """Set table_idx to value."""
    for nest_idx, element in enumerate(tf.nest.flatten(value)):
      self._array(nest_idx)[table_idx] = element
