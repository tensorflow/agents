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

"""A batched replay buffer of nests of Tensors which can be sampled uniformly.

- Each add assumes tensors have batch_size as first dimension, and will store
each element of the batch in an offset segment, so that each batch dimension has
its own contiguous memory. Within batch segments, behaves as a circular buffer.

The get_next function returns 'ids' in addition to the data. This is not really
needed for the batched replay buffer, but is returned to be consistent with
the API for a priority replay buffer, which needs the ids to update priorities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import numpy as np
import tensorflow as tf

from tf_agents.replay_buffers import replay_buffer
from tf_agents.replay_buffers import table
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from tensorflow.python.data.util import nest as data_nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


BufferInfo = collections.namedtuple('BufferInfo',
                                    ['ids', 'probabilities'])


@gin.configurable
class TFUniformReplayBuffer(replay_buffer.ReplayBuffer):
  """A TFUniformReplayBuffer with batched adds and uniform sampling."""

  def __init__(self,
               data_spec,
               batch_size,
               max_length=1000,
               scope='TFUniformReplayBuffer',
               device='cpu:*',
               table_fn=table.Table):
    """Creates a TFUniformReplayBuffer.

    Args:
      data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing a
        single item that can be stored in this buffer.
      batch_size: Batch dimension of tensors when adding to buffer.
      max_length: The maximum number of items that can be stored in a single
        batch segment of the buffer.
      scope: Scope prefix for variables and ops created by this class.
      device: A TensorFlow device to place the Variables and ops.
      table_fn: Function to create tables `table_fn(data_spec, capacity)` that
        can read/write nested tensors.

    Raises:
      ValueError: If batch_size does not evenly divide capacity.
    """
    self._batch_size = batch_size
    self._max_length = max_length
    capacity = self._batch_size * self._max_length
    super(TFUniformReplayBuffer, self).__init__(data_spec, capacity)

    self._id_spec = tensor_spec.TensorSpec([], dtype=tf.int64, name='id')
    self._capacity_value = np.int64(self._capacity)
    self._batch_offsets = (
        tf.range(self._batch_size, dtype=tf.int64) * self._max_length)
    self._scope = scope
    self._device = device
    self._table_fn = table_fn
    # TODO(sguada) move to create_variables function so we can use make_template
    # to handle this.
    with tf.device(self._device), tf.compat.v1.variable_scope(self._scope):
      self._capacity = tf.constant(capacity, dtype=tf.int64)
      self._data_table = table_fn(self._data_spec, self._capacity_value)
      self._id_table = table_fn(self._id_spec, self._capacity_value)
      self._last_id = common.create_variable('last_id', -1)
      self._last_id_cs = tf.CriticalSection(name='last_id')

  def variables(self):
    return (self._data_table.variables() +
            self._id_table.variables() +
            [self._last_id])

  @property
  def device(self):
    return self._device

  @property
  def table_fn(self):
    return self._table_fn

  @property
  def scope(self):
    return self._scope

  # Methods defined in ReplayBuffer base class

  def _add_batch(self, items):
    """Adds a batch of items to the replay buffer.

    Args:
      items: A tensor or list/tuple/nest of tensors representing a batch of
      items to be added to the replay buffer. Each element of `items` must match
      the data_spec of this class. Should be shape [batch_size, data_spec, ...]
    Returns:
      An op that adds `items` to the replay buffer.
    Raises:
      ValueError: If called more than once.
    """
    tf.nest.assert_same_structure(items, self._data_spec)

    with tf.device(self._device), tf.name_scope(self._scope):
      id_ = self._increment_last_id()
      write_rows = self._get_rows_for_id(id_)
      write_id_op = self._id_table.write(write_rows, id_)
      write_data_op = self._data_table.write(write_rows, items)
      return tf.group(write_id_op, write_data_op)

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    """Returns an item or batch of items sampled uniformly from the buffer.

    Sample transitions uniformly from replay buffer. When sub-episodes are
    desired, specify num_steps, although note that for the returned items to
    truly be sub-episodes also requires that experience collection be
    single-threaded.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See get_next() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See get_next() documentation.
      time_stacked: Bool, when true and num_steps > 1 get_next on the buffer
        would return the items stack on the time dimension. The outputs would be
        [B, T, ..] if sample_batch_size is given or [T, ..] otherwise.
    Returns:
      A 2 tuple, containing:
        - An item, sequence of items, or batch thereof sampled uniformly
          from the buffer.
        - BufferInfo NamedTuple, containing:
          - The items' ids.
          - The sampling probability of each item.
    Raises:
      ValueError: if num_steps is bigger than the capacity.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('get_next'):
        min_val, max_val = self._valid_range_ids(
            self._get_last_id(), self._max_length, num_steps)
        rows_shape = () if sample_batch_size is None else (sample_batch_size,)
        assert_nonempty = tf.compat.v1.assert_greater(
            max_val,
            min_val,
            message='TFUniformReplayBuffer is empty. Make sure to add items '
            'before sampling the buffer.')
        with tf.control_dependencies([assert_nonempty]):
          num_ids = max_val - min_val
          probability = tf.cond(
              pred=tf.equal(num_ids, 0),
              true_fn=lambda: 0.,
              false_fn=lambda: 1. / tf.cast(num_ids * self._batch_size,  # pylint: disable=g-long-lambda
                                            tf.float32))
          ids = tf.random.uniform(
              rows_shape, minval=min_val, maxval=max_val, dtype=tf.int64)

        # Move each id sample to a random batch.
        batch_offsets = tf.random.uniform(
            rows_shape, minval=0, maxval=self._batch_size, dtype=tf.int64)
        batch_offsets *= self._max_length
        ids += batch_offsets

        if num_steps is None:
          rows_to_get = tf.math.mod(ids, self._capacity)
          data = self._data_table.read(rows_to_get)
          data_ids = self._id_table.read(rows_to_get)
        else:
          if time_stacked:
            step_range = tf.range(num_steps, dtype=tf.int64)
            if sample_batch_size:
              step_range = tf.reshape(step_range, [1, num_steps])
              step_range = tf.tile(step_range, [sample_batch_size, 1])
              ids = tf.tile(tf.expand_dims(ids, -1), [1, num_steps])
            else:
              step_range = tf.reshape(step_range, [num_steps])

            rows_to_get = tf.math.mod(step_range + ids, self._capacity)
            data = self._data_table.read(rows_to_get)
            data_ids = self._id_table.read(rows_to_get)
          else:
            data = []
            data_ids = []
            for step in range(num_steps):
              steps_to_get = tf.math.mod(ids + step, self._capacity)
              items = self._data_table.read(steps_to_get)
              data.append(items)
              data_ids.append(self._id_table.read(steps_to_get))
            data = tuple(data)
            data_ids = tuple(data_ids)
        probabilities = tf.fill(rows_shape, probability)

        buffer_info = BufferInfo(ids=data_ids,
                                 probabilities=probabilities)
    return data, buffer_info

  @gin.configurable(
      'tf_agents.tf_uniform_replay_buffer.TFUniformReplayBuffer.as_dataset')
  def as_dataset(self,
                 sample_batch_size=None,
                 num_steps=None,
                 num_parallel_calls=None):
    return self._as_dataset(sample_batch_size, num_steps, num_parallel_calls)

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.
    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).

    Raises:
      ValueError: If the data spec contains lists that must be converted to
        tuples.
    """
    # data_tf.nest.flatten does not flatten python lists, nest.flatten does.
    if tf.nest.flatten(self._data_spec) != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  {}'.format(
              tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

    def get_next(_):
      return self.get_next(sample_batch_size, num_steps, time_stacked=True)

    return tf.data.experimental.Counter().map(
        get_next,
        num_parallel_calls=num_parallel_calls)

  def _gather_all(self):
    """Returns all the items in buffer, shape [batch_size, timestep, ...].

    Returns:
      All the items currently in the buffer.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('gather_all'):
        # Make ids, repeated over batch_size. Shape [batch_size, num_ids, ...].
        min_val, max_val = self._valid_range_ids(self._get_last_id(),
                                                 self._max_length)
        ids = tf.range(min_val, max_val)
        ids = tf.stack([ids] * self._batch_size)
        rows = tf.math.mod(ids, self._max_length)

        # Make batch_offsets, shape [batch_size, 1], then add to rows.
        batch_offsets = tf.expand_dims(tf.range(
            self._batch_size, dtype=tf.int64) * self._max_length, 1)
        rows += batch_offsets

        # Expected shape is [batch_size, max_length, ...].
        data = self._data_table.read(rows)
    return data

  def _clear(self, clear_all_variables=False):
    """Return op that resets the contents of replay buffer.

    Args:
      clear_all_variables: boolean indicating if all variables should be
        cleared. By default, table contents will be unlinked from
        replay buffer, but values are unmodified for efficiency. Set
        `clear_all_variables=True` to reset all variables including Table
        contents.

    Returns:
      op that clears or unlinks the replay buffer contents.
    """
    table_vars = self._data_table.variables() + self._id_table.variables()
    def _init_vars():
      assignments = [self._last_id.assign(-1)]
      if clear_all_variables:
        assignments += [v.assign(tf.zeros_like(v)) for v in table_vars]
      return tf.group(*assignments, name='clear')
    return self._last_id_cs.execute(_init_vars)

  #  Helper functions.

  def _valid_range_ids(self, last_id, max_length, num_steps=None):
    """Returns the [min_val, max_val) range of ids.

    When num_steps is provided, [min_val, max_val+num_steps) are also valid ids.

    Args:
      last_id: The last id added to the buffer.
      max_length: The max length of each batch segment in the buffer.
      num_steps: Optional way to specify that how many ids need to be valid.
    Returns:
      A tuple (min_id, max_id) for the range [min_id, max_id) of valid ids.
    """
    if num_steps is None:
      num_steps = tf.constant(1, tf.int64)

    def non_full():
      min_id = tf.constant(0, dtype=tf.int64)
      max_id = tf.maximum(last_id + 1 - num_steps + 1, 0)
      return min_id, max_id
    def full():
      min_id = last_id + 1 - max_length
      max_id = last_id + 1 - num_steps + 1
      return min_id, max_id

    return tf.cond(pred=last_id < max_length, true_fn=non_full, false_fn=full)

  def _increment_last_id(self, increment=1):
    """Increments the last_id in a thread safe manner.

    Args:
      increment: amount to increment last_id by.
    Returns:
      An op that increments the last_id.
    """
    def _assign_add():
      return self._last_id.assign_add(increment).value()
    return self._last_id_cs.execute(_assign_add)

  def _get_last_id(self):

    def last_id():
      return self._last_id.value()

    return self._last_id_cs.execute(last_id)

  def _get_rows_for_id(self, id_):
    """Make a batch_size length list of tensors, with row ids for write."""
    id_mod = tf.math.mod(id_, self._max_length)
    rows = self._batch_offsets + id_mod
    return rows
