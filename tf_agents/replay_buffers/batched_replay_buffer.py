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

import numpy as np
import tensorflow as tf

from tf_agents.replay_buffers import table
from tf_agents.specs import tensor_spec

from tensorflow.python.data.util import nest as data_nest  # TF internal
import gin.tf


nest = tf.contrib.framework.nest


def _valid_range_ids(last_id, max_length, num_steps=None):
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

  return tf.cond(last_id < max_length, non_full, full)


@gin.configurable
class BatchedReplayBuffer(tf.contrib.eager.Checkpointable):
  """A BatchedReplayBuffer with batched adds and uniform sampling."""

  def __init__(self,
               data_spec,
               batch_size,
               max_length=1000,
               scope='BatchedReplayBuffer',
               device='cpu:*',
               table_fn=table.Table):
    """Creates a BatchedReplayBuffer.

    Args:
      data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing
        a single item that can be stored in this buffer.
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
    self._data_spec = data_spec
    self._batch_size = batch_size
    self._max_length = max_length
    capacity = self._batch_size * self._max_length
    self._capacity = capacity
    self._id_spec = tensor_spec.TensorSpec([], dtype=tf.int64, name='id')
    self._capacity_value = np.int64(self._capacity)
    self._batch_offsets = (
        tf.range(self._batch_size, dtype=tf.int64) * self._max_length)
    self._scope = scope
    self._device = device
    self._table_fn = table_fn
    # TODO(sguada) move to create_variables function so we can use make_template
    # to handle this.
    with tf.device(self._device), tf.variable_scope(self._scope):
      self._capacity = tf.constant(capacity, dtype=tf.int64)
      self._data_table = table_fn(self._data_spec, self._capacity_value)
      self._id_table = table_fn(self._id_spec, self._capacity_value)
      self._last_id = tf.get_variable(
          name='last_id',
          shape=[],
          dtype=tf.int64,
          initializer=tf.constant_initializer(-1, dtype=tf.int64),
          use_resource=True,
          trainable=False)
      self._last_id_cs = tf.contrib.framework.CriticalSection(name='last_id')

  def variables(self):
    # TODO(sguada) - make this Eager-compatible. Don't rely on scopes.
    return tf.contrib.framework.get_variables(self._scope)

  @property
  def data_spec(self):
    return self._data_spec

  @property
  def capacity(self):
    return self._capacity_value

  @property
  def device(self):
    return self._device

  @property
  def table_fn(self):
    return self._table_fn

  @property
  def scope(self):
    return self._scope

  def clear(self, clear_all_variables=False):
    """Return op that resets the contents of replay buffer.

    Args:
      clear_all_variables: By default, table contents will be unlinked from
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

  def get_last_id(self):

    def last_id():
      return self._last_id.value()

    return self._last_id_cs.execute(last_id)

  def _get_rows_for_id(self, id_):
    """Make a batch_size length list of tensors, with row ids for write."""
    id_mod = tf.mod(id_, self._max_length)
    rows = self._batch_offsets + id_mod
    return rows

  def add_batch(self, items):
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
    nest.assert_same_structure(items, self._data_spec)

    with tf.device(self._device), tf.name_scope(self._scope):
      id_ = self._increment_last_id()
      write_rows = self._get_rows_for_id(id_)
      write_id_op = self._id_table.write(write_rows, id_)
      write_data_op = self._data_table.write(write_rows, items)
      return tf.group(write_id_op, write_data_op)

  def maybe_add(self, *_):
    """Adds items to the replay buffer. Assumes batch_size as first dim."""
    raise NotImplementedError('BatchedReplayBuffer does not support maybe_add.')

  def add_sequence(self, *_):
    """Adds a sequence of items to the replay buffer."""
    raise NotImplementedError('BatchedReplayBuffer does not support adding '
                              'sequences of steps.')

  @gin.configurable(
      'tf_agents.batched_replay_buffer.BatchedReplayBuffer.as_dataset')
  def as_dataset(self, num_parallel_calls=None, **kwargs):
    """Creates a dataset that returns entries from the buffer.

    A single entry from the dataset is equivalent to one output from
    `get_next(num_steps=num_steps)`.

    Args:
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process in parallel. If not
        specified, elements will be processed sequentially.
      **kwargs: Extra arguments passed to get_next; see `get_next`
        documentation.


    Returns:
      A dataset whose entries are identical to the output of `get_next`
      with `batch_size=None`:

      A 3-tuple containing:

        - An item or sequence of items, sampled uniformly from the buffer.
        - The items' ids.
        - The probabilities for each sampled item. This is useful for e.g.
          importance sampling corrections.

    Raises:
      ValueError: If the data spec contains lists that must be converted to
        tuples.
    """
    # data_nest.flatten does not flatten python lists, nest.flatten does.
    if nest.flatten(self._data_spec) != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  {}'.format(
              nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

    return tf.data.experimental.Counter().map(
        lambda _: self.get_next(**kwargs),
        num_parallel_calls=num_parallel_calls)

  def get_next(self, sample_batch_size=None, num_steps=None,
               time_stacked=False):
    """Returns an item or batch of items sampled uniformly from the buffer.

    Sample transitions uniformly from replay buffer. When sub-episodes are
    desired, specify num_steps, although note that for the returned items to
    truly be sub-episodes also requires that experience collection be
    single-threaded.

    Args:
      sample_batch_size: An optional batch_size to specify the number of items
        to sample. If None (default), a single item is returned which matches
        the data_spec of this class (without a batch dimension). Otherwise,
        a batch of batch_size items is returned, where each tensor in items
        will have its first dimension equal to batch_size and the rest of the
        dimensions match the corresponding spec in data_spec.
      num_steps: Optional way to specify that sub-episodes are desired.
        If None (default), a batch of single items is returned. Otherwise,
        a batch of sub-episodes is returned, where a sub-episode is a sequence
        of consecutive items in the replay_buffer.  The returned tensors
        will have first dimension equal to batch_size (if batch_size is not
        None), subsequent dimension equal to num_steps, and remaining
        dimensions which match the data_spec of this class.
      time_stacked: Bool, when true and num_steps > 1 it would return the items
        stack on the time dimension. The outputs would be [B, T, ..] if
        sample_batch_size is given or [T, ..] otherwise.
    Returns:
      An item, sequence of items, or batch thereof sampled uniformly from the
      buffer.
      The items' ids.
      The sampling probability of each item.
    Raises:
      ValueError: if num_steps is bigger than the capacity.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('get_next'):
        min_val, max_val = _valid_range_ids(
            self.get_last_id(), self._max_length, num_steps)
        rows_shape = () if sample_batch_size is None else (sample_batch_size,)
        assert_nonempty = tf.assert_greater(
            max_val,
            min_val,
            message='BatchedReplayBuffer is empty. Make sure to add items '
            'before sampling the buffer.')
        with tf.control_dependencies([assert_nonempty]):
          ids = tf.random_uniform(
              rows_shape, minval=min_val, maxval=max_val, dtype=tf.int64)

        # Move each id sample to a random batch.
        batch_offsets = tf.random_uniform(
            rows_shape, minval=0, maxval=self._batch_size, dtype=tf.int64)
        batch_offsets *= self._max_length
        ids += batch_offsets

        if num_steps is None:
          rows_to_get = tf.mod(ids, self._capacity)
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

            rows_to_get = tf.mod(step_range + ids, self._capacity)
            data = self._data_table.read(rows_to_get)
            data_ids = self._id_table.read(rows_to_get)
          else:
            data = []
            data_ids = []
            for step in range(num_steps):
              steps_to_get = tf.mod(ids + step, self._capacity)
              items = self._data_table.read(steps_to_get)
              data.append(items)
              data_ids.append(self._id_table.read(steps_to_get))
            data = tuple(data)
            data_ids = tuple(data_ids)
        num_ids = max_val - min_val
        probability = tf.cond(
            tf.equal(num_ids, 0), lambda: 0.,
            lambda: 1. / tf.to_float(num_ids * self._batch_size))
        probabilities = tf.fill(rows_shape, probability)
    return data, data_ids, probabilities

  def gather_all(self):
    """Returns all the items in buffer, shape [batch_size, timestep, ...].

    Returns:
      All the items currently in the buffer.
      The items ids.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('gather_all'):
        # Make ids, repeated over batch_size. Shape [batch_size, num_ids, ...].
        min_val, max_val = _valid_range_ids(self.get_last_id(),
                                            self._max_length)
        ids = tf.range(min_val, max_val)
        ids = tf.stack([ids] * self._batch_size)
        rows = tf.mod(ids, self._max_length)

        # Make batch_offsets, shape [batch_size, 1], then add to rows.
        batch_offsets = tf.expand_dims(tf.range(
            self._batch_size, dtype=tf.int64) * self._max_length, 1)
        rows += batch_offsets

        # Expected shape is [batch_size, max_length, ...].
        data = self._data_table.read(rows)
        ids = self._id_table.read(rows)
    return data, ids
