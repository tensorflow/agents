# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
# Modifications Copyright 2021 Kenneth Schroeder. (based on TF-Agents v0.8.0, commit 402b8aa81ca1b578ec1f687725d4ccb4115386d2)
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

"""A batched replay buffer of nests of Tensors which can be sampled prioritized by weights.
- Each add assumes tensors have batch_size as first dimension, and will store
each element of the batch in an offset segment, so that each batch dimension has
its own contiguous memory. Within batch segments, behaves as a circular buffer.
Priorities of new elements are initialized with the maximum observed priority.
The get_next function returns 'ids' in addition to the data, which can be used to update the elements priorities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tf_agents.replay_buffers import replay_buffer
from tf_agents.replay_buffers import table
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils


BufferInfo = collections.namedtuple('BufferInfo',
                                    ['ids', 'probabilities'])


class TFPrioritizedReplayBuffer(replay_buffer.ReplayBuffer):
  """A TFPrioritizedReplayBuffer with batched adds and prioritized sampling."""

  def __init__(self,
               data_spec,
               batch_size,
               max_length=1000,
               scope='TFPrioritizedReplayBuffer',
               device='cpu:*',
               table_fn=table.Table,
               dataset_drop_remainder=False,
               dataset_window_shift=None,
               stateful_dataset=False):
    """Creates a TFPrioritizedReplayBuffer.
    The TFPrioritizedReplayBuffer stores episodes in `B == batch_size` blocks of
    size `L == max_length`, with total frame capacity
    `C == L * B`.  Storage looks like:
    ```
    block1 ep1 frame1
               frame2
           ...
           ep2 frame1
               frame2
           ...
           <L frames total>
    block2 ep1 frame1
               frame2
           ...
           ep2 frame1
               frame2
           ...
           <L frames total>
    ...
    blockB ep1 frame1
               frame2
           ...
           ep2 frame1
               frame2
           ...
           <L frames total>
    ```
    Multiple episodes may be stored within a given block, up to `max_length`
    frames total.  In practice, new episodes will overwrite old ones as the
    block rolls over its `max_length`.
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
      dataset_drop_remainder: If `True`, then when calling
        `as_dataset` with arguments `single_deterministic_pass=True` and
        `sample_batch_size is not None`, the final batch will be dropped if it
        does not contain exactly `sample_batch_size` items.  This is helpful for
        static shape inference as the resulting tensors will always have
        leading dimension `sample_batch_size` instead of `None`.
      dataset_window_shift: Window shift used when calling `as_dataset` with
        arguments `single_deterministic_pass=True` and `num_steps is not None`.
        This determines how the resulting frames are windowed.  If `None`, then
        there is no overlap created between frames and each frame is seen
        exactly once.  For example, if `max_length=5`, `num_steps=2`,
        `sample_batch_size=None`, and `dataset_window_shift=None`, then the
        datasets returned will have frames `{[0, 1], [2, 3], [4]}`.
        If `dataset_window_shift is not None`, then windows are created with a
        window overlap of `dataset_window_shift` and you will see each frame up
        to `num_steps` times.  For example, if `max_length=5`, `num_steps=2`,
        `sample_batch_size=None`, and `dataset_window_shift=1`, then the
        datasets returned will have windows of shifted repeated frames:
        `{[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]}`.
        For more details, see the documentation of `tf.data.Dataset.window`,
        specifically for the `shift` argument.
        The default behavior is to not overlap frames
        (`dataset_window_shift=None`) but users often want to see all
        combinations of frame sequences, in which case `dataset_window_shift=1`
        is the appropriate value.
      stateful_dataset: whether the dataset contains stateful ops or not.
    """
    self._batch_size = batch_size
    self._max_length = max_length
    capacity = self._batch_size * self._max_length
    super(TFPrioritizedReplayBuffer, self).__init__(
        data_spec, capacity, stateful_dataset)

    self._id_spec = tensor_spec.TensorSpec([], dtype=tf.int64, name='id')
    self._prio_spec = tensor_spec.TensorSpec([], dtype=tf.float32, name='prio')
    self._capacity_value = np.int64(self._capacity)
    self._batch_offsets = (
        tf.range(self._batch_size, dtype=tf.int64) * self._max_length)
    self._scope = scope
    self._device = device
    self._table_fn = table_fn
    self._dataset_drop_remainder = dataset_drop_remainder
    self._dataset_window_shift = dataset_window_shift
    with tf.device(self._device), tf.compat.v1.variable_scope(self._scope):
      self._capacity = tf.constant(capacity, dtype=tf.int64)
      self._data_table = table_fn(self._data_spec, self._capacity_value)
      self._id_table = table_fn(self._id_spec, self._capacity_value)
      self._prio_table = table_fn(self._prio_spec, self._capacity_value)
      self._last_id = common.create_variable('last_id', -1)
      self._last_id_cs = tf.CriticalSection(name='last_id')
    self._max_prio = tf.constant(0.001)

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

  def _num_frames(self):
    num_items_single_batch_segment = self._get_last_id() + 1
    total_frames = num_items_single_batch_segment * self._batch_size
    return tf.minimum(total_frames, self._capacity)

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
    nest_utils.assert_same_structure(items, self._data_spec)
    # Calling get_outer_rank here will validate that all items have the same
    # outer rank. This was not usually an issue, but now that it's easier to
    # call this from an eager context it's easy to make the mistake.
    nest_utils.get_outer_rank(
        tf.nest.map_structure(tf.convert_to_tensor, items),
        self._data_spec)

    with tf.device(self._device), tf.name_scope(self._scope):
      id_ = self._increment_last_id()
      write_rows = self._get_rows_for_id(id_)
      write_id_op = self._id_table.write(write_rows, id_)
      write_data_op = self._data_table.write(write_rows, items)
      write_prio_op = self._prio_table.write(write_rows, tf.fill(tf.size(write_rows), self._max_prio))
      return tf.group(write_id_op, write_data_op, write_prio_op)

  def update_batch(self, ids, weights):
    nest_utils.assert_same_structure(weights, self._prio_spec)
    self._max_prio = tf.maximum(self._max_prio, tf.reduce_max(weights))
    with tf.device(self._device), tf.name_scope(self._scope):
      write_prio_op = self._prio_table.write(ids, weights)
      return write_prio_op

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
        min_val, max_val = _valid_range_ids(
            self._get_last_id(), self._max_length, num_steps)
        rows_shape = () if sample_batch_size is None else (sample_batch_size,)
        assert_nonempty = tf.compat.v1.assert_greater(
            max_val,
            min_val,
            message='TFPrioritizedReplayBuffer is empty. Make sure to add items '
            'before sampling the buffer.')
        with tf.control_dependencies([assert_nonempty]):
          num_valid_ids = max_val - min_val
          
          batch_offsets = tf.range(0, self._batch_size, dtype=tf.int64) * self._max_length
          batch_ids = tf.range(min_val, max_val, dtype=tf.int64)
          # to each batch_offset, add batch_ids
          prio_ids = tf.map_fn(fn=lambda t: tf.add(t, batch_ids), elems=batch_offsets)
          prio_ids = tf.reshape(prio_ids, [-1])
          prio_ids = tf.math.mod(prio_ids, self._capacity) # because batch-ids are not capped, we need to do a mod here

          probabilities = self._prio_table.read(prio_ids)
          log_probabilities = tf.math.log([self._prio_table.read(prio_ids)])
            
          prio_ids_ids = tf.random.categorical(log_probabilities, rows_shape[0])
          chosen_element_ids = tf.gather(prio_ids, prio_ids_ids)
          chosen_element_ids = tf.reshape(chosen_element_ids, [-1])
          chosen_element_probabilities = tf.gather(probabilities, prio_ids_ids)
          chosen_element_probabilities = tf.reshape(chosen_element_probabilities, [-1])

        if num_steps is None:
          rows_to_get = tf.math.mod(chosen_element_ids, self._capacity)
          data = self._data_table.read(rows_to_get)
          data_ids = self._id_table.read(rows_to_get)
        else:
          if time_stacked:
            step_range = tf.range(num_steps, dtype=tf.int64)
            if sample_batch_size:
              step_range = tf.reshape(step_range, [1, num_steps])
              step_range = tf.tile(step_range, [sample_batch_size, 1])
              chosen_element_ids_expanded = tf.tile(tf.expand_dims(chosen_element_ids, -1), [1, num_steps])
            else:
              step_range = tf.reshape(step_range, [num_steps])

            rows_to_get = tf.math.mod(step_range + chosen_element_ids_expanded, self._capacity)
            data = self._data_table.read(rows_to_get)
            data_ids = self._id_table.read(rows_to_get)
          else:
            data = []
            data_ids = []
            for step in range(num_steps):
              steps_to_get = tf.math.mod(chosen_element_ids + step, self._capacity)
              items = self._data_table.read(steps_to_get)
              data.append(items)
              data_ids.append(self._id_table.read(steps_to_get))
            data = tuple(data)
            data_ids = tuple(data_ids)

        buffer_info = BufferInfo(ids=chosen_element_ids, # this was returning the batch ID of each element, now we return actual ids to allow updates
                                 probabilities=chosen_element_probabilities)
    return data, buffer_info

  def as_dataset(self,
                 sample_batch_size=None,
                 num_steps=None,
                 num_parallel_calls=None,
                 single_deterministic_pass=False):
    return super(TFPrioritizedReplayBuffer, self).as_dataset(
        sample_batch_size, num_steps, num_parallel_calls,
        single_deterministic_pass=single_deterministic_pass)

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  sequence_preprocess_fn=None,
                  num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer in shuffled order.
    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.
      sequence_preprocess_fn: (Optional.) Preprocessing function for sequences
        before they are sharded into subsequences of length `num_steps` and
        batched.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.
    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).
    Raises:
      NotImplementedError: If `sequence_preprocess_fn != None` is passed in.
    """
    if sequence_preprocess_fn is not None:
      raise NotImplementedError('sequence_preprocess_fn is not supported.')

    def get_next(_):
      return self.get_next(sample_batch_size, num_steps, time_stacked=True)

    dataset = tf.data.experimental.Counter().map(
        get_next, num_parallel_calls=num_parallel_calls)
    return dataset

  def _single_deterministic_pass_dataset(self,
                                         sample_batch_size=None,
                                         num_steps=None,
                                         sequence_preprocess_fn=None,
                                         num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer in fixed order.
    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.
      sequence_preprocess_fn: (Optional.) Preprocessing function for sequences
        before they are sharded into subsequences of length `num_steps` and
        batched.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.
    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).
    Raises:
      ValueError: If `dataset_drop_remainder` is set, and
        `sample_batch_size > self.batch_size`.  In this case all data will
        be dropped.
      NotImplementedError: If `sequence_preprocess_fn != None` is passed in.
    """
    if sequence_preprocess_fn is not None:
      raise NotImplementedError('sequence_preprocess_fn is not supported.')
    static_size = tf.get_static_value(sample_batch_size)
    static_num_steps = tf.get_static_value(num_steps)
    static_self_batch_size = tf.get_static_value(self._batch_size)
    static_self_max_length = tf.get_static_value(self._max_length)
    if (self._dataset_drop_remainder
        and static_size is not None
        and static_self_batch_size is not None
        and static_size > static_self_batch_size):
      raise ValueError(
          'sample_batch_size ({}) > self.batch_size ({}) and '
          'dataset_drop_remainder is True.  In '
          'this case, ALL data will be dropped by the deterministic dataset.'
          .format(static_size, static_self_batch_size))
    if (self._dataset_drop_remainder
        and static_num_steps is not None
        and static_self_max_length is not None
        and static_num_steps > static_self_max_length):
      raise ValueError(
          'num_steps_size ({}) > self.max_length ({}) and '
          'dataset_drop_remainder is True.  In '
          'this case, ALL data will be dropped by the deterministic dataset.'
          .format(static_num_steps, static_self_max_length))

    def get_row_ids(_):
      """Passed to Dataset.range(self._batch_size).flat_map(.), gets row ids."""
      with tf.device(self._device), tf.name_scope(self._scope):
        with tf.name_scope('single_deterministic_pass_dataset'):
          # Here we pass num_steps=None because _valid_range_ids uses
          # num_steps to determine a hard stop when sampling num_steps starting
          # from the returned indices.  But in our case, we want all the indices
          # and we'll use TF dataset's window() mechanism to get
          # num_steps-length blocks.  The window mechanism handles this stuff
          # for us.
          min_frame_offset, max_frame_offset = _valid_range_ids(
              self._get_last_id(), self._max_length, num_steps=None)
          tf.compat.v1.assert_less(
              min_frame_offset,
              max_frame_offset,
              message='TFPrioritizedReplayBuffer is empty. Make sure to add items '
              'before asking the buffer for data.')

          min_max_frame_range = tf.range(min_frame_offset, max_frame_offset)

          window_shift = self._dataset_window_shift
          def group_windows(ds_, drop_remainder=self._dataset_drop_remainder):
            return ds_.batch(num_steps, drop_remainder=drop_remainder)

          if sample_batch_size is None:
            def row_ids(b):
              # Create a vector of shape [num_frames] and slice it along each
              # frame.
              ids = tf.data.Dataset.from_tensor_slices(
                  b * self._max_length + min_max_frame_range)
              if num_steps is not None:
                ids = (ids.window(num_steps, shift=window_shift)
                       .flat_map(group_windows))
              return ids
            return tf.data.Dataset.range(self._batch_size).flat_map(row_ids)
          else:
            def batched_row_ids(batch):
              # Create a matrix of indices shaped [num_frames, batch_size]
              # and slice it along each frame row to get groups of batches
              # for frame 0, frame 1, ...
              return tf.data.Dataset.from_tensor_slices(
                  (min_max_frame_range[:, tf.newaxis]
                   + batch * self._max_length))

            indices_ds = (
                tf.data.Dataset.range(self._batch_size)
                .batch(sample_batch_size,
                       drop_remainder=self._dataset_drop_remainder)
                .flat_map(batched_row_ids))

            if num_steps is not None:
              # We have sequences of num_frames rows shaped [sample_batch_size].
              # Window and group these to rows of shape
              # [num_steps, sample_batch_size], then
              # transpose them to get index tensors of shape
              # [sample_batch_size, num_steps].
              def group_windows_drop_remainder(d):
                return group_windows(d, drop_remainder=True)

              indices_ds = (indices_ds.window(num_steps, shift=window_shift)
                            .flat_map(group_windows_drop_remainder)
                            .map(tf.transpose))

            return indices_ds

    # Get our indices as a dataset; each time we reinitialize the iterator we
    # update our min/max id bounds from the state of the replay buffer.
    ds = tf.data.Dataset.range(1).flat_map(get_row_ids)

    def get_data(id_):
      with tf.device(self._device), tf.name_scope(self._scope):
        with tf.name_scope('single_deterministic_pass_dataset'):
          data = self._data_table.read(id_ % self._capacity)
      buffer_info = BufferInfo(ids=id_, probabilities=())
      return (data, buffer_info)

    # Deterministic even though num_parallel_calls > 1.  Operations are
    # run in parallel but then the results are returned in original stream
    # order.
    ds = ds.map(get_data, num_parallel_calls=num_parallel_calls)

    return ds

  def _gather_all(self):
    """Returns all the items in buffer, shape [batch_size, timestep, ...].
    Returns:
      All the items currently in the buffer.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('gather_all'):
        # Make ids, repeated over batch_size. Shape [batch_size, num_ids, ...].
        min_val, max_val = _valid_range_ids(
            self._get_last_id(), self._max_length)
        ids = tf.range(min_val, max_val)
        ids = tf.stack([ids] * self._batch_size)
        rows = tf.math.mod(ids, self._max_length)

        # Make batch_offsets, shape [batch_size, 1], then add to rows.
        batch_offsets = tf.expand_dims(
            tf.range(self._batch_size, dtype=tf.int64) * self._max_length,
            1)
        rows += batch_offsets

        # Expected shape is [batch_size, max_length, ...].
        data = self._data_table.read(rows)
    return data

  def _clear(self, clear_all_variables=False): # TODO may need to clear weights as well
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
    table_vars = self._data_table.variables() + self._id_table.variables() + self._prio_table.variables()
    def _init_vars():
      assignments = [self._last_id.assign(-1)]
      if clear_all_variables:
        assignments += [v.assign(tf.zeros_like(v)) for v in table_vars]
      return tf.group(*assignments, name='clear')
    self._max_prio = tf.constant(0.001)
    return self._last_id_cs.execute(_init_vars)

  #  Helper functions.
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

  min_id_not_full = tf.constant(0, dtype=tf.int64)
  max_id_not_full = tf.maximum(last_id + 1 - num_steps + 1, 0)

  min_id_full = last_id + 1 - max_length
  max_id_full = last_id + 1 - num_steps + 1

  return (tf.where(last_id < max_length, min_id_not_full, min_id_full),
          tf.where(last_id < max_length, max_id_not_full, max_id_full))