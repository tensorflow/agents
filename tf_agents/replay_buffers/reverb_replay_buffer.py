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

# Lint as: python3
"""Reverb as a TF-Agents replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow.compat.v2 as tf

from tf_agents.replay_buffers import replay_buffer
from tf_agents.typing import types
from tf_agents.utils import lazy_loader

# Lazy loading since not all users have the reverb package installed.
reverb = lazy_loader.LazyLoader('reverb', globals(), 'reverb')


# The default table name.
DEFAULT_TABLE = 'experience'


@gin.configurable
class ReverbReplayBuffer(replay_buffer.ReplayBuffer):
  """Reverb ReplayBuffer exposed as a TF-Agents replay buffer."""

  def __init__(self,
               data_spec,
               table_name,
               sequence_length,
               server_address=None,
               local_server=None,
               dataset_buffer_size=None,
               max_cycle_length=32,
               num_workers_per_iterator=-1,
               max_samples_per_stream=-1,
               rate_limiter_timeout_ms=-1,
               ):
    """Initializes a reverb replay buffer.

    *NOTE*: If the user calls `as_dataset()` with a value of `num_steps` that
    does not match the `sequence_length` used in the observer (or if
    `sequence_length is None` varies from episode to episode), then individual
    sequences will be truncated to the highest possible multiple of `num_steps`
    before being split. In this case, **data loss may occur**. For example, in
    sparse reward cases where there's a reward only at the final frame and the
    sequence length is not a multiple of `num_steps`, the reward will be
    dropped.

    When using sparse rewards at the end of sequences, the user should prefer
    to either request full episodes (e.g., call `as_dataset` with
    `num_steps=None` and `sample_batch_size=None` (for variable-length
    episodes)); or use `sequence_preprocess_fn` to first propagate the rewards
    from the end of the sequence through to the earlier frames using e.g.
    temporal discounting.

    Args:
      data_spec: Spec for the data held in the replay buffer.
      table_name: Name of the table that will be sampled.
      sequence_length: (can be set to `None`, i.e unknown) The number of
        timesteps that each sample consists of. If not `None`, then the lengths
        of samples received from the server will be validated against this
        number.

        **NOTE** This replay buffer will be at its most performant
        if the `sequence_length` here is equal to `num_steps` passed to
        `as_dataset`, and is also used when writing to the replay buffer
        (for example, see the `sequence_lengths` argument of the
        `Reverb.*Observer` classes).
      server_address: (Optional) Address of the reverb replay server. One of
        `server_address` or `local_server` must be provided.
      local_server: (Optional) An instance of `reverb.Server` that holds
        the replay's data.
      dataset_buffer_size: (Optional) This is the prefetch buffer size
        (in number of items) of the Reverb Dataset object.  A good rule of
        thumb is to set this value to 2-3x times the sample_batch_size you
        will use.
      max_cycle_length: (Optional) The number of sequences used to populate the
        batches of `as_dataset`.  By default, `min(32, sample_batch_size)` is
        used, but the number can be between `1` and `sample_batch_size`.
      num_workers_per_iterator: (Defaults to -1, i.e auto selected) The number
        of worker threads to create per dataset iterator. When the selected
        table uses a FIFO or Heap sampler (i.e a queue) then exactly 1 worker
        must be used to avoid races causing invalid ordering of items. For all
        other samplers, this value should be roughly equal to the number of
        threads available on the CPU.
      max_samples_per_stream: (Defaults to -1, i.e auto selected) The maximum
        number of samples to fetch from a stream before a new call is made.
        Keeping this number low ensures that the data is fetched uniformly from
        all servers.
      rate_limiter_timeout_ms: (Defaults to -1: infinite).  Timeout
        (in milliseconds) to wait on the rate limiter when sampling from the
        table. If `rate_limiter_timeout_ms >= 0`, this is the timeout passed to
        `Table::Sample` describing how long to wait for the rate limiter to
        allow sampling.
    """
    if (server_address is None) == (local_server is None):
      raise ValueError(
          'Exactly one of the server_address or local_server must be provided.')

    self._table_name = table_name
    self._sequence_length = sequence_length
    self._local_server = local_server
    self._server_address = server_address
    self._dataset_buffer_size = dataset_buffer_size
    self._max_cycle_length = max_cycle_length
    # TODO(b/156531956) Remove these
    self._num_workers_per_iterator = num_workers_per_iterator
    self._max_samples_per_stream = max_samples_per_stream
    self._rate_limiter_timeout_ms = rate_limiter_timeout_ms

    if local_server:
      self._server_address = 'localhost:{}'.format(local_server.port)

    self._py_client = reverb.Client(self._server_address)
    self._tf_client = reverb.TFClient(self._server_address, 'rb_tf_client')
    self._table_info = self.get_table_info()
    sampler = self._table_info.sampler_options
    remover = self._table_info.remover_options
    self._deterministic_table = (
        sampler.is_deterministic and remover.is_deterministic)

    capacity = self._table_info.max_size
    super(ReverbReplayBuffer, self).__init__(
        data_spec=data_spec, capacity=capacity, stateful_dataset=True)

  @property
  def py_client(self) -> types.ReverbClient:
    return self._py_client

  @property
  def local_server(self) -> types.ReverbServer:
    return self._local_server

  @property
  def tf_client(self) -> types.ReverbTFClient:
    return self._tf_client

  def get_table_info(self):
    return self._py_client.server_info()[self._table_name]

  def _num_frames(self):
    """Returns the number of frames in the replay buffer.

    **Note:** This might return inconsistent result if there are pending
    reads/writes are running on the particular table in parallel on the Reverb
    server side.

    Returns:
      The number of observations stored in `self._table_name` table of the
      Reverb server.
    """
    return self.get_table_info().current_size

  def add_batch(self, items):
    """Adds a batch of items to the replay buffer.

    ***Warning***: `ReverbReplayBuffer` does not support `add_batch`. See
    `reverb_utils.ReverbObserver` for more information on how to add data
    to the buffer.

    Args:
      items: Ignored.

    Returns:
      Nothing.

    Raises: NotImplementedError
    """
    raise NotImplementedError(
        'ReverbReplayBuffer does not support `add_batch`. See '
        '`reverb_utils.ReverbObserver` for more information on how to add data '
        'to the buffer.')

  def get_next(self, sample_batch_size=None, num_steps=None, time_stacked=True):
    """Returns an item or batch of items from the buffer.

    ***Warning***: `ReverbReplayBuffer` does not support `get_next`. See
    `reverb_utils.ReverbObserver` for more information on how to retrieve data
    from the buffer.

    Args:
      sample_batch_size: Ignored.
      num_steps: Ignored.
      time_stacked: Ignored.

    Returns:
      Nothing.

    Raises:
      NotImplementedError
    """
    raise NotImplementedError('ReverbReplayBuffer does not support `get_next`.')

  def _as_dataset(self,
                  sample_batch_size,
                  num_steps,
                  sequence_preprocess_fn,
                  num_parallel_calls):
    """Creates and returns a dataset that returns entries from the buffer.

    *NOTE*: If `num_steps` does not match the `sequence_length` used in the
    observer (or if `sequence_length is None` varies from episode to episode),
    then individual sequences will be truncated to the highest possible multiple
    of `num_steps` before being split.  In this case, **data loss may occur**.
    For example, in sparse reward cases where there's a reward only at the final
    frame and the sequence length is not a multiple of `num_steps`, the
    reward will be dropped.

    When using sparse rewards at the end of sequences, the user should prefer
    to either request full episodes (e.g., `num_steps=None` and
    `sample_batch_size=None` (for variable-length episodes)); or use
    `sequence_preprocess_fn` to first propagate the rewards from the end
    of the sequence through to the earlier frames using e.g.
    temporal discounting.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. If None (default), a single item is returned
        which matches the data_spec of this class (without a batch dimension).
        Otherwise, a batch of sample_batch_size items is returned, where each
        tensor in items will have its first dimension equal to sample_batch_size
        and the rest of the dimensions match the corresponding data_spec.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. If None (default), a batch of single items is returned.
        Otherwise, a batch of sub-episodes is returned, where a sub-episode is a
        sequence of consecutive items in the replay_buffer. The returned tensors
        will have first dimension equal to sample_batch_size (if
        sample_batch_size is not None), subsequent dimension equal to num_steps,
        and remaining dimensions which match the data_spec of this class.
      sequence_preprocess_fn: (Optional) fn for preprocessing the collected data
        before it is split into subsequences of length `num_steps`. Defined in
        `TFAgent.preprocess_sequence`. Defaults to pass through.
      num_parallel_calls: (Optional) Used to parallelize unpacking samples.

    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).
    """
    self._verify_num_steps(num_steps)

    if (num_parallel_calls and sample_batch_size
        and num_parallel_calls > sample_batch_size):
      raise ValueError(
          'num_parallel_calls cannot be bigger than sample_batch_size '
          '{} > {}'.format(num_parallel_calls, sample_batch_size))
    num_parallel_calls = num_parallel_calls or tf.data.experimental.AUTOTUNE
    total_batch_size = sample_batch_size or 1
    # This determines how many parallel Reverb dataset pipelines we create -
    # aka "how many interleaves."
    cycle_length = min(total_batch_size, self._max_cycle_length)
    batch_size_per_interleave = total_batch_size // cycle_length
    # Recomended buffer_size per connection is ~2-3x the batch size.
    dataset_buffer_size = (
        self._dataset_buffer_size or 3 * batch_size_per_interleave)
    # Set a maximum number of workers per iterator due to interleave
    num_workers_per_iterator = min(
        self._num_workers_per_iterator, batch_size_per_interleave)

    def per_sequence_fn(sample):
      # At this point, each sample data contains a sequence of trajectories.
      data, info = sample.data, sample.info
      if sequence_preprocess_fn:
        data = sequence_preprocess_fn(data)
      return data, info

    def dataset_transformation(dataset):
      if num_steps and num_steps != self._sequence_length:
        dataset = (
            dataset
            # Truncate and reshape elements to [X, num_steps, ...]
            .map(lambda *s: truncate_reshape_rows_by_num_steps(s, num_steps))
            # Unbatch to get elements shaped [num_steps, ...]; each element
            # contains non-overlapping time steps.
            .unbatch())
        shuffle_size = 100
        if self._sequence_length:
          shuffle_size = self._sequence_length // num_steps
        # We will receive batches from interleaves of size
        # cycle_length and batching them to size sample_batch_size.
        # To try and ensure i.i.d. samples in each minibatch, make the shuffle
        # buffer larger.
        shuffle_size *= batch_size_per_interleave
        dataset = dataset.shuffle(shuffle_size)
      return dataset

    dataset = make_reverb_dataset(
        self._server_address,
        self._table_name,
        data_spec=self._data_spec,
        max_in_flight_samples_per_worker=dataset_buffer_size,
        batch_size=sample_batch_size,
        sequence_length=self._sequence_length,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
        per_sequence_fn=per_sequence_fn,
        dataset_transformation=dataset_transformation,
        num_workers_per_iterator=num_workers_per_iterator,
        rate_limiter_timeout_ms=self._rate_limiter_timeout_ms)

    return dataset

  def _single_deterministic_pass_dataset(self,
                                         sample_batch_size=None,
                                         num_steps=None,
                                         sequence_preprocess_fn=None,
                                         num_parallel_calls=None):
    """Creates and returns a dataset that returns entries from the buffer.

    *NOTE*: If `num_steps` does not match the `sequence_length` used in the
    observer (or if `sequence_length is None` varies from episode to episode),
    then individual sequences will be truncated to the highest possible multiple
    of `num_steps` before being split.  In this case, data loss may occur.
    For example, in sparse reward cases where there's a reward only at the final
    frame and the sequence length is not a multiple of `num_steps`, the
    reward will be dropped.

    *NOTE*: If `num_steps` does not match the `sequence_length` used in the
    observer (or if `sequence_length is None` varies from episode to episode),
    then individual sequences will be truncated to the highest possible multiple
    of `num_steps` before being split.  In this case, **data loss may occur**.
    For example, in sparse reward cases where there's a reward only at the final
    frame and the sequence length is not a multiple of `num_steps`, the
    reward will be dropped.

    When using sparse rewards at the end of sequences, the user should prefer
    to either request full episodes (e.g., `num_steps=None` and
    `sample_batch_size=None` (for variable-length episodes)); or use
    `sequence_preprocess_fn` to first propagate the rewards from the end
    of the sequence through to the earlier frames using e.g.
    temporal discounting.

    Args:
      sample_batch_size: See _as_dataset.
      num_steps: See _as_dataset.
      sequence_preprocess_fn: See _as_dataset.
      num_parallel_calls: Ignored.

    Returns:
      See `_as_dataset`.
    """
    if not self._deterministic_table:
      raise ValueError(
          'Unable to perform a single deterministic pass over the dataset, '
          'since either the sampler or the remover is not deterministic '
          '(FIFO or Heap).  Table info:\n{}'.format(self._table_info))
    self._verify_num_steps(num_steps)

    def per_sequence_fn(sample):
      # At this point, each sample data contains a sequence of trajectories.
      data, info = sample.data, sample.info
      if sequence_preprocess_fn:
        data = sequence_preprocess_fn(data)
      return data, info

    def dataset_transformation(dataset):
      if num_steps and num_steps != self._sequence_length:
        dataset = (
            dataset
            # Truncate and reshape elements to [X, num_steps, ...]
            .map(lambda *s: truncate_reshape_rows_by_num_steps(s, num_steps))
            # Unbatch to get elements shaped [num_steps, ...]; each element
            # contains non-overlapping time steps.
            .unbatch())
      return dataset

    dataset = make_reverb_dataset(
        self._server_address,
        self._table_name,
        data_spec=self._data_spec,
        batch_size=sample_batch_size,
        sequence_length=self._sequence_length,
        per_sequence_fn=per_sequence_fn,
        dataset_transformation=dataset_transformation,
        # Try to make it as deterministic as possible.
        cycle_length=1,
        num_parallel_calls=1,
        max_in_flight_samples_per_worker=1,
        num_workers_per_iterator=1,
        rate_limiter_timeout_ms=self._rate_limiter_timeout_ms)

    return dataset

  def gather_all(self):
    """Returns all the items in buffer.

    ***Warning***: `ReverbReplayBuffer` does not support `gather_all`. See
    `reverb_utils.ReverbObserver` for more information on how to retrieve data
    from the buffer.

    Returns:
      Nothing.

    Raises:
      NotImplementedError

    """
    raise NotImplementedError(
        'ReverbReplayBuffer does not support `gather_all`.')

  def _clear(self):
    """Clears the replay buffer."""
    self._py_client.reset(self._table_name)

  def update_priorities(self, keys, priorities):
    """Updates the priorities for the given keys."""
    # TODO(b/144858635): Return ops here and support v1.
    self._tf_client.update_priorities(self._table_name, keys, priorities)

  def _verify_num_steps(self, num_steps):
    if num_steps and self._sequence_length:
      if num_steps > self._sequence_length:
        raise ValueError(
            'Can not guarantee sequential data for num_steps as sequence '
            'length of the data is smaller.  This is not supported.  '
            'num_steps > sequence_length ({} vs. {})'
            .format(num_steps, self._sequence_length))
      if self._sequence_length % num_steps != 0:
        raise ValueError(
            'Can not guarantee sequential data since sequence_length is not a '
            'multiple of num_steps ({} vs. {})'
            .format(num_steps, self._sequence_length))


def make_reverb_dataset(server_address: str,
                        table: str,
                        data_spec,
                        max_in_flight_samples_per_worker=10,
                        batch_size=None,
                        prefetch_size=None,
                        sequence_length=None,
                        cycle_length=None,
                        num_parallel_calls=None,
                        per_sequence_fn=None,
                        dataset_transformation=None,
                        num_workers_per_iterator=-1,
                        max_samples_per_stream=-1,
                        rate_limiter_timeout_ms=-1) -> tf.data.Dataset:
  """Makes a TensorFlow dataset.

  Args:
    server_address: The server address of the replay server.
    table: The name of the table to sample from replay.
    data_spec: The data's spec.
    max_in_flight_samples_per_worker: Optional, dataset buffer capacity.
    batch_size: Optional. If specified the dataset returned will combine
      consecutive elements into batches. This argument is also used to determine
      the cycle_length for `tf.data.Dataset.interleave` -- if unspecified the
      cycle length is set to `tf.data.experimental.AUTOTUNE`.
    prefetch_size: How many batches to prefectch in the pipeline.
    sequence_length: Optional. If specified consecutive elements of each
      interleaved dataset will be combined into sequences.
    cycle_length: Optional. When equal to batch_size it would make take a sample
      from a different sequence. For reducing memory usage use a smaller number.
    num_parallel_calls: Optional. If specified number of parallel calls in
      iterleave. By default use `tf.data.experimental.AUTOTUNE`.
    per_sequence_fn: Optional, per sequence function.
    dataset_transformation: Optional, per dataset interleave transformation.
    num_workers_per_iterator: (Defaults to -1, i.e auto selected) The number
      of worker threads to create per dataset iterator. When the selected
      table uses a FIFO sampler (i.e a queue) then exactly 1 worker must be
      used to avoid races causing invalid ordering of items. For all other
      samplers, this value should be roughly equal to the number of threads
      available on the CPU.
    max_samples_per_stream: (Defaults to -1, i.e auto selected) The maximum
      number of samples to fetch from a stream before a new call is made.
      Keeping this number low ensures that the data is fetched uniformly from
      all server.
    rate_limiter_timeout_ms: Timeout (in milliseconds) to wait on the rate
      limiter when sampling from the table. If `rate_limiter_timeout_ms >= 0`,
      this is the timeout passed to `Table::Sample` describing how long to wait
      for the rate limiter to allow sampling.

  Returns:
    A tf.data.Dataset that streams data from the replay server.
  """
  # Extract the shapes and dtypes from these specs.
  get_dtype = lambda x: tf.as_dtype(x.dtype)
  get_shape = lambda x: (sequence_length,) + x.shape
  shapes = tf.nest.map_structure(get_shape, data_spec)
  dtypes = tf.nest.map_structure(get_dtype, data_spec)
  # TODO(b/144858901): Validate Tableinfo when it's available from reverb.

  def generate_reverb_dataset(_):
    dataset = reverb.TrajectoryDataset(
        server_address,
        table,
        dtypes,
        shapes,
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
        num_workers_per_iterator=num_workers_per_iterator,
        max_samples_per_stream=max_samples_per_stream,
        rate_limiter_timeout_ms=rate_limiter_timeout_ms,
    )

    def broadcast_info(
        info_traj: types.ReverbReplaySample
    ) -> types.ReverbReplaySample:
      # Assumes that the first element of traj is shaped
      # (sequence_length, ...); and we extract this length.
      info, traj = info_traj
      first_elem = tf.nest.flatten(traj)[0]
      length = first_elem.shape[0] or tf.shape(first_elem)[0]
      info = tf.nest.map_structure(lambda t: tf.repeat(t, [length]), info)
      return reverb.ReplaySample(info, traj)

    dataset = dataset.map(broadcast_info)

    if per_sequence_fn:
      dataset = dataset.map(per_sequence_fn)
    if dataset_transformation:
      dataset = dataset_transformation(dataset)
    return dataset

  cycle_length = cycle_length or batch_size or 1
  num_parallel_calls = num_parallel_calls or tf.data.experimental.AUTOTUNE

  if cycle_length == 1:
    dataset = generate_reverb_dataset(0)
  else:
    dataset = tf.data.Dataset.range(cycle_length).interleave(
        generate_reverb_dataset,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls)

  # Allows interleave to retrieve data from the first `reverb.ReplayDataset`
  # available.
  options = tf.data.Options()
  # reverb replay buffers are not considered deterministic for tf.data.
  options.experimental_deterministic = False
  dataset = dataset.with_options(options)

  if batch_size:
    dataset = dataset.batch(batch_size, drop_remainder=True)
  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)
  return dataset


def truncate_reshape_rows_by_num_steps(sample, num_steps):
  """Reshapes tensors in `sample` to have shape `[rows, num_steps, ...]`.

  This function takes a structure `sample` and for each tensor `t`, it truncates
  the tensor's outer dimension to be the highest possible multiple of
  `num_steps`.

  This is done by first calculating `rows = tf.shape(t[0]) // num_steps`, then
  truncating the `tensor` to shape `t_trunc = t[: (rows * num_steps), ...]`.
  For each tensor, it returns `tf.reshape(t_trunc, [rows, num_steps, ...])`.

  Args:
    sample: Nest of tensors.
    num_steps: Python integer.

  Returns:
    A next with tensors reshaped to `[rows, num_steps, ...]`.
  """
  first_elem = tf.nest.flatten(sample)[0]
  static_sequence_length = tf.compat.dimension_value(first_elem.shape[0])
  if static_sequence_length is not None:
    num_rows = static_sequence_length // num_steps
    static_num_rows = num_rows
  else:
    num_rows = tf.shape(first_elem)[0] // num_steps
    static_num_rows = None
  def _truncate_and_reshape(t):
    truncated = t[:(num_rows * num_steps), ...]
    reshaped = tf.reshape(
        truncated,
        tf.concat(([num_rows, num_steps], tf.shape(t)[1:]), axis=0))
    reshaped.set_shape([static_num_rows, num_steps] + t.shape[1:])
    return reshaped
  return tf.nest.map_structure(_truncate_and_reshape, sample)
