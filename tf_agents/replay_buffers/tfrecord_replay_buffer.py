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

"""A replay buffer of nests of Tensors, backed by TFRecords files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading
import time
import uuid

from absl import logging
from six.moves import queue as Queue
import tensorflow as tf
from tf_agents.replay_buffers import replay_buffer
from tf_agents.trajectories import time_step

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.data.util import nest as data_nest  # TF internal
from tensorflow.python.ops.distributions import util as distributions_util  # TF internal
# pylint:enable=g-direct-tensorflow-import


StepType = time_step.StepType

FILE_FORMAT = (
    '{file_prefix}_{experiment_id}_'
    '{YYYY:04d}{MM:02d}{DD:02d}_'
    '{hh:02d}{mm:02d}{ss:02d}_'
    '{hash}.tfrecord')


BufferInfo = collections.namedtuple('BufferInfo', ['ids'])


class _Stop(object):
  pass


class _Flush(object):

  def __init__(self, lock):
    self.lock = lock
    self.condition_var = threading.Condition(lock=lock)


# Signal object for writers to wrap up, flush their queues, and exit.
_STOP = _Stop()

# The buffer size of values to put on a write queue at a time.
# Tuned via benchmarks in tfrecord_replay_buffer_test.
_QUEUE_CHUNK_SIZE = 16


class _RBData(object):
  """Container class for internal storage of TFRecordReplayBuffer.

  This class exists to avoid circular dependencies on the replay buffer.  For
  more details, see comments in TFRecordReplayBuffer.__init__.
  """

  def __init__(self, **params):
    self._params = params

  def __getattr__(self, key):
    return self._params[key]


class TFRecordReplayBuffer(replay_buffer.ReplayBuffer):
  """A replay buffer that stores data in TFRecords.

  The TFRecords files are located on path `{file_prefix}_{experiment_id}_*`,
  and executing the operation returned by `add_batch()` performs a buffered
  write to one or more such tfrecords files.

  The files themselves contain multiple records, one per time step.  **Episodes
  are stored sequentially**, and unless `time_steps_per_file` is set, a file
  will always contain up to `episodes_per_file` episodes stored as sequential
  time steps.  The records in each file will be, in order:

  ```
  episode 0 @ t=0
  episode 0 @ t=1
  ...
  episode 0 @ t=T_0
  episode_1 @ t=0
  ...
  episode_1 @ t=T_1
  ...
  ```

  Each record is a serialized `tf.train.FeatureList` proto.  The length of
  the `FeatureList` matches `len(flatten(data_spec))` and the `i`th feature is
  the encoding of the `i`th entry in the flattened data spec.  Integral and
  bool tensors are converted to `Int64List`, floating point tensors are
  converted to `FloatList` and string tensors are converted to `BytesList`.

  Assuming the data spec is a `Trajectory` with the fields `step_type`,
  `observation`, `action`, `policy_info`, `next_time_step`, ..., then the
  format for a record storing a time-step trajectory object `trajectory`
  will be (here we use the protobuf text form):

  ```
  feature { int64_list { value: trajectory.step_type } }
  feature { ???_list { value: flatten(trajectory.observation)[0] ... } }
  feature { ???_list { value: flatten(trajectory.observation)[1] ... } }
  ...
  feature { ???_list { value: flatten(trajectory.action)[0] ... } }
  feature { ???_list { value: flatten(trajectory.action)[1] ... } }
  ...
  feature { ???_list { value: flatten(trajectory.policy_info)[0] ... } }
  ...
  feature { int64_list { value: trajectory.next_step_type } }
  feature { ???_list { value: flatten(trajectory.reward)[0] ... } }
  feature { ???_list { value: flatten(trajectory.reward)[1] ... } }
  ...
  feature { float_list { value: trajectory.discount } }
  ```

  **NOTE** We do not currently provide backwards compatibility guarantees for
  this internal per-record data representation; we may change it in a future
  version.

  Shapes and original dtypes are lost, and to read these values out correctly,
  the matching data_spec must be provided to this replay buffer when calling
  `as_dataset` in the process reading the data.

  We attempt to ensure that that all writes are flushed before the program
  exits.  However, to be completely sure, call the `flush()` method after
  the final execution of `add_batch()`.  Alternatively, use the replay buffer
  as a context:

  ```python
  with replay_buffer:
    ...
    session.run(add_batch_op)
  ```

  or, in eager mode:
  ```python
  with replay_buffer:
    ...
    replay_buffer.add_batch(...)
  ```
  """

  def __init__(self,
               experiment_id,
               data_spec,
               file_prefix,
               episodes_per_file,
               time_steps_per_file=None,
               record_options=None,
               sampling_dataset_timesteps_per_episode_hint=256,
               dataset_block_keep_prob=1.0,
               dataset_batch_drop_remainder=True,
               seed=None):
    """Creates a TFRecordReplayBuffer.

    This class receives an experiment_id, data_spec, and file_prefix and creates
    a replay buffer backed by TFRecords files.

    Args:
      experiment_id: A python or tensorflow scalar string tensor,
        the experiment id.  Files written via `add_batch` and read via
        `as_dataset` will be restricted to this `experiment_id`.
      data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing
        a single item that can be stored in this buffer.  The spec **must** have
        a field named `step_type` which contains integer entries representing
        `time_step.StepType` values.
      file_prefix: A python string containing the root path for all TFRecords
        files to write and read from.
      episodes_per_file: The number of episodes to write per file.  Note that
        episodes are always stored in contiguous order.
      time_steps_per_file: (Optional) The number of time steps to write per
        file.  If this value is set, episodes may be broken across file
        boundaries.  In this case, **it is unlikely that episodes will be
        reconstructed fully when `num_steps != 1` is requested via `as_dataset`.
        If you need full episodes, do not set this argument when collecting.
      record_options: `options` argument when creating a new `TFRecordWriter`.
        This should be a `TFRecordOptions` object.
      sampling_dataset_timesteps_per_episode_hint: A hint to `as_dataset` with
        the average expected number of frames per episode.  This hint helps
        the dataset to create shuffle buffers of an appropriate size.
      dataset_block_keep_prob: A python or scalar float `Tensor`
        with value in `[0.0, 1.0]`.  When calling `as_dataset` with
        `num_steps is not  None`, overlapping windows of size `num_steps` are
        read from TFRecords files, with window stride 1.  This parameter
        controls how many of these windows are kept.  A value of `1.0` means for
        an episode with 5 frames and `num_steps == 2`, the following blocks are
        emitted- `[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]`.  A value smaller
        than `1.0` means the set of blocks emitted is subsampled by the
        given percentage.
      dataset_batch_drop_remainder: Whether to drop any remaining blocks
        in the final batch of the dataset returned by `as_dataset`.
      seed: (Optional.) Random seed for `as_dataset` sampling operations.

    Raises:
      ValueError: If `data_spec` lacks field `step_type` with integer dtype.
      TypeError: If `record_options` is passed in and is not an instance of
        `tf.io.TFRecordOptions`.
    """
    if (not getattr(data_spec, 'step_type', None)
        or not data_spec.step_type.dtype.is_integer):
      raise ValueError(
          'data_spec must have a field \'step_type\' '
          'with integer dtype, but saw data_spec: %s' % (data_spec,))
    if (record_options is not None
        and not isinstance(record_options, tf.io.TFRecordOptions)):
      raise TypeError(
          'record_options should be a tf.io.TFRecordOptions instance, '
          'but saw: %s' % (record_options,))
    if record_options is not None and record_options.compression_type:
      # TODO(b/128997422): Fix bug when using compression.
      raise NotImplementedError(
          'Support for compressed TFRecords is not implemented')
    super(TFRecordReplayBuffer, self).__init__(data_spec, capacity=None)
    # We use a separate container for all class data to avoid a circular
    # dependency in a py_function in add_batch.  For more details, see add_batch
    # and _create_send_batch_py.
    self._data = _RBData(
        experiment_id=experiment_id,
        file_prefix=file_prefix,
        episodes_per_file=episodes_per_file,
        time_steps_per_file=time_steps_per_file,
        seed=seed,
        record_options=record_options,
        sampling_dataset_timesteps_per_episode_hint=(
            sampling_dataset_timesteps_per_episode_hint),
        dataset_block_keep_prob=dataset_block_keep_prob,
        per_file_shuffle_buffer_size=(
            episodes_per_file *
            sampling_dataset_timesteps_per_episode_hint),
        queue_chunk_size=_QUEUE_CHUNK_SIZE,
        drop_remainder=dataset_batch_drop_remainder,
        lock=threading.RLock(),
        batch_size=None,
        writer_threads=[],  # A list of threading.Thread.
        queues=[],  # A list of Queue, one for each batch entry.
        queue_buffers=[])  # A list of lists, one for each batch entry.

    # Helps flush queues and write to disk when the object is GC'd.
    # This approach is necessary if the class may ever end up in a reference
    # cycle (we assume that any public user-facing class may end up in a
    # reference cycle at one point or another).
    self._cleanup = WriterCleanup(self._data)

  def flush(self):
    """Flushes all queues and writers created by previous add_batch calls."""
    flusher = _Flush(lock=self._data.lock)
    with self._data.lock:
      for queue, buf in zip(self._data.queues, self._data.queue_buffers):
        if buf:
          queue.put(list(buf))
          del buf[:]
        queue.put(flusher)
        flusher.condition_var.wait()

  def __enter__(self):
    pass

  def __exit__(self, type_arg, value_arg, traceback_arg):
    self.flush()
    return False

  def _add_batch(self, items):
    """Adds a batch of single steps for the corresponding episodes IDs.

    Args:
      items: A batch of items to be added to the buffer. Items will have the
        same structure as the data_spec of this class, but the tensors in items
        will have an extra outer dimension `(batch_size, ...)` in addition to
        the corresponding spec in data_spec.

    Returns:
      An op that adds `items` to the replay buffer.

    Raises:
      ValueError: If item shapes are not compatible with their
        corresponding specs.
    """
    def with_check(t, spec):
      converted = tf.convert_to_tensor(t, spec.dtype)
      expected_shape = tf.TensorShape([None]).concatenate(spec.shape)
      if not expected_shape.is_compatible_with(converted.shape):
        raise ValueError(
            'Expected tensor to be compatible with spec shape %s but saw '
            'tensor: %s' % (expected_shape, converted))
      return converted

    items = tf.nest.map_structure(with_check, items, self._data_spec)

    with tf.name_scope('add_batch'):
      flat_items = tf.nest.flatten(items)
      num_items = len(flat_items)
      batch_size = (
          tf.compat.dimension_at_index(flat_items[0].shape, 0)
          or tf.shape(flat_items[0])[0])
      flat_items_features = [_encode_to_feature(x) for x in flat_items]
      batch_first_features = tf.transpose(flat_items_features)
      flat_items_feature_list = tf.io.encode_proto(
          sizes=tf.fill([batch_size, 1], num_items),
          values=[batch_first_features],
          field_names=['feature'],
          message_type='tensorflow.FeatureList')
      # We create a function without any reference to self to avoid a circular
      # dependency on the replay buffer.  For more details, see the docstring of
      # _create_send_batch_py.
      captured_data_send_batch_py = _create_send_batch_py(self._data)
      status, = tf.py_function(
          captured_data_send_batch_py,
          [items.step_type, flat_items_feature_list],
          Tout=[tf.bool])
      status.set_shape(())
      return status

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer.

    The dataset behaves differently depending on if `num_steps` is provided or
    not.  If `num_steps is None`, then entire episodes are sampled uniformly at
    random from the buffer.  If `num_steps is not None`, is an integer, then we
    return batches of subsets of length `num_steps`.

    We attempt to shuffle the entries in the batches as well as possible.  The
    algorithm for this is roughly:

    1. Shuffle all the TFRecord files found with prefix
       "{file_prefix}_{experiment_id}_*".

    2. Read from `sample_batch_size` TFRecord files in parallel.

    If `num_steps is not None`:

    3. For each file, create blocks of size `num_steps` for all records
      in that file with shift 1.  Shuffle these blocks, possibly dropping some
      depending on the value of `dataset_block_keep_prob`, and return them
      as a stream.

    4. Interleave the block streams coming from each file and shuffle the
      results.

    5. Create batches from the shuffled blocks.

    6. Parse the batches to match the shape and dtype of `self._data_spec`.

    If `num_steps is None`:

    3. For each file, read in entire sequences of episodes, parse them to match
       the shape and dtype of `self._data_spec`, and emit the episodes.

    4. Interleave the episodes coming from each file and shuffle the
       results.

    5. Return a stream of individual episodes, which can be combined via
       e.g. `tf.data.Dataset.padded_batch()`, `bucket_by_sequence_length()`,
       or other batching approach.

    **NOTE** If `num_steps is None`, then the class properties
    `dataset_block_keep_prob` and `dataset_batch_drop_remainder` are ignored.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See `as_dataset` documentation.  This
        argument should be `None` iff `num_steps` is not `None`.
      num_steps: (Optional.) Scalar int.  How many contiguous frames to get
        per entry. Default is `None`: return full-length episodes.  This
        argument should be `None` iff `sample_batch_size` is not `None`.
      num_parallel_calls: (Optional.) Number of parallel calls to use in the
        dataset pipeline when interleaving reads from parallel TFRecord files.

    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items sampled uniformly from the buffer.
        - BufferInfo namedtuple, containing the episode ids.

    Raises:
      ValueError: If `sample_batch_size is None` but `num_steps is not None`,
        or if `sample_batch_size is not None` but `num_steps is None`.
      ValueError: If the data spec contains lists that must be converted to
        tuples.
    """
    if num_steps is None:
      if sample_batch_size is not None:
        raise ValueError(
            'When num_steps is None, sample_batch_size must be '
            'None but saw: %s' % (sample_batch_size,))
    else:
      if sample_batch_size is None or sample_batch_size <= 0:
        raise ValueError(
            'When num_steps is not None, sample_batch_size must be '
            'an integer > 0, saw: %s' % (sample_batch_size,))

    # data_tf.nest.flatten does not flatten python lists, tf.nest.flatten does.
    flat_data_spec = tf.nest.flatten(self._data_spec)
    if flat_data_spec != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  {}'.format(
              tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

    filename_seed = distributions_util.gen_new_seed(self._data.seed,
                                                    salt='filename_seed')

    batch_seed = distributions_util.gen_new_seed(self._data.seed,
                                                 salt='batch_seed')

    drop_block_seed = distributions_util.gen_new_seed(self._data.seed,
                                                      salt='drop_block')

    # TODO(b/128998627): Use a different seed for each file by mapping a count
    # with the filename and doing the seed generation in graph mode.
    per_episode_seed = distributions_util.gen_new_seed(self._data.seed,
                                                       salt='per_episode_seed')

    block_keep_prob = self._data.dataset_block_keep_prob
    dropping_blocks = (tf.is_tensor(block_keep_prob) or block_keep_prob != 1.0)
    if dropping_blocks:
      # empty_block_ds is in format (is_real_data=False, empty_data)
      empty_block_ds = tf.data.Dataset.from_tensors(
          (False, tf.fill([num_steps], '')))
      def select_true_or_empty(_):
        # When this returns 0, select the true block.  When this returns 1,
        # select the empty block.
        return tf.cast(
            tf.random.uniform((), seed=drop_block_seed) > block_keep_prob,
            tf.int64)
      true_or_empty_block_selector_ds = (
          tf.data.experimental.Counter().map(select_true_or_empty))

    def list_and_shuffle_files(_):
      filenames = tf.io.matching_files(
          tf.strings.join(
              (self._data.file_prefix, self._data.experiment_id, '*'),
              separator='_'))
      shuffled = tf.random.shuffle(filenames, seed=filename_seed)
      return shuffled

    def parse_blocks_from_record(records):
      """Decode `FeatureList` tensor `records`.

      Args:
        records: `tf.string` tensor of shape either `[]` or `[batch_size]`.

      Outputs:
        A struct matching `self._data_spec` containing tensors.
        If `num_steps is not None`, it contains tensors with shape
        `[batch_size, num_steps, ...]`; otherwise they have shape `[...]`.
      """
      # If `num_steps is None`, then:
      #  records is shaped [].
      #  features is shaped [len(flatten(self._data_spec))].
      # otherwise:
      #  records is shaped [batch_size].
      #  features is shaped [batch_size, len(flatten(self._data_spec))].
      _, features = tf.io.decode_proto(
          bytes=records,
          message_type='tensorflow.FeatureList',
          field_names=['feature'],
          output_types=[tf.string])
      features = features.pop()
      num_features = len(flat_data_spec)
      features = tf.unstack(features, num_features, axis=-1)
      decoded_features = []
      for feature, spec in zip(features, flat_data_spec):
        decoded_feature = _decode_feature(
            feature,
            spec,
            has_outer_dims=num_steps is not None)
        decoded_features.append(decoded_feature)
      return tf.nest.pack_sequence_as(self._data_spec, decoded_features)

    def read_and_block_fixed_length_tfrecord_file(filename):
      """Read records from `filename`, window them into fixed len blocks.

      This function also optionally subsamples and shuffles the blocks.

      Windowed records from filename come as a stream and prior to subsampling
      and shuffling, the stream contains blocks of the form:

         [r0, r1, ..., r_{num_steps - 1}]
         [r1, r2, ..., r_{num_steps}]
         [r2, r3, ..., r_{num_steps + 1}]
         ...

      Args:
        filename: A scalar string `Tensor` with the TFRecord filename.

      Returns:
        A `tf.data.Dataset` instance.
      """
      def drop_or_batch_window(ds):
        if not dropping_blocks:
          return ds.batch(num_steps, drop_remainder=True)
        else:
          # batched_ds is in format (is_real_data=True, true_ds)
          batched_ds = tf.data.Dataset.zip(
              (tf.data.Dataset.from_tensors(True),
               ds.batch(num_steps, drop_remainder=True)))
          return (
              tf.data.experimental.choose_from_datasets(
                  (batched_ds, empty_block_ds),
                  true_or_empty_block_selector_ds)
              .take(1)
              .filter(lambda is_real_data, _: is_real_data)
              .map(lambda _, true_block: true_block))
      return (
          tf.data.TFRecordDataset(
              filename,
              compression_type=_compression_type_string(
                  self._data.record_options))
          .window(num_steps, shift=1, stride=1, drop_remainder=True)
          .flat_map(drop_or_batch_window)
          .shuffle(buffer_size=self._data.per_file_shuffle_buffer_size,
                   seed=per_episode_seed))

    def read_and_block_variable_length_tfrecord_file(filename):
      """Read records from `filename`, window them into variable len blocks."""
      def create_ta(spec):
        return tf.TensorArray(
            size=0, dynamic_size=True, element_shape=spec.shape,
            dtype=spec.dtype)
      empty_tas = tf.nest.map_structure(create_ta, self._data_spec)

      def parse_and_block_on_episode_boundaries(partial_tas, record):
        frame = parse_blocks_from_record(record)
        updated_tas = tf.nest.map_structure(
            lambda ta, f: ta.write(ta.size(), f),
            partial_tas, frame)
        # If we see a LAST field, then emit empty TAs for the state and updated
        # TAs for the output.  Otherwise emit updated TAs for the state and
        # empty TAs for the output (the empty output TAs will be filtered).
        return tf.cond(
            tf.equal(frame.step_type, StepType.LAST),
            lambda: (empty_tas, updated_tas),
            lambda: (updated_tas, empty_tas))

      stack_tas = lambda tas: tf.nest.map_structure(lambda ta: ta.stack(), tas)
      remove_intermediate_arrays = lambda tas: tas.step_type.size() > 0

      return (
          tf.data.TFRecordDataset(
              filename,
              compression_type=_compression_type_string(
                  self._data.record_options))
          .apply(
              tf.data.experimental.scan(empty_tas,
                                        parse_and_block_on_episode_boundaries))
          .filter(remove_intermediate_arrays)
          .map(stack_tas)
          .shuffle(buffer_size=self._data.per_file_shuffle_buffer_size,
                   seed=per_episode_seed))

    interleave_shuffle_buffer_size = (
        (num_parallel_calls or sample_batch_size or 4)
        * self._data.sampling_dataset_timesteps_per_episode_hint)

    if num_steps is None:
      read_and_block_fn = read_and_block_variable_length_tfrecord_file
    else:
      read_and_block_fn = read_and_block_fixed_length_tfrecord_file

    # Use tf.data.Dataset.from_tensors(0).map(...) to call the map() code once
    # per initialization.  This means that when the iterator is reinitialized,
    # we get a new list of files.
    ds = (tf.data.Dataset.from_tensors(0)
          .map(list_and_shuffle_files)
          .flat_map(tf.data.Dataset.from_tensor_slices)
          # Interleave between blocks of records from different files.
          .interleave(read_and_block_fn,
                      cycle_length=max(
                          num_parallel_calls or sample_batch_size or 0,
                          4),
                      block_length=1,
                      num_parallel_calls=(num_parallel_calls
                                          or tf.data.experimental.AUTOTUNE))
          .shuffle(
              buffer_size=interleave_shuffle_buffer_size,
              seed=batch_seed))

    # Batch and parse the blocks.  If `num_steps is None`, parsing has already
    # happened and we're not batching.
    if num_steps is not None:
      ds = (ds.batch(batch_size=sample_batch_size,
                     drop_remainder=self._data.drop_remainder)
            .map(parse_blocks_from_record))

    return ds

  def _clear(self, clear_all_variables=False):
    """Clears the replay buffer.

    Args:
      clear_all_variables: Boolean to indicate whether to clear all variables or
      just the data table and episode lengths (i.e. keep the current episode ids
      that are in flight in the buffer).
    Returns:
      An op to clear the buffer.
    """
    raise NotImplementedError(
        'Unable to clear this type of replay buffer.  To get a clean slate, '
        'make your experiment_id argument a tf.Variable and modify its value '
        'via .assign() when you want to swithc to a new experiment scope.')


def _generate_filename(file_prefix, experiment_id):
  now = time.gmtime()
  return FILE_FORMAT.format(
      file_prefix=file_prefix,
      experiment_id=experiment_id,
      YYYY=now.tm_year,
      MM=now.tm_mon,
      DD=now.tm_mday,
      hh=now.tm_hour,
      mm=now.tm_min,
      ss=now.tm_sec,
      hash=uuid.uuid1())


def _create_send_batch_py(rb_data):
  """Return a function to send data to writer queues.

  This functor returns a function

    send_batch_py(step_type, flat_serialized) -> True

  which (possibly) initializes and sends data to writer queues pointed to
  by `rb_data`.

  This function takes an `_RBData` instance instead of being a method of
  `TFRecordReplayBuffer` because we want to avoid an extra dependency
  on the replay buffer object from within TensorFlow's py_function.  This
  circular dependency occurs because of a cell capture from the TF runtime
  on `self` if `self` is ever used within a py_function.

  Args:
    rb_data: An instance of `_RBData`.

  Returns:
    A function `send_batch_py(step_type, flat_serialized) -> True`.
  """
  def send_batch_py(step_type, flat_serialized):
    """py_function that sends data to a writer thread."""
    # NOTE(ebrevdo): Here we have a closure over ONLY rb._data, not over rb
    # itself.  This avoids the circular dependency.
    step_type = step_type.numpy()
    flat_serialized = flat_serialized.numpy()
    batch_size = step_type.shape[0]
    _maybe_initialize_writers(batch_size, rb_data)
    for batch in range(batch_size):
      queue_buffer = rb_data.queue_buffers[batch]
      queue_buffer.append(
          (rb_data.experiment_id, step_type[batch], flat_serialized[batch]))
      if len(queue_buffer) >= rb_data.queue_chunk_size:
        rb_data.queues[batch].put(list(queue_buffer))
        del queue_buffer[:]
    return True
  return send_batch_py


def _maybe_initialize_writers(batch_size, rb_data):
  """Initialize the queues and threads in `rb_data` given `batch_size`."""
  with rb_data.lock:
    if rb_data.batch_size is None:
      rb_data.batch_size = batch_size
      if batch_size > 64:
        logging.warning(
            'Using a batch size = %d > 64 when writing to the '
            'TFRecordReplayBuffer, which can cause python thread contention '
            'and impact performance.')
    if batch_size != rb_data.batch_size:
      raise ValueError(
          'Batch size does not match previous batch size: %d vs. %d'
          % (batch_size, rb_data.batch_size))
    if not rb_data.writer_threads:
      rb_data.queue_buffers.extend([list() for _ in range(batch_size)])
      rb_data.queues.extend([Queue.Queue() for _ in range(batch_size)])
      # pylint: disable=g-complex-comprehension
      rb_data.writer_threads.extend([
          threading.Thread(
              target=_process_write_queue,
              name='process_write_queue_%d' % i,
              kwargs={
                  'queue': rb_data.queues[i],
                  'episodes_per_file': rb_data.episodes_per_file,
                  'time_steps_per_file': rb_data.time_steps_per_file,
                  'file_prefix': rb_data.file_prefix,
                  'record_options': rb_data.record_options
              })
          for i in range(batch_size)
      ])
      # pylint: enable=g-complex-comprehension
      for thread in rb_data.writer_threads:
        thread.start()


def _process_write_queue(queue, episodes_per_file, time_steps_per_file,
                         file_prefix, record_options):
  """Process running in a separate thread that writes the TFRecord files."""
  writer = None
  num_steps = 0
  num_episodes = 0

  while True:
    item = queue.get()
    if item is _STOP:
      if writer is not None:
        try:
          writer.close()
        except tf.errors.OpError as e:
          logging.error(str(e))
      return
    elif isinstance(item, _Flush):
      if writer is not None:
        try:
          writer.flush()
        except tf.errors.OpError as e:
          logging.error(str(e))
      with item.lock:
        item.condition_var.notify()
      continue

    for experiment_id, step_type, serialized_feature_list in item:
      num_steps += 1
      if step_type == StepType.FIRST:
        num_episodes += 1
      if (writer is None
          or (step_type == StepType.FIRST
              and num_episodes >= episodes_per_file)
          or (time_steps_per_file is not None
              and num_steps >= time_steps_per_file)):
        filename = _generate_filename(
            file_prefix=file_prefix, experiment_id=experiment_id)
        if writer is None:
          try:
            tf.io.gfile.makedirs(filename[:filename.rfind('/')])
          except tf.errors.OpError as e:
            logging.error(str(e))
        else:
          try:
            writer.close()
          except (AttributeError, tf.errors.OpError) as e:
            logging.error(str(e))
        num_episodes = 0
        num_steps = 0
        try:
          writer = tf.io.TFRecordWriter(filename, record_options)
        except tf.errors.OpError as e:
          logging.error(str(e))

      try:
        writer.write(serialized_feature_list)
      except (tf.errors.OpError, AttributeError) as e:
        logging.error(str(e))


def _encode_to_feature(t):
  """Encodes batched tensor `t` to a batched `tensorflow.Feature` tensor string.

  - Integer tensors are encoded as `int64_list`.
  - Floating point tensors are encoded as `float_list`.
  - String tensors are encoded as `bytes_list`.

  Args:
    t: A `tf.Tensor` shaped `[batch_size, ...]`.

  Returns:
    A string `tf.Tensor` with shape `[batch_size]` containing serialized
    `tensorflow.Feature` proto.

  Raises:
    NotImplementedError: If `t.dtype` is not a supported encoding dtype.
  """
  batch_size = tf.compat.dimension_at_index(t.shape, 0) or tf.shape(t)[0]
  t = tf.reshape(t, [batch_size, -1])
  num_elements_per_batch = (
      tf.compat.dimension_at_index(t.shape, 1) or tf.shape(t)[1])
  # TODO(b/129368627): Revisit writing everything as a BytesList once we have
  # support for getting string-like views of dense tensors.
  if t.dtype.is_integer or t.dtype.base_dtype == tf.bool:
    field_name = 'int64_list'
    message_type = 'tensorflow.Int64List'
    t = tf.cast(t, tf.int64)
  elif t.dtype.is_floating:
    field_name = 'float_list'
    message_type = 'tensorflow.FloatList'
    t = tf.cast(t, tf.float32)  # We may lose precision here
  elif t.dtype.base_dtype == tf.string:
    field_name = 'bytes_list'
    message_type = 'tensorflow.BytesList'
  else:
    raise NotImplementedError('Encoding tensor type %d is is not supported.'
                              % t.dtype)
  batch_size = tf.shape(t)[0]
  values_list = tf.io.encode_proto(
      sizes=tf.fill([batch_size, 1], num_elements_per_batch),
      values=[t],
      field_names=['value'],
      message_type=message_type)
  return tf.io.encode_proto(
      sizes=tf.fill([batch_size, 1], 1),
      values=[tf.expand_dims(values_list, 1)],
      field_names=[field_name],
      message_type='tensorflow.Feature')


def _decode_feature(feature, spec, has_outer_dims):
  """Decodes batched serialized `tensorflow.Feature` to a batched `spec` tensor.

  - Integer tensors are decoded from `int64_list`.
  - Floating point tensors are decoded from `float_list`.
  - String tensors are decoded `bytes_list`.

  Args:
    feature: A `tf.Tensor` of type string shaped `[batch_size, num_steps]`.
    spec: A `tf.TensorSpec`.
    has_outer_dims: Python bool, whether the feature has a batch dim or not.

  Returns:
    Returns `tf.Tensor` with shape `[batch_size, num_steps] + spec.shape`
    having type `spec.dtype`.  If `not has_outer_dims`, then the tensor has no
    `[batch_size, num_steps]` prefix.

  Raises:
    NotImplementedError: If `spec.dtype` is not a supported decoding dtype.
  """
  if has_outer_dims:
    feature.shape.assert_has_rank(2)
  else:
    feature.shape.assert_has_rank(0)

  # This function assumes features come in as encoded tensorflow.Feature strings
  # with shape [batch, num_steps].
  if spec.dtype.is_integer or spec.dtype.base_dtype == tf.bool:
    field_name = 'int64_list'
    message_type = 'tensorflow.Int64List'
    feature_dtype = tf.int64
  elif spec.dtype.is_floating:
    field_name = 'float_list'
    message_type = 'tensorflow.FloatList'
    feature_dtype = tf.float32
  elif spec.dtype.base_dtype == tf.string:
    field_name = 'bytes_list'
    message_type = 'tensorflow.BytesList'
    feature_dtype = tf.string
  else:
    raise NotImplementedError('Decoding spec type %d is is not supported.'
                              % spec.dtype)

  _, value_message = tf.io.decode_proto(
      bytes=feature,
      message_type='tensorflow.Feature',
      field_names=[field_name],
      output_types=[tf.string])
  value_message = value_message.pop()
  value_message = tf.squeeze(value_message, axis=-1)

  _, values = tf.io.decode_proto(
      bytes=value_message,
      message_type=message_type,
      field_names=['value'],
      output_types=[feature_dtype])
  values = values.pop()
  values_shape = tf.shape(values)
  if has_outer_dims:
    batch_size = tf.compat.dimension_value(values.shape[0]) or values_shape[0]
    num_steps = tf.compat.dimension_value(values.shape[1]) or values_shape[1]
    values = tf.reshape(
        values, [batch_size, num_steps] + spec.shape.as_list())
  else:
    values = tf.reshape(values, spec.shape.as_list())
  return tf.cast(values, spec.dtype)


class WriterCleanup(object):
  """WriterCleanup class.

  When garbage collected, this class flushes all queue buffers and writer
  threads.
  """

  def __init__(self, rb_data):
    self._data = rb_data

  def __del__(self):
    # NOTE(ebrevdo): This may spew a lot of noise at program shutdown due to
    # random module unloading.  If it does, it's safe to ignore but we might
    # consider wrapping it inside a broad try: ... except:, to avoid the noise.
    if not self._data.writer_threads:
      return
    for queue, buf in zip(self._data.queues, self._data.queue_buffers):
      if buf:
        queue.put(list(buf))
        del buf[:]
      queue.put(_STOP)
    for t in self._data.writer_threads:
      t.join()


def _compression_type_string(record_options):
  if record_options is None:
    return None
  assert isinstance(record_options, tf.io.TFRecordOptions), type(record_options)
  return record_options.compression_type
