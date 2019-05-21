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

"""Tests for tfrecord_replay_buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time
import uuid
import weakref

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tfrecord_replay_buffer
from tf_agents.trajectories import time_step

StepType = time_step.StepType

Data = collections.namedtuple('Data', ('step_type', 'value'))


# Pulls out the 'value' from Data encoded in a FeatureList, and reads the first
# (floating point) entry.
_key_by_first_value = lambda episode: episode[0].feature[1].float_list.value[0]


def read_feature_lists(fn):
  feature_lists = []
  for record in tf.compat.v1.io.tf_record_iterator(fn):
    feature_list = tf.train.FeatureList.FromString(record)
    feature_lists.append(feature_list)
  return feature_lists


def create_feature_list(*value_lists):
  feature_list = tf.train.FeatureList()
  for values in value_lists:
    values = np.array(values)
    if values.dtype.kind == 'i':
      feature_list.feature.add(
          int64_list=tf.train.Int64List(value=values.astype(np.int64)))
    elif values.dtype.kind == 'f':
      feature_list.feature.add(
          float_list=tf.train.FloatList(value=values.astype(np.float32)))
    else:
      feature_list.feature.add(
          bytes_list=tf.train.BytesList(value=values.astype(np.bytes_)))
  return feature_list


class TFRecordReplayBufferTest(tf.test.TestCase, parameterized.TestCase):

  _simple_data_spec = Data(step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
                           value=tf.TensorSpec(shape=(2,), dtype=tf.float64))

  def setUp(self):
    super(TFRecordReplayBufferTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._tempdir = self.create_tempdir()
    self._file_prefix = '%s/test-%s/' % (self._tempdir.full_path, uuid.uuid4())

  def tearDown(self):
    super(TFRecordReplayBufferTest, self).tearDown()
    del self._tempdir

  def compareEpisode(self, expected, received):
    self.assertLen(expected, len(received))
    for expected_frame, received_frame in zip(expected, received):
      self.assertProtoEquals(expected_frame, received_frame)

  def testCreateAndDoNothing(self):
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        data_spec=self._simple_data_spec)
    rb = weakref.ref(rb)  # RB should be garbage collected
    try:
      self.assertFalse(tf.io.gfile.glob(self._file_prefix + '*'))
    except tf.errors.NotFoundError:  # Some fs impls raise this error.
      pass
    self.assertFalse(rb())

  def testFailsWithDifferingBatchSize(self):
    data_0 = Data(
        step_type=tf.constant([StepType.FIRST]),
        value=tf.constant([[1.0, -1.0]], dtype=tf.float64))
    data_1 = Data(
        step_type=tf.constant([StepType.MID, StepType.LAST]),
        value=tf.constant([[3.0, -3.0], [4.0, -4.0]], dtype=tf.float64))
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        data_spec=self._simple_data_spec)
    self.evaluate(rb.add_batch(data_0))
    with self.assertRaisesOpError(
        r'Batch size does not match previous batch size: 2 vs. 1'):
      self.evaluate(rb.add_batch(data_1))

  def testAddBatch(self):
    data_0 = Data(
        step_type=tf.constant([StepType.FIRST, StepType.MID]),
        value=tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float64))
    data_1 = Data(
        step_type=tf.constant([StepType.MID, StepType.LAST]),
        value=tf.constant([[3.0, -3.0], [4.0, -4.0]], dtype=tf.float64))
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        data_spec=self._simple_data_spec)
    self.evaluate(rb.add_batch(data_0))
    self.evaluate(rb.add_batch(data_1))
    rb = weakref.ref(rb)  # RB should be garbage collected
    files = tf.io.gfile.glob(self._file_prefix + '*')
    self.assertLen(files, 2)
    self.assertFalse(rb())

    episode_0, episode_1 = sorted(
        [read_feature_lists(fn) for fn in files],
        key=_key_by_first_value)

    expected_episode_0 = [
        create_feature_list([StepType.FIRST], [1.0, -1.0]),
        create_feature_list([StepType.MID], [3.0, -3.0])
    ]
    expected_episode_1 = [
        create_feature_list([StepType.MID], [2.0, -2.0]),
        create_feature_list([StepType.LAST], [4.0, -4.0])
    ]
    self.compareEpisode(expected_episode_0, episode_0)
    self.compareEpisode(expected_episode_1, episode_1)

  def testAsContextManager(self):
    data_0 = Data(
        step_type=tf.constant([StepType.FIRST, StepType.MID]),
        value=tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float64))
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        data_spec=self._simple_data_spec)
    with rb:
      self.evaluate(rb.add_batch(data_0))
    files = tf.io.gfile.glob(self._file_prefix + '*')
    self.assertLen(files, 2)

  def testAddBatchTwiceWithNewEpisode(self):
    data_0 = Data(
        step_type=tf.constant([StepType.FIRST, StepType.MID]),
        value=tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float64))
    data_1 = Data(
        step_type=tf.constant([StepType.MID, StepType.FIRST]),
        value=tf.constant([[3.0, -3.0], [4.0, -4.0]], dtype=tf.float64))
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=1,
        data_spec=self._simple_data_spec)
    self.evaluate(rb.add_batch(data_0))
    self.evaluate(rb.add_batch(data_1))
    rb = weakref.ref(rb)  # RB should be garbage collected

    # There should be exactly 3 files because we force 1 episode per file.
    files = tf.io.gfile.glob(self._file_prefix + '*')
    self.assertLen(files, 3)

    self.assertFalse(rb())

    episode_0, episode_1, episode_2 = sorted(
        [read_feature_lists(fn) for fn in files],
        key=_key_by_first_value)

    expected_episode_0 = [
        create_feature_list([StepType.FIRST], [1.0, -1.0]),
        create_feature_list([StepType.MID], [3.0, -3.0]),
    ]
    expected_episode_1 = [
        create_feature_list([StepType.MID], [2.0, -2.0])
    ]
    expected_episode_2 = [
        create_feature_list([StepType.FIRST], [4.0, -4.0])
    ]
    self.compareEpisode(expected_episode_0, episode_0)
    self.compareEpisode(expected_episode_1, episode_1)
    self.compareEpisode(expected_episode_2, episode_2)

  def testAddBatchTwiceWithNewFrameLimitPerFile(self):
    data_0 = Data(
        step_type=tf.constant([StepType.FIRST, StepType.MID]),
        value=tf.constant([[1.0, -1.0], [2.0, -2.0]], dtype=tf.float64))
    data_1 = Data(
        step_type=tf.constant([StepType.MID, StepType.LAST]),
        value=tf.constant([[3.0, -3.0], [4.0, -4.0]], dtype=tf.float64))
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        time_steps_per_file=1,
        data_spec=self._simple_data_spec)
    self.evaluate(rb.add_batch(data_0))
    self.evaluate(rb.add_batch(data_1))
    del rb

    # There should be exactly 3 files because we force 1 step per file.
    files = tf.io.gfile.glob(self._file_prefix + '*')
    self.assertLen(files, 4)

    episode_0, episode_1, episode_2, episode_3 = sorted(
        [read_feature_lists(fn) for fn in files],
        key=_key_by_first_value)

    expected_episode_0 = [create_feature_list([StepType.FIRST], [1.0, -1.0])]
    expected_episode_1 = [create_feature_list([StepType.MID], [2.0, -2.0])]
    expected_episode_2 = [create_feature_list([StepType.MID], [3.0, -3.0])]
    expected_episode_3 = [create_feature_list([StepType.LAST], [4.0, -4.0])]

    self.compareEpisode(expected_episode_0, episode_0)
    self.compareEpisode(expected_episode_1, episode_1)
    self.compareEpisode(expected_episode_2, episode_2)
    self.compareEpisode(expected_episode_3, episode_3)

  def testAsDatasetFromOneFile(self):
    episode_values = [
        ([StepType.FIRST], [1.0, -1.0]),
        ([StepType.MID], [2.0, -2.0]),
        ([StepType.MID], [3.0, -3.0]),
        ([StepType.FIRST], [4.0, -4.0]),
        ([StepType.LAST], [5.0, -5.0])
    ]
    episode = [create_feature_list(*value) for value in episode_values]
    # Maps e.g. 1 => ([StepType.FIRST], [1.0, -1.0])
    #           2 => ([StepType.MID], [2.0, -2.0])
    episode_map = dict((int(x[1][0]), x) for x in episode_values)
    tf.io.gfile.makedirs(self._file_prefix[:self._file_prefix.rfind('/')])
    with tf.io.TFRecordWriter(self._file_prefix + '_exp_0') as wr:
      for step in episode:
        wr.write(step.SerializeToString())

    self._evaluate_written_records(episode_map, num_episodes=1)

  @parameterized.parameters(
      {'keep_prob': 0.0},
      {'keep_prob': 0.25},
      {'keep_prob': 0.5},
      {'keep_prob': 0.9999})
  def testAsDatasetBlockKeepProb(self, keep_prob):
    episode_values = [
        ([StepType.FIRST], [1.0, -1.0]),
        ([StepType.MID], [2.0, -2.0]),
        ([StepType.MID], [3.0, -3.0]),
        ([StepType.MID], [4.0, -4.0]),
        ([StepType.LAST], [5.0, -5.0])
    ]
    episode = [create_feature_list(*value) for value in episode_values]
    tf.io.gfile.makedirs(self._file_prefix[:self._file_prefix.rfind('/')])
    with tf.io.TFRecordWriter(self._file_prefix + '_exp_0') as wr:
      for step in episode:
        wr.write(step.SerializeToString())

    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        seed=12345,
        dataset_block_keep_prob=keep_prob,
        data_spec=self._simple_data_spec)

    batch_size = 1
    num_steps = 2
    ds = rb.as_dataset(sample_batch_size=batch_size, num_steps=num_steps)
    # Get enough samples to form statistically significant counts
    ds = ds.repeat(1000)
    evaluate_gn = self.get_iterator_callback(ds)
    frames = []
    while True:
      try:
        frames.append(evaluate_gn())
      except (tf.errors.OutOfRangeError, StopIteration):
        break

    # The total number of windows per file is 4:
    #   [[1.0, -1.0], [2.0, -2.0]]
    #   [[2.0, -2.0], [3.0, -3.0]]
    #   [[3.0, -3.0], [4.0, -4.0]]
    #   [[4.0, -4.0], [5.0, -5.0]]
    #
    # We ask for 1000 copies of full reads from the file.  If we read 1000
    # files, that 4 * 1000 = 4000 records total.  The windows will come in
    #   shuffled.
    if keep_prob == 0.0:
      self.assertEmpty(frames)
    else:
      self.assertNear(
          1.0 * len(frames) / 4000, keep_prob, err=0.025)
      # Pull out the first values from block tensors, e.g. if the block value is
      #   [[1.0, -1.0], [2.0, -2.0]],
      # then the first value is `1.0`.
      first_values = np.asarray([x.value[0, 0, 0] for x in frames])
      for allowed in (1.0, 2.0, 3.0, 4.0):
        self.assertNear(1.0 * np.sum(first_values == allowed) / len(frames),
                        0.25,
                        err=0.025,
                        msg=('Expected to see value %g about 1/4 of the time, '
                             'but saw %d such occurences of %d frames total.'
                             % (allowed, np.sum(first_values == allowed),
                                len(frames))))

  def testAsDatasetFrom10Files(self):
    episode_map = {}
    for i in range(10):
      c = 10 * i
      episode_values = [
          ([StepType.FIRST], [c + 1.0, -c - 1.0]),
          ([StepType.MID], [c + 2.0, -c - 2.0]),
          ([StepType.MID], [c + 3.0, -c - 3.0]),
          ([StepType.FIRST], [c + 4.0, -c - 4.0]),
          ([StepType.LAST], [c + 5.0, -c - 5.0])
      ]

      episode = [create_feature_list(*value) for value in episode_values]

      # Maps e.g. 1 => ([StepType.FIRST], [1.0, -1.0])
      #           2 => ([StepType.MID], [2.0, -2.0])
      episode_map.update(
          dict((int(x[1][0]), x) for x in episode_values))

      tf.io.gfile.makedirs(self._file_prefix[:self._file_prefix.rfind('/')])
      with tf.io.TFRecordWriter(self._file_prefix + '_exp_%d' % i) as wr:
        for step in episode:
          wr.write(step.SerializeToString())

    self._evaluate_written_records(episode_map, num_episodes=10)

  def get_iterator_callback(self, ds):
    if tf.executing_eagerly():
      it = iter(ds)
      evaluate_gn = lambda: tf.nest.map_structure(lambda t: t.numpy(), next(it))
    else:
      it = tf.compat.v1.data.make_initializable_iterator(ds)
      self.evaluate(it.initializer)
      gn = it.get_next()
      evaluate_gn = lambda: self.evaluate(gn)
    return evaluate_gn

  def _evaluate_written_records(self, episode_map, num_episodes):
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=10,
        seed=12345,
        data_spec=self._simple_data_spec)
    batch_size = 2
    num_steps = 3
    ds = rb.as_dataset(sample_batch_size=batch_size, num_steps=num_steps)
    ds = ds.repeat()  # Repeat forever to get a good statistical sample.
    evaluate_gn = self.get_iterator_callback(ds)

    def check_shape_dtype(val, spec):
      self.assertEqual(val.dtype, spec.dtype.as_numpy_dtype)
      self.assertEqual(
          val.shape,
          (batch_size, num_steps) + tuple(spec.shape.as_list()))

    starting_time_step_counter = collections.defaultdict(lambda: 0)
    num_trials = 512
    for _ in range(num_trials):
      gn_value = evaluate_gn()
      tf.nest.map_structure(check_shape_dtype, gn_value, self._simple_data_spec)

      # Flatten the batched gn_values, then unstack each component of Data()
      # individually, and group the results into a batch-size list of Data().
      flat_gn_value = tf.nest.flatten(gn_value)
      squeezed = []
      for y in flat_gn_value:
        squeezed.append([np.squeeze(x, 0) for x in np.split(y, batch_size)])
      for batch_item in zip(*squeezed):
        # batch_item is now a Data() containing one batch entry.
        batch_item = tf.nest.pack_sequence_as(gn_value, batch_item)

        # Identify the frame associated with each of num_steps' value[0]
        which_frame = batch_item.value[:, 0].squeeze().astype(np.int32)

        # Add the first frame's value[0] to a counter for later statistical
        # testing of evenness of sampling.
        starting_time_step_counter[which_frame[0]] += 1

        # Ensure frames are increasing in order (since value[0]s are in
        # increasing order)
        self.assertAllEqual(
            np.diff(which_frame),
            [1] * (num_steps - 1))

        # Ensure values are correct in the form float([x, -x])
        self.assertAllEqual(
            batch_item.value,
            np.vstack((which_frame, -which_frame)).T)

        # Ensure step_type is the correct step_type matching this starting
        # frame.
        self.assertAllEqual(
            batch_item.step_type,
            # Look up the step type from episode_values.
            [episode_map[x][0][0] for x in which_frame])

    # blocks start with value 1, 2, 3 (in multi-episode case, also 11, 12, 13,
    # 21, 22, 23, ...)
    self.assertLen(starting_time_step_counter, 3 * num_episodes)
    for k in range(num_episodes):
      self.assertIn(10 * k + 1, starting_time_step_counter)
      self.assertIn(10 * k + 2, starting_time_step_counter)
      self.assertIn(10 * k + 3, starting_time_step_counter)

    num_time_step_init = len(starting_time_step_counter)
    for k in starting_time_step_counter.keys():
      # Check that the initial start value is represented relatively evenly.
      self.assertGreater(starting_time_step_counter[k],
                         num_trials / (num_time_step_init + 1))

  # TODO(b/128997422): Fix bug when using compression.
  @parameterized.parameters(
      {'compression': None},
      {'compression': ''},
      # {'compression': 'GZIP'},
      # {'compression': 'ZLIB'}
  )
  def testAddBatchAndAsDatasetFixedNumSteps(self, compression):
    data = [
        Data(
            step_type=tf.constant([StepType.FIRST,
                                   StepType.MID,
                                   StepType.MID]),
            value=tf.constant([[1.0, -1.0],
                               [4.0, -4.0],
                               [7.0, -7.0]],
                              dtype=tf.float64)),
        Data(
            step_type=tf.constant([StepType.MID,
                                   StepType.MID,
                                   StepType.LAST]),
            value=tf.constant([[2.0, -2.0],
                               [5.0, -5.0],
                               [8.0, -8.0]],
                              dtype=tf.float64)),
        Data(
            step_type=tf.constant([StepType.MID,
                                   StepType.LAST,
                                   StepType.FIRST]),
            value=tf.constant([[3.0, -3.0],
                               [6.0, -6.0],
                               [9.0, -9.0]],
                              dtype=tf.float64))]
    if compression is None:
      record_options = None
    else:
      record_options = tf.io.TFRecordOptions(compression_type=compression)
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=1,
        data_spec=self._simple_data_spec,
        record_options=record_options,
        dataset_batch_drop_remainder=False)
    with rb:
      for data_i in data:
        self.evaluate(rb.add_batch(data_i))
    ds = rb.as_dataset(sample_batch_size=2, num_steps=2)
    evaluate_gn = self.get_iterator_callback(ds)
    batches = []
    while True:
      try:
        batches.append(evaluate_gn())
      except (tf.errors.OutOfRangeError, StopIteration):
        break

    expected_blocks = [
        Data(step_type=[StepType.FIRST, StepType.MID],
             value=[[1.0, -1.0], [2.0, -2.0]]),
        Data(step_type=[StepType.MID, StepType.MID],
             value=[[2.0, -2.0], [3.0, -3.0]]),
        Data(step_type=[StepType.MID, StepType.MID],
             value=[[4.0, -4.0], [5.0, -5.0]]),
        Data(step_type=[StepType.MID, StepType.LAST],
             value=[[5.0, -5.0], [6.0, -6.0]]),
        Data(step_type=[StepType.MID, StepType.LAST],
             value=[[7.0, -7.0], [8.0, -8.0]]),
    ]

    blocks = []
    for b in batches:
      for i in range(len(b.step_type)):
        blocks.append(Data(step_type=b.step_type[i].tolist(),
                           value=b.value[i].tolist()))

    self.assertLen(blocks, len(expected_blocks))
    for expected in expected_blocks:
      self.assertIn(expected, blocks)

  # TODO(b/128997422): Fix bug when using compression.
  @parameterized.parameters(
      {'compression': None},
      {'compression': ''},
      # {'compression': 'GZIP'},
      # {'compression': 'ZLIB'}
  )
  def testAddBatchAndAsDatasetVarLenNumSteps(self, compression):
    data = [
        Data(
            step_type=tf.constant([StepType.FIRST,
                                   StepType.MID,
                                   StepType.MID]),
            value=tf.constant([[1.0, -1.0],
                               [4.0, -4.0],
                               [7.0, -7.0]],
                              dtype=tf.float64)),
        Data(
            step_type=tf.constant([StepType.LAST,
                                   StepType.MID,
                                   StepType.LAST]),
            value=tf.constant([[2.0, -2.0],
                               [5.0, -5.0],
                               [8.0, -8.0]],
                              dtype=tf.float64)),
        Data(
            step_type=tf.constant([StepType.FIRST,
                                   StepType.LAST,
                                   StepType.FIRST]),
            value=tf.constant([[3.0, -3.0],
                               [6.0, -6.0],
                               [9.0, -9.0]],
                              dtype=tf.float64)),
        Data(
            step_type=tf.constant([StepType.LAST,
                                   StepType.MID,
                                   StepType.MID]),
            value=tf.constant([[10.0, -10.0],
                               [11.0, -11.0],
                               [12.0, -12.0]],
                              dtype=tf.float64))
    ]

    if compression is None:
      record_options = None
    else:
      record_options = tf.io.TFRecordOptions(compression_type=compression)
    rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
        experiment_id='exp',
        file_prefix=self._file_prefix,
        episodes_per_file=1,
        data_spec=self._simple_data_spec,
        record_options=record_options,
        dataset_batch_drop_remainder=False)
    with rb:
      for data_i in data:
        self.evaluate(rb.add_batch(data_i))
    ds = rb.as_dataset(num_steps=None)
    evaluate_gn = self.get_iterator_callback(ds)
    blocks = []
    while True:
      try:
        blocks.append(
            tf.nest.map_structure(lambda arr: arr.tolist(),
                                  evaluate_gn()))
      except (tf.errors.OutOfRangeError, StopIteration):
        break

    # We'll only see blocks that had a StepType.LAST at the end.
    expected_blocks = [
        Data(step_type=[StepType.FIRST, StepType.LAST],
             value=[[1.0, -1.0], [2.0, -2.0]]),
        Data(step_type=[StepType.FIRST, StepType.LAST],
             value=[[3.0, -3.0], [10.0, -10.0]]),
        Data(step_type=[StepType.MID, StepType.MID, StepType.LAST],
             value=[[4.0, -4.0], [5.0, -5.0], [6.0, -6.0]]),
        Data(step_type=[StepType.MID, StepType.LAST],
             value=[[7.0, -7.0], [8.0, -8.0]]),
    ]

    self.assertLen(blocks, len(expected_blocks))
    for expected in expected_blocks:
      self.assertIn(expected, blocks)


class TFRecordReplayBufferBenchmark(tf.test.Benchmark):

  def __init__(self, testing=False):
    self._testing = testing

  def benchmark_write_performance(self):
    sess = tf.Session(config=tf.test.benchmark_config())
    num_iters = 2 if self._testing else 200
    file_prefix = '%s/test-%s/' % (tf.compat.v1.test.get_temp_dir(),
                                   uuid.uuid4())
    n = 16 if self._testing else 128
    simple_data_spec = Data(
        step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
        value=tf.TensorSpec(shape=(n, n, 3), dtype=tf.int32))

    batch_sizes = [1, 4] if self._testing else [1, 4, 8, 16, 32, 64]

    for batch_size in batch_sizes:
      data = Data(step_type=tf.constant(np.full([batch_size], StepType.MID)),
                  value=tf.constant(np.ones([batch_size, n, n, 3], np.int32)))
      rb = tfrecord_replay_buffer.TFRecordReplayBuffer(
          experiment_id='write_performance_bs_%d' % batch_size,
          file_prefix=file_prefix,
          episodes_per_file=100000,
          data_spec=simple_data_spec)
      add_batch = rb.add_batch(data)

      # Burn-in.
      for _ in range(3):
        sess.run(add_batch)

      start = time.time()
      for _ in range(num_iters):
        sess.run(add_batch)
      rb.flush()
      end = time.time()

      self.report_benchmark(
          iters=num_iters,
          wall_time=(end - start) / num_iters,
          name='write_performance_bs_%d' % batch_size)


class TFRecordReplayBufferBenchmarkTest(tf.test.TestCase):

  def setUp(self):
    self._bench = TFRecordReplayBufferBenchmark(testing=True)

  def test_benchmark_write_performance(self):
    if tf.executing_eagerly():
      self.skipTest('Not running benchmak if executing eagerly.')
    self._bench.benchmark_write_performance()


if __name__ == '__main__':
  tf.test.main()
