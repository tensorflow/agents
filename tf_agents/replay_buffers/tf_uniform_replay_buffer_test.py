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

"""Tests for tf_uniform_replay_buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents import specs
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import test_utils


def _get_add_op(spec, replay_buffer, batch_size):
  # TODO(b/68398658) Remove dtypes once scatter_update is fixed.
  action = tf.constant(1 * np.ones(spec[0].shape.as_list(), dtype=np.float32))
  lidar = tf.constant(2 * np.ones(spec[1][0].shape.as_list(), dtype=np.float32))
  camera = tf.constant(
      3 * np.ones(spec[1][1].shape.as_list(), dtype=np.float32))
  values = [action, [lidar, camera]]
  values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size),
                                         values)

  return values, replay_buffer.add_batch(values_batched)


class TFUniformReplayBufferTest(parameterized.TestCase, tf.test.TestCase):

  def _assertContains(self, list1, list2):
    self.assertTrue(
        test_utils.contains(list1, list2), '%s vs. %s' % (list1, list2))

  def _assertCircularOrdering(self, expected_order, given_order):
    for i in xrange(len(given_order)):
      self.assertIn(given_order[i], expected_order)
      if i > 0:
        prev_idx = expected_order.index(given_order[i - 1])
        cur_idx = expected_order.index(given_order[i])
        self.assertEqual(cur_idx, (prev_idx + 1) % len(expected_order))

  def _data_spec(self):
    return [
        specs.TensorSpec([3], tf.float32, 'action'),
        [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testAdd(self, batch_size):
    spec = self._data_spec()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec,
        batch_size=batch_size,
        max_length=10,
        scope='rb{}'.format(batch_size))

    values, add_op = _get_add_op(spec, replay_buffer, batch_size)
    sample, _ = replay_buffer.get_next()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_op)
    sample_ = self.evaluate(sample)
    values_ = self.evaluate(values)
    tf.nest.map_structure(self.assertAllClose, values_, sample_)

  def testGetNextEmpty(self):
    spec = self._data_spec()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=1, max_length=10)

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError, 'TFUniformReplayBuffer is empty. Make '
        'sure to add items before sampling the buffer.'):
      self.evaluate(tf.compat.v1.global_variables_initializer())
      sample, _ = replay_buffer.get_next()
      self.evaluate(sample)

  def testAddSingleSampleBatch(self):
    batch_size = 1
    spec = self._data_spec()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size, max_length=10)

    values, add_op = _get_add_op(spec, replay_buffer, batch_size)
    sample, _ = replay_buffer.get_next(sample_batch_size=3)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_op)
    values_ = self.evaluate(values)
    sample_ = self.evaluate(sample)
    tf.nest.map_structure(lambda x, y: self._assertContains([x], list(y)),
                          values_, sample_)

  def testClear(self):
    batch_size = 1
    spec = self._data_spec()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size, max_length=10)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    initial_id = self.evaluate(replay_buffer._get_last_id())
    empty_items = self.evaluate(replay_buffer.gather_all())

    values, _ = self.evaluate(_get_add_op(spec, replay_buffer, batch_size))
    sample, _ = self.evaluate(replay_buffer.get_next(sample_batch_size=3))
    tf.nest.map_structure(lambda x, y: self._assertContains([x], list(y)),
                          values, sample)
    self.assertNotEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

    self.evaluate(replay_buffer.clear())
    self.assertEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

    def check_np_arrays_everything_equal(x, y):
      np.testing.assert_equal(x, y)
      self.assertEqual(x.dtype, y.dtype)

    tf.nest.map_structure(check_np_arrays_everything_equal, empty_items,
                          self.evaluate(replay_buffer.gather_all()))

  def testClearAllVariables(self):
    batch_size = 1
    spec = self._data_spec()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size, max_length=10)

    action = tf.constant(1 * np.ones(spec[0].shape.as_list(), dtype=np.float32))
    lidar = tf.constant(
        2 * np.ones(spec[1][0].shape.as_list(), dtype=np.float32))
    camera = tf.constant(
        3 * np.ones(spec[1][1].shape.as_list(), dtype=np.float32))
    values = [action, [lidar, camera]]
    values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * batch_size),
                                           values)

    if tf.executing_eagerly():
      add_op = lambda: replay_buffer.add_batch(values_batched)
    else:
      add_op = replay_buffer.add_batch(values_batched)

    def get_table_vars():
      return [var for var in replay_buffer.variables() if 'Table' in var.name]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(replay_buffer._clear(clear_all_variables=True))
    empty_table_vars = self.evaluate(get_table_vars())
    initial_id = self.evaluate(replay_buffer._get_last_id())
    empty_items = self.evaluate(replay_buffer.gather_all())
    self.evaluate(add_op)
    self.evaluate(add_op)
    self.evaluate(add_op)
    self.evaluate(add_op)
    values_ = self.evaluate(values)
    sample, _ = self.evaluate(replay_buffer.get_next(sample_batch_size=3))
    tf.nest.map_structure(lambda x, y: self._assertContains([x], list(y)),
                          values_, sample)
    self.assertNotEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

    tf.nest.map_structure(lambda x, y: self.assertFalse(np.all(x == y)),
                          empty_table_vars, self.evaluate(get_table_vars()))

    self.evaluate(replay_buffer._clear(clear_all_variables=True))
    self.assertEqual(initial_id, self.evaluate(replay_buffer._get_last_id()))

    def check_np_arrays_everything_equal(x, y):
      np.testing.assert_equal(x, y)
      self.assertEqual(x.dtype, y.dtype)

    tf.nest.map_structure(check_np_arrays_everything_equal, empty_items,
                          self.evaluate(replay_buffer.gather_all()))

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testMultiStepSampling(self, batch_size):
    spec = specs.TensorSpec([], tf.int64, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    @common.function(autograph=True)
    def add_data():
      for i in tf.range(10, dtype=tf.int64):
        replay_buffer.add_batch(tf.ones((batch_size,), dtype=tf.int64) * i)

    if tf.executing_eagerly():
      sample = lambda: replay_buffer.get_next(num_steps=2, time_stacked=False)
    else:
      sample = replay_buffer.get_next(
          num_steps=2, time_stacked=False)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_data())

    for _ in range(100):
      (step_, next_step_), _ = self.evaluate(sample)
      self.assertEqual((step_ + 1) % 10, next_step_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testMultiStepStackedSampling(self, batch_size):
    spec = specs.TensorSpec([], tf.int64, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    @common.function(autograph=True)
    def add_data():
      for i in tf.range(10, dtype=tf.int64):
        replay_buffer.add_batch(tf.ones((batch_size,), dtype=tf.int64) * i)

    if tf.executing_eagerly():
      steps = lambda: replay_buffer.get_next(num_steps=2)[0]
    else:
      steps, _ = replay_buffer.get_next(num_steps=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_data())
    for _ in range(100):
      steps_ = self.evaluate(steps)
      self.assertEqual((steps_[0] + 1) % 10, steps_[1])

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testMultiStepStackedBatchedSampling(self, batch_size):
    spec = specs.TensorSpec([], tf.int64, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    @common.function(autograph=True)
    def add_data():
      for i in tf.range(10, dtype=tf.int64):
        replay_buffer.add_batch(tf.ones((batch_size,), dtype=tf.int64) * i)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_data())

    if tf.executing_eagerly():
      steps = lambda: replay_buffer._get_next(3,  # pylint: disable=g-long-lambda
                                              num_steps=2,
                                              time_stacked=True)[0]
    else:
      steps, _ = replay_buffer._get_next(3, num_steps=2, time_stacked=True)
    self.assertEqual(self.evaluate(steps).shape, (3, 2))

    for _ in range(100):
      steps_ = self.evaluate(steps)
      self.assertAllEqual((steps_[:, 0] + 1) % 10, steps_[:, 1])

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testGatherAll(self, batch_size):
    spec = specs.TensorSpec([], tf.int64, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    @common.function(autograph=True)
    def add_data():
      for i in tf.range(10, dtype=tf.int64):
        batch = tf.range(i, i + batch_size, 1, dtype=tf.int64)
        replay_buffer.add_batch(batch)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_data())

    items = replay_buffer.gather_all()
    expected = [list(range(i, i + 10)) for i in range(0, batch_size)]

    items_ = self.evaluate(items)
    self.assertAllClose(expected, items_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testGatherAllOverCapacity(self, batch_size):
    spec = specs.TensorSpec([], tf.int64, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size, max_length=10)

    @common.function(autograph=True)
    def add_data():
      # Each element has its batch index in the 100s place.
      for i in tf.range(15, dtype=tf.int64):
        batch = tf.range(0, batch_size * 100, 100, dtype=tf.int64) + i
        replay_buffer.add_batch(batch)

    expected = [
        list(range(5 + x * 100, 15 + x * 100)) for x in range(batch_size)
    ]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_data())
    items = replay_buffer.gather_all()
    items_ = self.evaluate(items)
    self.assertAllClose(expected, items_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testGatherAllEmpty(self, batch_size):
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    items = replay_buffer.gather_all()
    expected = [[]] * batch_size

    self.evaluate(tf.compat.v1.global_variables_initializer())
    items_ = self.evaluate(items)
    self.assertAllClose(expected, items_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testSampleBatchCorrectProbabilities(self, buffer_batch_size):
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=buffer_batch_size, max_length=4)

    actions = tf.stack([tf.Variable(0).count_up_to(9)] * buffer_batch_size)
    sample_batch_size = 2

    @common.function
    def add(actions):
      replay_buffer.add_batch(actions)

    @common.function
    def probabilities():
      _, buffer_info = replay_buffer.get_next(
          sample_batch_size=sample_batch_size)
      return buffer_info.probabilities

    self.evaluate(tf.compat.v1.global_variables_initializer())
    num_adds = 3
    for i in range(1, num_adds):
      self.evaluate(add(actions))
      expected_probabilities = [1. /
                                (i * buffer_batch_size)] * sample_batch_size
      probabilities_ = self.evaluate(probabilities())
      self.assertAllClose(expected_probabilities, probabilities_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testSampleSingleCorrectProbability(self, buffer_batch_size):
    max_length = 3
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=buffer_batch_size, max_length=max_length)

    actions = tf.stack([tf.Variable(0).count_up_to(9)] * buffer_batch_size)

    @common.function
    def add(actions):
      replay_buffer.add_batch(actions)

    @common.function
    def probabilities():
      _, buffer_info = replay_buffer.get_next()
      return buffer_info.probabilities

    self.evaluate(tf.compat.v1.global_variables_initializer())

    num_adds = 5
    for i in range(1, num_adds):
      self.evaluate(add(actions))
      probabilities_ = self.evaluate(probabilities())
      expected_probability = (
          1. / min(i * buffer_batch_size, max_length * buffer_batch_size))
      self.assertAllClose(expected_probability, probabilities_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testSampleSingleCorrectProbabilityAsDataset(self, buffer_batch_size):
    max_length = 3
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=buffer_batch_size, max_length=max_length)

    actions = tf.stack([tf.Variable(0).count_up_to(9)] * buffer_batch_size)

    self.evaluate(tf.compat.v1.global_variables_initializer())

    ds = replay_buffer.as_dataset()
    if tf.executing_eagerly():
      add_op = lambda: replay_buffer.add_batch(actions)
      itr = iter(ds)
      sample = lambda: next(itr)
    else:
      add_op = replay_buffer.add_batch(actions)
      itr = tf.compat.v1.data.make_initializable_iterator(ds)
      self.evaluate(itr.initializer)
      sample = itr.get_next()

    num_adds = 5
    for i in range(1, num_adds):
      self.evaluate(add_op)
      probabilities_ = self.evaluate(sample)[1].probabilities
      expected_probability = (
          1. / min(i * buffer_batch_size, max_length * buffer_batch_size))
      self.assertAllClose(expected_probability, probabilities_)


if __name__ == '__main__':
  tf.test.main()
