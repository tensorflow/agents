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
    if tf.executing_eagerly():
      self.skipTest('b/123886086')
    batch_size = 1
    spec = self._data_spec()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size, max_length=10)

    values, add_op = _get_add_op(spec, replay_buffer, batch_size)
    sample, _ = replay_buffer.get_next(sample_batch_size=3)
    clear_op = replay_buffer.clear()
    items_op = replay_buffer.gather_all()
    last_id_op = replay_buffer._get_last_id()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    last_id = self.evaluate(last_id_op)
    empty_items = self.evaluate(items_op)
    self.evaluate(add_op)
    values_ = self.evaluate(values)
    sample_ = self.evaluate(sample)
    tf.nest.map_structure(lambda x, y: self._assertContains([x], list(y)),
                          values_, sample_)
    self.assertNotEqual(last_id, self.evaluate(last_id_op))

    self.evaluate(clear_op)
    self.assertEqual(last_id, self.evaluate(last_id_op))

    def check_np_arrays_everything_equal(x, y):
      np.testing.assert_equal(x, y)
      self.assertEqual(x.dtype, y.dtype)

    tf.nest.map_structure(check_np_arrays_everything_equal, empty_items,
                          self.evaluate(items_op))

  def testClearAllVariables(self):
    if tf.executing_eagerly():
      self.skipTest('b/123886086')
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

    last_id_op = replay_buffer._get_last_id()
    add_op = replay_buffer.add_batch(values_batched)
    sample, _ = replay_buffer.get_next(sample_batch_size=3)
    clear_op = replay_buffer._clear(clear_all_variables=True)
    items_op = replay_buffer.gather_all()
    table_vars = [
        var for var in replay_buffer.variables() if 'Table' in var.name
    ]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(clear_op)
    empty_table_vars = self.evaluate(table_vars)
    last_id = self.evaluate(last_id_op)
    empty_items = self.evaluate(items_op)
    self.evaluate(add_op)
    self.evaluate(add_op)
    self.evaluate(add_op)
    self.evaluate(add_op)
    values_ = self.evaluate(values)
    sample_ = self.evaluate(sample)
    tf.nest.map_structure(lambda x, y: self._assertContains([x], list(y)),
                          values_, sample_)
    self.assertNotEqual(last_id, self.evaluate(last_id_op))

    tf.nest.map_structure(lambda x, y: self.assertFalse(np.all(x == y)),
                          empty_table_vars, self.evaluate(table_vars))

    self.evaluate(clear_op)
    self.assertEqual(last_id, self.evaluate(last_id_op))

    def check_np_arrays_everything_equal(x, y):
      np.testing.assert_equal(x, y)
      self.assertEqual(x.dtype, y.dtype)

    tf.nest.map_structure(check_np_arrays_everything_equal, empty_items,
                          self.evaluate(items_op))

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testMultiStepSampling(self, batch_size):
    if tf.executing_eagerly():
      self.skipTest('b/123885577')
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    action = tf.stack([tf.Variable(0).count_up_to(10)] * batch_size)

    add_op = replay_buffer.add_batch(action)
    (step, next_step), _ = replay_buffer.get_next(num_steps=2,
                                                  time_stacked=False)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(10):
      self.evaluate(add_op)
    for _ in range(100):
      step_, next_step_ = self.evaluate([step, next_step])
      self.assertEqual((step_ + 1) % 10, next_step_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testMultiStepStackedSampling(self, batch_size):
    if tf.executing_eagerly():
      self.skipTest('b/123885577')
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    actions = tf.stack([tf.Variable(0).count_up_to(10)] * batch_size)

    add_op = replay_buffer.add_batch(actions)
    steps, _ = replay_buffer.get_next(num_steps=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(10):
      self.evaluate(add_op)
    for _ in range(100):
      steps_ = self.evaluate(steps)
      self.assertEqual((steps_[0] + 1) % 10, steps_[1])

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testMultiStepStackedBatchedSampling(self, batch_size):
    if tf.executing_eagerly():
      self.skipTest('b/123885577')
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    actions = tf.stack([tf.Variable(0).count_up_to(10)] * batch_size)

    add_op = replay_buffer.add_batch(actions)
    steps, _ = replay_buffer._get_next(3, num_steps=2, time_stacked=True)
    self.assertEqual(steps.shape, [3, 2])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(10):
      self.evaluate(add_op)
    for _ in range(100):
      steps_ = self.evaluate(steps)
      self.assertAllEqual((steps_[:, 0] + 1) % 10, steps_[:, 1])

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testGatherAll(self, batch_size):
    if tf.executing_eagerly():
      self.skipTest('b/123883577')
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size)

    action_variables = [
        tf.Variable(i).count_up_to(i + 10) for i in range(0, batch_size)
    ]
    actions = tf.stack(action_variables)

    add_op = replay_buffer.add_batch(actions)
    items = replay_buffer.gather_all()
    expected = [list(range(i, i + 10)) for i in range(0, batch_size)]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(10):
      self.evaluate(add_op)
    items_ = self.evaluate(items)
    self.assertAllClose(expected, items_)

  @parameterized.named_parameters(
      ('BatchSizeOne', 1),
      ('BatchSizeFive', 5),
  )
  def testGatherAllOverCapacity(self, batch_size):
    if tf.executing_eagerly():
      self.skipTest('b/123883577')
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=batch_size, max_length=10)

    # Each element has its batch index in the 100s place.
    actions = tf.stack([
        tf.Variable(x * 100).count_up_to(15 + x * 100)
        for x in range(batch_size)
    ])

    add_op = replay_buffer.add_batch(actions)
    items = replay_buffer.gather_all()
    expected = [
        list(range(5 + x * 100, 15 + x * 100)) for x in range(batch_size)
    ]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(15):
      self.evaluate(add_op)
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
    if tf.executing_eagerly():
      self.skipTest('b/123771990')
    max_length = 3
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=buffer_batch_size, max_length=max_length)

    actions = tf.stack([tf.Variable(0).count_up_to(9)] * buffer_batch_size)
    add_op = replay_buffer.add_batch(actions)

    ds = replay_buffer.as_dataset()
    itr = tf.compat.v1.data.make_initializable_iterator(ds)
    _, buffer_info = itr.get_next()
    probabilities = buffer_info.probabilities

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(itr.initializer)
    num_adds = 5
    for i in range(1, num_adds):
      self.evaluate(add_op)
      probabilities_ = self.evaluate(probabilities)
      expected_probability = (
          1. / min(i * buffer_batch_size, max_length * buffer_batch_size))
      self.assertAllClose(expected_probability, probabilities_)


if __name__ == '__main__':
  tf.test.main()
