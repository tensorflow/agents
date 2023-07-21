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

"""Test for tf_agents.replay_buffers.table."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.replay_buffers import episodic_table

from tensorflow.python.framework import test_util  # TF internal


class EpisodicTableTest(tf.test.TestCase):

  def default_specs(self):
    return {'action': specs.TensorSpec([3], tf.float32, 'action'),
            'camera': specs.TensorSpec([5], tf.float32, 'camera'),
            'lidar': specs.TensorSpec([3, 2], tf.float32, 'lidar')}

  def np_values(self, spec, batch_size=1):
    def ones(sp):
      return np.ones(
          [batch_size] + sp.shape.as_list(), dtype=sp.dtype.as_numpy_dtype)

    return tf.nest.map_structure(ones, spec)

  @test_util.run_in_graph_and_eager_modes()
  def testGetAddSingle(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)

    expected_values = self.np_values(spec)

    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32),
        expected_values)

    add_op = replay_table.add([0], tensors)
    values = replay_table.get_episode_values(0)
    # Check static shape
    assert_same_shape = lambda s, v: self.assertEqual(s.shape[1:], v.shape)
    tf.nest.map_structure(assert_same_shape, values, spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(add_op)
    values_ = self.evaluate(values)
    tf.nest.map_structure(self.assertAllClose, expected_values, values_)

  @test_util.run_in_graph_and_eager_modes()
  def testGetEmpty(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    empty_values = self.np_values(spec, 0)
    values = replay_table.get_episode_values(0)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    values_ = self.evaluate(values)
    tf.nest.map_structure(self.assertAllClose, empty_values, values_)

  @test_util.run_in_graph_and_eager_modes()
  def testGetAddMultiple(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=4)

    batch_size = 2
    input_values = self.np_values(spec, batch_size)
    expected_values = self.np_values(spec)
    empty_values = self.np_values(spec, 0)
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32), input_values)

    write_op = replay_table.add([0, 1], tensors)
    values_0 = replay_table.get_episode_values(0)
    values_1 = replay_table.get_episode_values(1)
    # This should be empty
    values_2 = replay_table.get_episode_values(2)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_0))
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_1))
    tf.nest.map_structure(self.assertAllClose, empty_values,
                          self.evaluate(values_2))

  @test_util.run_in_graph_and_eager_modes()
  def testGetAddAppendMultiple(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=4)

    batch_size = 2
    input_values = self.np_values(spec, batch_size)
    expected_values = self.np_values(spec)
    empty_values = self.np_values(spec, 0)
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32), input_values)

    # Pull out the first entry in the batch, and add an outer
    # dimension to represent a single time step that we'll append.
    tensors_batch0 = tf.nest.map_structure(
        lambda x: tf.expand_dims(x[0, ...], 0), tensors)

    # We will append tensors_batch0 to row 0, which contains x[0].
    expected_appended_values = tf.nest.map_structure(
        lambda x: np.stack((x[0], x[0])), input_values)

    # batch_size == 2, so add [0, 1]
    write_op = replay_table.add([0, 1], tensors)
    append_op_0 = lambda: replay_table.append(0, tensors_batch0)

    values_0 = lambda: replay_table.get_episode_values(0)
    values_1 = lambda: replay_table.get_episode_values(1)
    values_2 = lambda: replay_table.get_episode_values(2)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_0()))
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_1()))
    tf.nest.map_structure(self.assertAllClose, empty_values,
                          self.evaluate(values_2()))

    self.evaluate(append_op_0())
    tf.nest.map_structure(self.assertAllClose, expected_appended_values,
                          self.evaluate(values_0()))

  @test_util.run_in_graph_and_eager_modes()
  def testExtend(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    test_values = self.np_values(spec, 5)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(replay_table.append(1, test_values))
    # Extend row 2 by the contents of row 1.
    self.evaluate(replay_table.extend(2, replay_table.get_episode_lists(1)))
    # Extend rows 0 and 1 by the contents of rows 1 and 2.
    self.evaluate(
        replay_table.extend([0, 1], replay_table.get_episode_lists([1, 2])))
    episode_0, episode_1, episode_2 = self.evaluate(
        [replay_table.get_episode_values(r) for r in range(3)])
    self.assertAllClose(episode_0, self.np_values(spec, 5))
    self.assertAllClose(episode_1, self.np_values(spec, 10))
    self.assertAllClose(episode_2, self.np_values(spec, 5))

  @test_util.run_in_graph_and_eager_modes()
  def testClear(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=4)

    batch_size = 2
    input_values = self.np_values(spec, batch_size)
    expected_values = self.np_values(spec)
    empty_values = self.np_values(spec, 0)
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32), input_values)

    write_op = replay_table.add([0, 1], tensors)
    values_0 = replay_table.get_episode_values(0)
    values_1 = replay_table.get_episode_values(1)
    # This should be empty
    clear_op = replay_table.clear()
    values_0_after_clear = replay_table.get_episode_values(0)
    values_1_after_clear = replay_table.get_episode_values(1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_0))
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_1))
    self.evaluate(clear_op)
    tf.nest.map_structure(self.assertAllClose, empty_values,
                          self.evaluate(values_0_after_clear))
    tf.nest.map_structure(self.assertAllClose, empty_values,
                          self.evaluate(values_1_after_clear))

  @test_util.run_in_graph_and_eager_modes()
  def testClearRows(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=4)

    batch_size = 2
    input_values = self.np_values(spec, batch_size)
    expected_values = self.np_values(spec)
    empty_values = self.np_values(spec, 0)
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32), input_values)

    write_op = replay_table.add([0, 1], tensors)
    values_0 = replay_table.get_episode_values(0)
    values_1 = replay_table.get_episode_values(1)
    # This should be empty
    clear_0 = replay_table.clear_rows([0])
    values_0_after_clear = replay_table.get_episode_values(0)
    values_1_after_clear = replay_table.get_episode_values(1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_0))
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_1))
    self.evaluate(clear_0)
    tf.nest.map_structure(self.assertAllClose, empty_values,
                          self.evaluate(values_0_after_clear))
    tf.nest.map_structure(self.assertAllClose, expected_values,
                          self.evaluate(values_1_after_clear))

  @test_util.run_in_graph_and_eager_modes()
  def testGetEpisodeListsOneRow(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    test_values = self.np_values(spec, 5)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(replay_table.append(1, test_values))
    # Get the episode lists we just added. The returned value should look like:
    # {'action': <tensorlist>, 'camera': <tensorlist>, 'lidar': <tensorlist>}
    episode_lists = replay_table.get_episode_lists(1)
    for episode_list_slot in tf.nest.flatten(episode_lists):
      self.assertEqual(episode_list_slot.shape.rank, 0)
    episode_tensors = tf.nest.map_structure(replay_table._stack_tensor_list,
                                            replay_table.slots, episode_lists)
    self.assertAllClose(self.evaluate(episode_tensors), test_values)

  @test_util.run_in_graph_and_eager_modes()
  def testGetEpisodeListsSomeRows(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    empty_values = self.np_values(spec, 0)
    test_values = self.np_values(spec, 5)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(replay_table.append(1, test_values))
    # Get episode lists for rows 1 and 2. The returned values should look like:
    # {'action': [<tensorlist for row 1>, <tensorlist for row 2>],
    #  'camera': [<tensorlist for row 1>, <tensorlist for row 2>],
    #  'lidar': [<tensorlist for row 1>, <tensorlist for row 2>]}
    episode_lists = replay_table.get_episode_lists([1, 2])
    for episode_list_slot in tf.nest.flatten(episode_lists):
      self.assertEqual(episode_list_slot.shape.rank, 1)
      self.assertEqual(self.evaluate(tf.size(input=episode_list_slot)), 2)
    # Stack episode tensors for row 1, i.e.:
    # {'action': <five items>, 'camera': <five items>, 'lidar': <five items>}
    episode_tensors_1 = tf.nest.map_structure(
        lambda slot, lists: replay_table._stack_tensor_list(slot, lists[0]),
        replay_table.slots, episode_lists)
    # Should be equivalent to
    # Stack episode tensors for row 2, which is empty.
    episode_tensors_2 = tf.nest.map_structure(
        lambda slot, lists: replay_table._stack_tensor_list(slot, lists[1]),
        replay_table.slots, episode_lists)
    self.assertAllClose(self.evaluate(episode_tensors_1), test_values)
    self.assertAllClose(self.evaluate(episode_tensors_2), empty_values)

  @test_util.run_in_graph_and_eager_modes()
  def testGetEpisodeListsAllRows(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    empty_values = self.np_values(spec, 0)
    test_values = self.np_values(spec, 5)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(replay_table.append(1, test_values))
    # Get episode lists for all rows. The returned values should look like:
    # {'action': [<tensorlist for row 0>, <tensorlist for row 1>,
    #             <tensorlist for row 2>],
    #  ... }
    self._test_episode_lists(replay_table, empty_values, test_values)

  def _test_episode_lists(self, replay_table, empty_values, test_values):
    episode_lists = replay_table.get_episode_lists()
    for episode_list_slot in tf.nest.flatten(episode_lists):
      self.assertEqual(episode_list_slot.shape.rank, 1)
      self.assertEqual(self.evaluate(tf.size(input=episode_list_slot)), 3)
    # Stack each row individually.
    episode_tensors_0 = tf.nest.map_structure(
        lambda slot, lists: replay_table._stack_tensor_list(slot, lists[0]),
        replay_table.slots, episode_lists)
    episode_tensors_1 = tf.nest.map_structure(
        lambda slot, lists: replay_table._stack_tensor_list(slot, lists[1]),
        replay_table.slots, episode_lists)
    episode_tensors_2 = tf.nest.map_structure(
        lambda slot, lists: replay_table._stack_tensor_list(slot, lists[2]),
        replay_table.slots, episode_lists)
    # Only row 1 should have non-empty values.
    self.assertAllClose(self.evaluate(episode_tensors_0), empty_values)
    self.assertAllClose(self.evaluate(episode_tensors_1), test_values)
    self.assertAllClose(self.evaluate(episode_tensors_2), empty_values)

  def testCheckpoint(self):
    spec = self.default_specs()
    replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    empty_values = self.np_values(spec, 0)
    test_values = self.np_values(spec, 5)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(replay_table.append(1, test_values))

    checkpoint = tf.train.Checkpoint(table=replay_table)
    tmpdir = self.get_temp_dir()
    if tf.executing_eagerly():
      location = checkpoint.save(os.path.join(tmpdir, 'ckpt'))
    else:
      with self.cached_session() as sess:
        location = checkpoint.save(os.path.join(tmpdir, 'ckpt'), session=sess)
    reload_replay_table = episodic_table.EpisodicTable(spec, capacity=3)
    reload_checkpoint = tf.train.Checkpoint(table=reload_replay_table)
    status = reload_checkpoint.restore(location)
    status.assert_consumed()
    with self.cached_session() as sess:
      status.initialize_or_restore(session=sess)
    self._test_episode_lists(replay_table, empty_values, test_values)
    self._test_episode_lists(reload_replay_table, empty_values, test_values)


if __name__ == '__main__':
  tf.test.main()
