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

import collections
import os

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.replay_buffers import table

from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


class TableTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testReadWriteSingle(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'camera'),
            specs.TensorSpec([3, 2], tf.float32, 'lidar')
        ]
    ]
    replay_table = table.Table(spec, capacity=3)
    variables = replay_table.variables()
    self.assertEqual(3, len(variables))
    self.assertAllEqual(['Table/action:0', 'Table/camera:0', 'Table/lidar:0'],
                        [v.name for v in variables])

    expected_values = [
        1 * np.ones(spec[0].shape.as_list()),
        [2 * np.ones(spec[1][0].shape.as_list()),
         3 * np.ones(spec[1][1].shape.as_list())]
    ]
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32),
        expected_values)

    write_op = replay_table.write(0, tensors)
    read_op = replay_table.read(0)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    read_value_ = self.evaluate(read_op)
    tf.nest.map_structure(self.assertAllClose, read_value_, expected_values)

  @test_util.run_in_graph_and_eager_modes()
  def testReadWriteBatch(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'camera'),
            specs.TensorSpec([3, 2], tf.float32, 'lidar')
        ]
    ]
    replay_table = table.Table(spec, capacity=4)

    batch_size = 2
    expected_values = [
        1 * np.ones([batch_size] + spec[0].shape.as_list()),
        [2 * np.ones([batch_size] + spec[1][0].shape.as_list()),
         3 * np.ones([batch_size] + spec[1][1].shape.as_list())]
    ]
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32),
        expected_values)

    write_op = replay_table.write(list(range(batch_size)), tensors)
    read_op = replay_table.read(list(range(batch_size)))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    read_value_ = self.evaluate(read_op)
    tf.nest.map_structure(self.assertAllClose, read_value_, expected_values)

  @test_util.run_in_graph_and_eager_modes()
  def testReadPartialSlots(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'camera'),
            specs.TensorSpec([3, 2], tf.float32, 'lidar')
        ]
    ]
    replay_table = table.Table(spec, capacity=4)

    batch_size = 2
    action = 1 * np.ones([batch_size] + spec[0].shape.as_list())
    camera = 2 * np.ones([batch_size] + spec[1][0].shape.as_list())
    lidar = 3 * np.ones([batch_size] + spec[1][1].shape.as_list())

    values = [action, [camera, lidar]]
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32), values)

    write_op = replay_table.write(list(range(batch_size)), tensors)
    read_op = replay_table.read(
        list(range(batch_size)), slots=['lidar', ['action']])
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    read_value_ = self.evaluate(read_op)
    expected_values = [lidar, [action]]
    tf.nest.map_structure(self.assertAllClose, read_value_, expected_values)

  @test_util.run_in_graph_and_eager_modes()
  def testWritePartialSlots(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'camera'),
            specs.TensorSpec([3, 2], tf.float32, 'lidar')
        ]
    ]
    replay_table = table.Table(spec, capacity=4)

    batch_size = 2

    action1 = 1 * np.ones([batch_size] + spec[0].shape.as_list())
    camera1 = 2 * np.ones([batch_size] + spec[1][0].shape.as_list())
    lidar1 = 3 * np.ones([batch_size] + spec[1][1].shape.as_list())
    write_op1 = replay_table.write(
        list(range(batch_size)), [action1, [camera1, lidar1]])

    lidar2 = 10 * np.ones([batch_size] + spec[1][1].shape.as_list())
    action2 = 20 * np.ones([batch_size] + spec[0].shape.as_list())
    write_op2 = replay_table.write(
        list(range(batch_size)), [lidar2, [action2]], ['lidar', ['action']])
    read_op = replay_table.read(list(range(batch_size)))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op1)
    self.evaluate(write_op2)
    read_value_ = self.evaluate(read_op)
    expected_values = [action2, [camera1, lidar2]]
    tf.nest.map_structure(self.assertAllClose, read_value_, expected_values)

  @test_util.run_in_graph_and_eager_modes()
  def testReadWriteDict(self):
    spec = {
        'action': specs.TensorSpec([3], tf.float32, 'action'),
        'camera': specs.TensorSpec([5], tf.float32, 'camera'),
        'lidar': specs.TensorSpec([3, 2], tf.float32, 'lidar')
    }
    replay_table = table.Table(spec, capacity=3)

    variables = replay_table.variables()
    self.assertEqual(3, len(variables))
    self.assertAllEqual(['Table/action:0', 'Table/camera:0', 'Table/lidar:0'],
                        [v.name for v in variables])

    expected_values = {
        'action': 1 * np.ones(spec['action'].shape.as_list()),
        'camera': 2 * np.ones(spec['camera'].shape.as_list()),
        'lidar': 3 * np.ones(spec['lidar'].shape.as_list())
    }
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32),
        expected_values)

    write_op = replay_table.write(0, tensors)
    read_op = replay_table.read(0)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    read_value_ = self.evaluate(read_op)
    tf.nest.map_structure(self.assertAllClose, read_value_, expected_values)

  @test_util.run_in_graph_and_eager_modes()
  def testReadWriteNamedTuple(self):
    # pylint: disable=invalid-name
    Observation = collections.namedtuple('Observation',
                                         ['action', 'camera', 'lidar'])
    # pylint: enable=invalid-name
    spec = Observation(
        action=specs.TensorSpec([3], tf.float32, 'action'),
        camera=specs.TensorSpec([5], tf.float32, 'camera'),
        lidar=specs.TensorSpec([3, 2], tf.float32, 'lidar')
    )
    replay_table = table.Table(spec, capacity=3)

    variables = replay_table.variables()
    self.assertEqual(3, len(variables))
    self.assertAllEqual(['Table/action:0', 'Table/camera:0', 'Table/lidar:0'],
                        [v.name for v in variables])

    expected_values = Observation(
        action=1 * np.ones(spec.action.shape.as_list()),
        camera=2 * np.ones(spec.camera.shape.as_list()),
        lidar=3 * np.ones(spec.lidar.shape.as_list())
    )
    tensors = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(value=x, dtype=tf.float32),
        expected_values)

    write_op = replay_table.write(0, tensors)
    read_op = replay_table.read(0)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    read_value_ = self.evaluate(read_op)
    tf.nest.map_structure(self.assertAllClose, read_value_, expected_values)

  @test_util.run_in_graph_and_eager_modes()
  def testEmptySpecNames(self):
    spec = [
        specs.TensorSpec([3], tf.float32),
        specs.TensorSpec([5], tf.float32, ''),
        specs.TensorSpec([3, 2], tf.float32, 'lidar')
    ]
    replay_table = table.Table(spec, capacity=3)

    variables = replay_table.variables()
    self.assertEqual(3, len(variables))
    self.assertAllEqual(['Table/slot:0', 'Table/slot_1:0', 'Table/lidar:0'],
                        [v.name for v in variables])

    expected_slots = ['slot', 'slot_1', 'lidar']
    self.assertAllEqual(replay_table.slots, expected_slots)
    tensors = replay_table.read(0, expected_slots)
    tf.nest.map_structure(lambda x, y: self.assertEqual(x.shape, y.shape), spec,
                          tensors)

  @test_util.run_in_graph_and_eager_modes()
  def testDuplicateSpecNames(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'lidar'),
        specs.TensorSpec([5], tf.float32, 'lidar'),
        specs.TensorSpec([3, 2], tf.float32, 'lidar')
    ]
    replay_table = table.Table(spec, capacity=3)

    variables = replay_table.variables()
    self.assertEqual(3, len(variables))
    self.assertAllEqual(['Table/lidar:0', 'Table/lidar_1:0', 'Table/lidar_2:0'],
                        [v.name for v in variables])

    expected_slots = ['lidar', 'lidar_1', 'lidar_2']
    self.assertAllEqual(replay_table.slots, expected_slots)
    tensors = replay_table.read(0, expected_slots)
    tf.nest.map_structure(lambda x, y: self.assertEqual(x.shape, y.shape), spec,
                          tensors)

  @test_util.run_in_graph_and_eager_modes()
  def testReadWriteString(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([], tf.string, 'camera'),
            specs.TensorSpec([3, 2], tf.float32, 'lidar')
        ]
    ]
    replay_table = table.Table(spec, capacity=3)
    variables = replay_table.variables()
    self.assertEqual(3, len(variables))
    self.assertAllEqual(['Table/action:0', 'Table/camera:0', 'Table/lidar:0'],
                        [v.name for v in variables])

    expected_values = [
        1 * np.ones(spec[0].shape.as_list()),
        [b'foo',
         3 * np.ones(spec[1][1].shape.as_list())]
    ]
    tensors = tf.nest.map_structure(
        lambda x, dtype: tf.convert_to_tensor(value=x, dtype=dtype),
        expected_values, [tf.float32, [tf.string, tf.float32]])

    write_op = replay_table.write(0, tensors)
    read_op = replay_table.read(0)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(write_op)
    read_value_ = self.evaluate(read_op)
    self.assertAllClose(read_value_[0], expected_values[0])
    self.assertEqual(read_value_[1][0], expected_values[1][0])
    self.assertAllClose(read_value_[1][1], expected_values[1][1])

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestore(self):
    spec = [
        specs.TensorSpec([3], tf.float32),
        specs.TensorSpec([5], tf.float32, 'lidar'),
        specs.TensorSpec([3, 2], tf.float32, 'lidar')
    ]
    replay_table = table.Table(spec, capacity=3)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, 'table')
    root = tf.train.Checkpoint(table=replay_table)
    save_path = root.save(prefix)
    root.restore(save_path).assert_consumed().run_restore_ops()


if __name__ == '__main__':
  tf.test.main()
