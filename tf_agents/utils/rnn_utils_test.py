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

"""Tests for agents.rnn_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.utils import rnn_utils
from tensorflow.python.framework import test_util  # TF internal

nest = tf.contrib.framework.nest


class AddInputAndStateRNNCell(tf.contrib.rnn.LayerRNNCell):

  @property
  def state_size(self):
    return tf.TensorShape([1])

  @property
  def output_size(self):
    return tf.TensorShape([1])

  def call(self, input_, state):
    s = input_ + state
    return s, s


class AddInputAndStateKerasRNNCell(tf.keras.layers.Layer):

  def __init__(self):
    super(AddInputAndStateKerasRNNCell, self).__init__()
    self.output_size = 1
    self.state_size = 1

  def call(self, input_, state):
    s = input_ + state
    return s, s

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      return tf.zeros_like(inputs)
    return tf.zeros([batch_size, 1], dtype)


class FixedOneRNNCell(tf.contrib.rnn.LayerRNNCell):

  @property
  def state_size(self):
    return tf.TensorShape([1])

  @property
  def output_size(self):
    return tf.TensorShape([1])

  def call(self, input_, state):
    s = tf.ones_like(state)
    return input_, s


class RangeWithResetMaskTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testSimple(self):
    reset_mask = np.array(
        [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
         [0, 1, 1, 1, 0, 0, 0, 0, 0, 1]])
    expected = np.array(
        [[0, 1, 0, 1, 2, 3, 0, 1, 2, 3],
         [0, 1, 2, 0, 1, 2, 0, 1, 0, 1],
         [0, 0, 0, 0, 1, 2, 3, 4, 5, 0]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))

  @test_util.run_in_graph_and_eager_modes()
  def testSimpleNoResets(self):
    reset_mask = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    expected = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))

  @test_util.run_in_graph_and_eager_modes()
  def testSpecialCases(self):
    reset_mask = np.array([[0], [1]])
    expected = np.array([[0], [0]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))

    reset_mask = np.array([[1], [1], [1]])
    expected = np.array([[0], [0], [0]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))

    reset_mask = np.array([[0]])
    expected = np.array([[0]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))

    reset_mask = np.array([[1]])
    expected = np.array([[0]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))

    reset_mask = np.array([[]])
    expected = np.array([[]])
    self.assertAllEqual(
        expected,
        self.evaluate(
            rnn_utils.range_with_reset_mask(reset_mask)))


class DynamicUnrollTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testDynamicUnrollMatchesDynamicRNNWhenNoReset(self):
    cell = tf.nn.rnn_cell.LSTMCell(3)
    batch_size = 4
    max_time = 7
    inputs = tf.random_uniform((batch_size, max_time, 2), dtype=tf.float32)
    reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
    outputs_dun, final_state_dun, _ = rnn_utils.dynamic_unroll(
        cell, inputs, reset_mask, dtype=tf.float32)
    outputs_drnn, final_state_drnn = tf.nn.dynamic_rnn(
        cell, inputs, dtype=tf.float32)
    self.evaluate(tf.global_variables_initializer())
    outputs_dun, final_state_dun, outputs_drnn, final_state_drnn = (
        self.evaluate(
            (outputs_dun, final_state_dun, outputs_drnn, final_state_drnn)))
    self.assertAllClose(outputs_dun, outputs_drnn)
    self.assertAllClose(final_state_dun, final_state_drnn)

  @test_util.run_in_graph_and_eager_modes()
  def testDynamicUnrollMatchesDynamicRNNWhenNoResetSingleTimeStep(self):
    cell = tf.nn.rnn_cell.LSTMCell(3)
    batch_size = 4
    max_time = 1
    inputs = tf.random_uniform((batch_size, max_time, 2), dtype=tf.float32)
    reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
    outputs_dun, final_state_dun, _ = rnn_utils.dynamic_unroll(
        cell, inputs, reset_mask, dtype=tf.float32)
    outputs_drnn, final_state_drnn = tf.nn.dynamic_rnn(
        cell, inputs, dtype=tf.float32)
    self.evaluate(tf.global_variables_initializer())
    outputs_dun, final_state_dun, outputs_drnn, final_state_drnn = (
        self.evaluate(
            (outputs_dun, final_state_dun, outputs_drnn, final_state_drnn)))
    self.assertAllClose(outputs_dun, outputs_drnn)
    self.assertAllClose(final_state_dun, final_state_drnn)

  @test_util.run_in_graph_and_eager_modes()
  def testDynamicUnrollResetsStateOnReset(self):
    self._testDynamicUnrollResetsStateOnReset(
        AddInputAndStateRNNCell)
    self._testDynamicUnrollResetsStateOnReset(
        AddInputAndStateKerasRNNCell)

  def _testDynamicUnrollResetsStateOnReset(self, cell_type):
    cell = cell_type()
    batch_size = 4
    max_time = 7
    inputs = tf.random_uniform((batch_size, max_time, 1))
    reset_mask = (tf.random_normal((batch_size, max_time)) > 0)

    outputs, final_state, _ = rnn_utils.dynamic_unroll(
        cell, inputs, reset_mask, dtype=tf.float32)

    nest.assert_same_structure(outputs, cell.output_size)
    nest.assert_same_structure(final_state, cell.state_size)

    reset_mask, inputs, outputs, final_state = self.evaluate(
        (reset_mask, inputs, outputs, final_state))

    self.assertAllClose(outputs[:, -1, :], final_state)

    # outputs will contain cumulative sums up until a reset
    expected_outputs = []
    state = np.zeros_like(final_state)
    for i, frame in enumerate(np.transpose(inputs, [1, 0, 2])):
      state = state * np.reshape(~reset_mask[:, i], state.shape) + frame
      expected_outputs.append(np.array(state))
    expected_outputs = np.transpose(expected_outputs, [1, 0, 2])
    self.assertAllClose(outputs, expected_outputs)


class RNNUtilsBenchmark(tf.test.Benchmark):

  def benchmark_range_with_reset_mask(self):
    for batch_size in [1, 32, 128, 512]:
      for n in [1, 10, 100, 1000]:
        tf.reset_default_graph()
        # reset == true for ~33% of the locations.
        reset_mask = tf.get_variable(
            'v', initializer=tf.random_normal((batch_size, n)) > 2)
        v = rnn_utils.range_with_reset_mask(reset_mask)
        v_scan = rnn_utils._range_with_reset_mask_scan(reset_mask)  # pylint: disable=protected-access
        s = tf.Session()
        s.run(reset_mask.initializer)
        self.run_op_benchmark(
            s, v.op,
            name='range_with_reset_mask_bs_%d_n_%d' % (batch_size, n))
        self.run_op_benchmark(
            s, v_scan.op,
            name='range_with_reset_mask_scan_bs_%d_n_%d' % (batch_size, n))


if __name__ == '__main__':
  tf.test.main()
