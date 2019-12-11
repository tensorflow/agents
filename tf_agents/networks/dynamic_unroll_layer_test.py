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

"""Tests for networks.dynamic_unroll_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.networks import dynamic_unroll_layer
from tensorflow.python.framework import test_util  # TF internal


class AddInputAndStateKerasRNNCell(tf.keras.layers.Layer):

  def __init__(self):
    super(AddInputAndStateKerasRNNCell, self).__init__()
    self.output_size = 1
    self.state_size = 1

  def call(self, input_, state, training=False):
    s = input_ + state
    return s, s

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      return tf.zeros_like(inputs)
    return tf.zeros([batch_size, 1], dtype)

  def get_dropout_mask_for_cell(self, **kwargs):
    pass

  def get_recurrent_dropout_mask_for_cell(self, **kwargs):
    pass

  def reset_dropout_mask(self):
    pass

  def reset_recurrent_dropout_mask(self):
    pass

class DynamicUnrollTest(tf.test.TestCase):

  def testFromConfigLSTM(self):
    l1 = dynamic_unroll_layer.DynamicUnroll(
        tf.keras.layers.LSTMCell(units=3), parallel_iterations=10)
    l2 = dynamic_unroll_layer.DynamicUnroll.from_config(l1.get_config())
    self.assertEqual(l1.get_config(), l2.get_config())


  def testDropoutKerasLSTM(self):
    def _testDropoutKerasLSTMHelper(training, dropout=0.0,
                                    recurrent_dropout=0.0):
      cell = tf.keras.layers.LSTMCell(3, dropout=dropout,
                                      recurrent_dropout=recurrent_dropout)
      batch_size = 4
      max_time = 7
      inputs = tf.random.uniform((batch_size, max_time, 2), dtype=tf.float32)
      reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
      layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
      outputs_dun1, final_state_dun1 = layer(inputs, reset_mask, training=training)
      outputs_dun2, final_state_dun2 = layer(inputs, reset_mask, training=training)

      if not training:
        self.assertAllEqual(outputs_dun1, outputs_dun2)
      else:
        self.assertGreater(np.linalg.norm(outputs_dun1 - outputs_dun2), 0)

    _testDropoutKerasLSTMHelper(training=False, dropout=0.50)
    _testDropoutKerasLSTMHelper(training=True, dropout=0.50)
    _testDropoutKerasLSTMHelper(training=False, recurrent_dropout=0.50)
    _testDropoutKerasLSTMHelper(training=True, recurrent_dropout=0.50)

  @test_util.run_in_graph_and_eager_modes()
  def testDynamicUnrollMatchesDynamicRNNWhenNoReset(self):
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(3)
    batch_size = 4
    max_time = 7
    inputs = tf.random.uniform((batch_size, max_time, 2), dtype=tf.float32)
    reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
    layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
    outputs_dun, final_state_dun = layer(inputs, reset_mask)
    outputs_drnn, final_state_drnn = tf.compat.v1.nn.dynamic_rnn(
        cell, inputs, dtype=tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    outputs_dun, final_state_dun, outputs_drnn, final_state_drnn = (
        self.evaluate(
            (outputs_dun, final_state_dun, outputs_drnn, final_state_drnn)))
    self.assertAllClose(outputs_dun, outputs_drnn)
    self.assertAllClose(final_state_dun, final_state_drnn)

  @test_util.run_in_graph_and_eager_modes()
  def testDynamicUnrollMatchesDynamicRNNWhenNoResetSingleTimeStep(self):
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(3)
    batch_size = 4
    max_time = 1
    inputs = tf.random.uniform((batch_size, max_time, 2), dtype=tf.float32)
    reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
    layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
    outputs_dun, final_state_dun = layer(inputs, reset_mask)
    outputs_drnn, final_state_drnn = tf.compat.v1.nn.dynamic_rnn(
        cell, inputs, dtype=tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    outputs_dun, final_state_dun, outputs_drnn, final_state_drnn = (
        self.evaluate(
            (outputs_dun, final_state_dun, outputs_drnn, final_state_drnn)))
    self.assertAllClose(outputs_dun, outputs_drnn)
    self.assertAllClose(final_state_dun, final_state_drnn)

  @test_util.run_in_graph_and_eager_modes()
  def testDynamicUnrollResetsStateOnReset(self):
    if hasattr(tf, 'contrib'):
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

      self._testDynamicUnrollResetsStateOnReset(
          AddInputAndStateRNNCell)

    self._testDynamicUnrollResetsStateOnReset(
        AddInputAndStateKerasRNNCell)

  def _testDynamicUnrollResetsStateOnReset(self, cell_type):
    cell = cell_type()
    batch_size = 4
    max_time = 7
    inputs = tf.random.uniform((batch_size, max_time, 1))
    reset_mask = (tf.random.normal((batch_size, max_time)) > 0)

    layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
    outputs, final_state = layer(inputs, reset_mask)

    tf.nest.assert_same_structure(outputs, cell.output_size)
    tf.nest.assert_same_structure(final_state, cell.state_size)

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


if __name__ == '__main__':
  tf.test.main()
