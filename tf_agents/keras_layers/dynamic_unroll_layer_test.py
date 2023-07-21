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

"""Tests for tf_agents.keras_layers.dynamic_unroll_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import dynamic_unroll_layer


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


class DynamicUnrollTest(parameterized.TestCase, tf.test.TestCase):

  def testFromConfigLSTM(self):
    l1 = dynamic_unroll_layer.DynamicUnroll(
        tf.keras.layers.LSTMCell(units=3), parallel_iterations=10)
    l2 = dynamic_unroll_layer.DynamicUnroll.from_config(l1.get_config())
    self.assertEqual(l1.get_config(), l2.get_config())

  @parameterized.named_parameters(
      ('WithMask', True,),
      ('NoMask', False))
  def testDynamicUnrollMatchesDynamicRNNWhenNoReset(self, with_mask):
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(3)
    batch_size = 4
    max_time = 7
    inputs = tf.random.uniform((batch_size, max_time, 2), dtype=tf.float32)
    layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
    if with_mask:
      reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
    else:
      reset_mask = None
    outputs_dun, final_state_dun = layer(inputs, reset_mask=reset_mask)
    outputs_drnn, final_state_drnn = tf.compat.v1.nn.dynamic_rnn(
        cell, inputs, dtype=tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    outputs_dun, final_state_dun, outputs_drnn, final_state_drnn = (
        self.evaluate(
            (outputs_dun, final_state_dun, outputs_drnn, final_state_drnn)))
    self.assertAllClose(outputs_dun, outputs_drnn)
    self.assertAllClose(final_state_dun, final_state_drnn)

  @parameterized.named_parameters(
      ('WithMask', True,),
      ('NoMask', False))
  def testDynamicUnrollMatchesDynamicRNNWhenNoResetSingleTimeStep(
      self, with_mask):
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(3)
    batch_size = 4
    max_time = 1
    inputs = tf.random.uniform((batch_size, max_time, 2), dtype=tf.float32)
    layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
    if with_mask:
      reset_mask = tf.zeros((batch_size, max_time), dtype=tf.bool)
    else:
      reset_mask = None
    outputs_dun, final_state_dun = layer(inputs, reset_mask=reset_mask)
    outputs_drnn, final_state_drnn = tf.compat.v1.nn.dynamic_rnn(
        cell, inputs, dtype=tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    outputs_dun, final_state_dun, outputs_drnn, final_state_drnn = (
        self.evaluate(
            (outputs_dun, final_state_dun, outputs_drnn, final_state_drnn)))
    self.assertAllClose(outputs_dun, outputs_drnn)
    self.assertAllClose(final_state_dun, final_state_drnn)

  def testNoTimeDimensionMatchesSingleStep(self):
    cell = tf.keras.layers.LSTMCell(3)
    batch_size = 4
    max_time = 1
    inputs = tf.random.uniform((batch_size, max_time, 2), dtype=tf.float32)
    inputs_no_time = tf.squeeze(inputs, axis=1)
    layer = dynamic_unroll_layer.DynamicUnroll(cell, dtype=tf.float32)
    outputs, next_state = layer(inputs)
    outputs_squeezed_time = tf.squeeze(outputs, axis=1)
    outputs_no_time, next_state_no_time = layer(inputs_no_time)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    outputs_squeezed_time, next_state, outputs_no_time, next_state_no_time = (
        self.evaluate((outputs_squeezed_time, next_state,
                       outputs_no_time, next_state_no_time)))
    self.assertAllEqual(outputs_squeezed_time, outputs_no_time)
    self.assertAllEqual(next_state, next_state_no_time)

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
    outputs, final_state = layer(inputs, reset_mask=reset_mask)

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
