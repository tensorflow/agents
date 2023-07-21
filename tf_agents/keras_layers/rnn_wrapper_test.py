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

"""Tests for tf_agents.keras_layers.rnn_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tf_agents.keras_layers import rnn_wrapper
from tf_agents.utils import test_utils


class RNNWrapperTest(test_utils.TestCase):

  def testWrapperBuild(self):
    wrapper = rnn_wrapper.RNNWrapper(
        tf.keras.layers.LSTM(3, return_state=True, return_sequences=True))
    # Make sure wrapper.build() works when no time dimension is passed in.
    wrapper.build((1, 4))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    variables = self.evaluate(wrapper.trainable_variables)
    self.assertLen(variables, 3)
    self.assertLen(wrapper.variables, 3)
    self.assertTrue(wrapper.trainable)
    wrapper.trainable = False
    self.assertFalse(wrapper.trainable)
    self.assertEmpty(wrapper.trainable_variables)
    self.assertLen(wrapper.variables, 3)

  def testWrapperCall(self):
    wrapper = rnn_wrapper.RNNWrapper(
        tf.keras.layers.LSTM(3, return_state=True, return_sequences=True))

    batch_size = 2
    input_depth = 5
    inputs = np.random.rand(batch_size, input_depth).astype(np.float32)

    # Make sure wrapper call works when no time dimension is passed in.
    outputs, next_state = wrapper(inputs)

    inputs_time_dim = tf.expand_dims(inputs, axis=1)
    outputs_time_dim, next_state_time_dim = wrapper(inputs_time_dim)
    outputs_time_dim = tf.squeeze(outputs_time_dim, axis=1)

    outputs_manual_state, next_state_manual_state = wrapper(
        inputs, wrapper.get_initial_state(inputs))

    self.evaluate(tf.compat.v1.global_variables_initializer())
    for out_variant in (outputs, outputs_time_dim, outputs_manual_state):
      self.assertEqual(out_variant.shape, (batch_size, 3))
    for state_variant in (next_state, next_state_time_dim,
                          next_state_manual_state):
      self.assertLen(state_variant, 2)
      self.assertEqual(state_variant[0].shape, (batch_size, 3))
      self.assertEqual(state_variant[1].shape, (batch_size, 3))

    self.assertAllClose(outputs, outputs_time_dim)
    self.assertAllClose(outputs, outputs_manual_state)
    self.assertAllClose(next_state, next_state_time_dim)
    self.assertAllClose(next_state, next_state_manual_state)

  def testCopy(self):
    wrapper = rnn_wrapper.RNNWrapper(
        tf.keras.layers.LSTM(3, return_state=True, return_sequences=True))
    clone = type(wrapper).from_config(wrapper.get_config())
    self.assertEqual(wrapper.wrapped_layer.dtype, clone.wrapped_layer.dtype)
    self.assertEqual(wrapper.wrapped_layer.units, clone.wrapped_layer.units)

  def testRequiredConstructorArgs(self):
    with self.assertRaisesRegex(NotImplementedError,
                                'with return_state==False'):
      rnn_wrapper.RNNWrapper(tf.keras.layers.LSTM(3, return_sequences=True))

    with self.assertRaisesRegex(NotImplementedError,
                                'with return_sequences==False'):
      rnn_wrapper.RNNWrapper(tf.keras.layers.LSTM(3, return_state=True))


if __name__ == '__main__':
  test_utils.main()
