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

"""Tests for tf_agents.keras_layers.sequential_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import sequential_layer


class SequentialLayerTest(tf.test.TestCase):

  def testBuild(self):
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(4, use_bias=False),
         tf.keras.layers.ReLU()])
    inputs = np.ones((2, 3))
    out = sequential(inputs)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    out = self.evaluate(out)
    weights = self.evaluate(sequential.layers[0].weights[0])
    expected = np.dot(inputs, weights)
    expected[expected < 0] = 0
    self.assertAllClose(expected, out)

  def testTrainableVariables(self):
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(3), tf.keras.layers.Dense(4)])
    sequential.build((3, 2))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    variables = self.evaluate(sequential.trainable_variables)
    self.assertLen(variables, 4)
    self.assertLen(sequential.variables, 4)
    self.assertTrue(sequential.trainable)
    sequential.trainable = False
    self.assertFalse(sequential.trainable)
    self.assertEmpty(sequential.trainable_variables)
    self.assertLen(sequential.variables, 4)

  def testCopy(self):
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(3), tf.keras.layers.Dense(4, use_bias=False)])
    clone = type(sequential).from_config(sequential.get_config())
    self.assertLen(clone.layers, 2)
    for l1, l2 in zip(sequential.layers, clone.layers):
      self.assertEqual(l1.dtype, l2.dtype)
      self.assertEqual(l1.units, l2.units)
      self.assertEqual(l1.use_bias, l2.use_bias)


if __name__ == '__main__':
  tf.test.main()
