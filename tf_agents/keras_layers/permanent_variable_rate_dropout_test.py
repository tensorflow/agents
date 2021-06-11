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

"""Tests for tf_agents.keras_layers.permanent_variable_rate_dropout."""

import tensorflow as tf

from tf_agents.keras_layers import permanent_variable_rate_dropout
from tf_agents.utils import test_utils


class PermanentVariableRateDropoutTest(test_utils.TestCase):

  def testPermanent(self):
    var = tf.Variable(0.5, dtype=tf.float32)
    def dropout_fn():
      return tf.identity(var)

    layer = permanent_variable_rate_dropout.PermanentVariableRateDropout(
        rate=dropout_fn, permanent=True)
    inputs = tf.reshape(tf.range(4 * 12, dtype=tf.float32), shape=(2, 2, 3, 4))
    out = layer(inputs)
    scaled = inputs * 2
    # All elements should be either zero or the scaled input.
    self.assertAllClose(out * (scaled - out), tf.zeros_like(inputs))

    out = layer(inputs, training=False)
    self.assertAllClose(out * (scaled - out), tf.zeros_like(inputs))

    var.assign(0.3)
    out = layer(inputs)
    scaled = inputs / 0.7
    self.assertAllClose(out * (scaled - out), tf.zeros_like(inputs))

  def testNonPermanent(self):
    var = tf.Variable(0.5, dtype=tf.float32)
    def dropout_fn():
      return tf.identity(var)

    layer = permanent_variable_rate_dropout.PermanentVariableRateDropout(
        rate=dropout_fn)
    inputs = tf.reshape(tf.range(4 * 12, dtype=tf.float32), shape=(2, 2, 3, 4))
    out = layer(inputs, training=True)
    scaled = inputs * 2
    # All elements should be either zero or the scaled input.
    self.assertAllClose(out * (scaled - out), tf.zeros_like(inputs))

    out = layer(inputs, training=False)
    self.assertAllClose(out, inputs)


if __name__ == '__main__':
  test_utils.main()
