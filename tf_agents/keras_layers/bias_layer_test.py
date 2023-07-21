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

"""Tests for tf_agents.keras_layers.bias_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import bias_layer


class BiasLayerTest(tf.test.TestCase):

  def testBuild(self):
    bias = bias_layer.BiasLayer()
    states = tf.ones((2, 3))
    out = bias(states)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np.testing.assert_almost_equal([[1.0] * 3] * 2, self.evaluate(out))

  def testBuildScalar(self):
    bias = bias_layer.BiasLayer()
    states = tf.ones((2,))
    out = bias(states)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np.testing.assert_almost_equal([1.0] * 2, self.evaluate(out))

  def testTrainableVariables(self):
    bias = bias_layer.BiasLayer(
        bias_initializer=tf.constant_initializer(value=1.0))
    states = tf.zeros((2, 3))
    _ = bias(states)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    variables = bias.trainable_variables
    np.testing.assert_almost_equal([[1.0] * 3], self.evaluate(variables))


if __name__ == '__main__':
  tf.test.main()
