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

"""Tests for tf_agents.keras_layers.squashed_outer_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.keras_layers import squashed_outer_wrapper
from tf_agents.utils import common
from tf_agents.utils import test_utils


class SquashedOuterWrapperTest(test_utils.TestCase):

  def testFromConfigBatchNorm(self):
    l1 = squashed_outer_wrapper.SquashedOuterWrapper(
        tf.keras.layers.BatchNormalization(axis=-1), inner_rank=3)
    l2 = squashed_outer_wrapper.SquashedOuterWrapper.from_config(
        l1.get_config())
    self.assertEqual(l1.get_config(), l2.get_config())

  def testSquashedOuterWrapperSimple(self):
    bn = tf.keras.layers.BatchNormalization(axis=-1)
    layer = squashed_outer_wrapper.SquashedOuterWrapper(bn, inner_rank=3)

    inputs_flat = tf.range(3 * 4 * 5 * 6 * 7, dtype=tf.float32)
    inputs_2_batch = tf.reshape(inputs_flat, [3, 4, 5, 6, 7])
    outputs_2_batch = layer(inputs_2_batch)

    inputs_1_batch = tf.reshape(inputs_flat, [3 * 4, 5, 6, 7])
    outputs_1_batch = layer(inputs_1_batch)
    outputs_1_batch_reshaped = tf.reshape(outputs_1_batch, [3, 4, 5, 6, 7])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(outputs_2_batch), self.evaluate(outputs_1_batch_reshaped))

  def testIncompatibleShapes(self):
    bn = tf.keras.layers.BatchNormalization(axis=-1)
    layer = squashed_outer_wrapper.SquashedOuterWrapper(bn, inner_rank=3)

    with self.assertRaisesRegex(ValueError, 'must have known rank'):
      fn = common.function(layer)
      fn.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.float32))


if __name__ == '__main__':
  test_utils.main()
