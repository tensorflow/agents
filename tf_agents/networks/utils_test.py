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

"""Tests tf_agents.utils.network_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import permanent_variable_rate_dropout
from tf_agents.networks import utils


def _to_dense(st):
  return tf.scatter_nd(st.indices, st.values, st.dense_shape)


class NetworkUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(NetworkUtilsTest, self).setUp()
    self._tensors = [
        tf.constant(0, shape=(5, 4, 3, 2, 1)),
        tf.SparseTensor(
            indices=tf.zeros((0, 5), dtype=tf.int64),
            values=tf.zeros((0,), dtype=tf.float32),
            dense_shape=tf.constant([5, 4, 3, 2, 1], dtype=tf.int64))
    ]

  def test_flatten_and_unflatten_ops(self):
    batch_squash = utils.BatchSquash(2)
    for tensor in self._tensors:
      flat = batch_squash.flatten(tensor)
      unflat = batch_squash.unflatten(flat)
      self.assertAllEqual((20, 3, 2, 1), flat.shape)
      self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_and_unflatten_ops_no_batch_dims(self):
    batch_squash = utils.BatchSquash(0)

    for tensor in self._tensors:
      flat = batch_squash.flatten(tensor)
      unflat = batch_squash.unflatten(flat)

      self.assertAllEqual((1, 5, 4, 3, 2, 1), flat.shape)
      self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_and_unflatten_ops_one_batch_dims(self):
    batch_squash = utils.BatchSquash(1)

    for tensor in self._tensors:
      flat = batch_squash.flatten(tensor)
      unflat = batch_squash.unflatten(flat)

      self.assertAllEqual((5, 4, 3, 2, 1), flat.shape)
      self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_undefined_shapes_in_tffun(self):
    batch_squash = utils.BatchSquash(2)

    @tf.function
    def tf_fn():
      # Construct a tensor with some unknown shapes.
      x_default = tf.random.uniform([2, 2, 128, 128, 3])
      input_tensor = tf.compat.v1.placeholder_with_default(
          x_default, shape=[2, 2, None, None, None])
      flat = batch_squash.flatten(input_tensor)

      self.assertAllEqual((4, None, None, None), flat.shape)

    tf_fn()

  def test_flatten_undefined_shapes_in_tffun_sparse_tensor_undefined(self):
    batch_squash = utils.BatchSquash(2)

    @tf.function
    def tf_fn():
      # Construct a SparseTensor with some unknown shapes.
      x_default = tf.random.uniform([2, 2, 128, 128, 3])
      input_dense = tf.compat.v1.placeholder_with_default(
          x_default, shape=[2, 2, None, None, None])
      input_sparse = tf.sparse.from_dense(input_dense)
      flat = batch_squash.flatten(input_sparse)

      # SparseTensors are still fully undefined, as setting the known part of
      # the shape is not implemented in flatten. If implemented, this assertion
      # should be changed to (4, None, None, None).
      self.assertAllEqual((None, None, None, None), flat.shape)

    tf_fn()

  def test_mlp_layers(self):
    layers = utils.mlp_layers(conv_layer_params=[(3, 4, 5), (4, 6, 8)],
                              fc_layer_params=[10, 20],
                              activation_fn=tf.keras.activations.tanh,
                              name='testnet')
    self.assertEqual(5, len(layers))

    self.assertAllEqual([tf.keras.layers.Conv2D, tf.keras.layers.Conv2D,
                         tf.keras.layers.Flatten, tf.keras.layers.Dense,
                         tf.keras.layers.Dense],
                        [type(layer) for layer in layers])

    layers = utils.mlp_layers(conv_layer_params=[(3, 4, 5), (4, 6, 8)],
                              fc_layer_params=[10, 20],
                              activation_fn=tf.keras.activations.tanh,
                              dropout_layer_params=[0.5, 0.3],
                              name='testnet')
    self.assertEqual(7, len(layers))

    self.assertAllEqual([
        tf.keras.layers.Conv2D, tf.keras.layers.Conv2D, tf.keras.layers.Flatten,
        tf.keras.layers.Dense,
        permanent_variable_rate_dropout.PermanentVariableRateDropout,
        tf.keras.layers.Dense,
        permanent_variable_rate_dropout.PermanentVariableRateDropout
    ], [type(layer) for layer in layers])

    layers = utils.mlp_layers(conv_layer_params=[(3, 4, 5), (4, 6, 8)],
                              fc_layer_params=[10, 20],
                              activation_fn=tf.keras.activations.tanh,
                              dropout_layer_params=[None, 0.3],
                              name='testnet')
    self.assertEqual(6, len(layers))

    self.assertAllEqual([
        tf.keras.layers.Conv2D, tf.keras.layers.Conv2D, tf.keras.layers.Flatten,
        tf.keras.layers.Dense, tf.keras.layers.Dense,
        permanent_variable_rate_dropout.PermanentVariableRateDropout
    ], [type(layer) for layer in layers])

    layers = utils.mlp_layers(conv_layer_params=[(3, 4, 5), (4, 6, 8)],
                              fc_layer_params=[10, 20],
                              activation_fn=tf.keras.activations.tanh,
                              dropout_layer_params=[
                                  dict(rate=0.5, permanent=True), None],
                              name='testnet')
    self.assertEqual(6, len(layers))

    self.assertAllEqual([
        tf.keras.layers.Conv2D, tf.keras.layers.Conv2D, tf.keras.layers.Flatten,
        tf.keras.layers.Dense,
        permanent_variable_rate_dropout.PermanentVariableRateDropout,
        tf.keras.layers.Dense
    ], [type(layer) for layer in layers])


if __name__ == '__main__':
  tf.test.main()
