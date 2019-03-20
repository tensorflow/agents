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

"""Tests tf_agents.utils.network_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import utils


class NetworkUtilsTest(tf.test.TestCase):

  def test_flatten_and_unflatten_ops(self):
    batch_squash = utils.BatchSquash(2)

    tensor = tf.constant(0, shape=(5, 4, 3, 2, 1))
    flat = batch_squash.flatten(tensor)
    unflat = batch_squash.unflatten(flat)

    self.assertAllEqual((20, 3, 2, 1), flat.shape)
    self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_and_unflatten_ops_no_batch_dims(self):
    batch_squash = utils.BatchSquash(0)

    tensor = tf.constant(0, shape=(5, 4, 3, 2, 1))
    flat = batch_squash.flatten(tensor)
    unflat = batch_squash.unflatten(flat)

    self.assertAllEqual((1, 5, 4, 3, 2, 1), flat.shape)
    self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_and_unflatten_ops_one_batch_dims(self):
    batch_squash = utils.BatchSquash(1)

    tensor = tf.constant(0, shape=(5, 4, 3, 2, 1))
    flat = batch_squash.flatten(tensor)
    unflat = batch_squash.unflatten(flat)

    self.assertAllEqual((5, 4, 3, 2, 1), flat.shape)
    self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

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

    self.assertAllEqual([tf.keras.layers.Conv2D, tf.keras.layers.Conv2D,
                         tf.keras.layers.Flatten, tf.keras.layers.Dense,
                         tf.keras.layers.Dropout, tf.keras.layers.Dense,
                         tf.keras.layers.Dropout],
                        [type(layer) for layer in layers])

    layers = utils.mlp_layers(conv_layer_params=[(3, 4, 5), (4, 6, 8)],
                              fc_layer_params=[10, 20],
                              activation_fn=tf.keras.activations.tanh,
                              dropout_layer_params=[None, 0.3],
                              name='testnet')
    self.assertEqual(6, len(layers))

    self.assertAllEqual([tf.keras.layers.Conv2D, tf.keras.layers.Conv2D,
                         tf.keras.layers.Flatten, tf.keras.layers.Dense,
                         tf.keras.layers.Dense, tf.keras.layers.Dropout],
                        [type(layer) for layer in layers])

    layers = utils.mlp_layers(conv_layer_params=[(3, 4, 5), (4, 6, 8)],
                              fc_layer_params=[10, 20],
                              activation_fn=tf.keras.activations.tanh,
                              dropout_layer_params=[
                                  dict(rate=0.5, permanent=True), None],
                              name='testnet')
    self.assertEqual(6, len(layers))

    self.assertAllEqual([tf.keras.layers.Conv2D, tf.keras.layers.Conv2D,
                         tf.keras.layers.Flatten, tf.keras.layers.Dense,
                         tf.keras.layers.Lambda, tf.keras.layers.Dense],
                        [type(layer) for layer in layers])


if __name__ == '__main__':
  tf.test.main()
