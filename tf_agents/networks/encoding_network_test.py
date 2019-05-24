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

"""Tests for tf_agents.networks.encoding_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import encoding_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import test_utils


class EncodingNetworkTest(test_utils.TestCase):

  def test_non_preprocessing_layers(self):
    input_spec = tensor_spec.TensorSpec((32, 32, 3), tf.float32)
    network = encoding_network.EncodingNetwork(
        input_spec,
        conv_layer_params=((16, 2, 1), (15, 2, 1)),
        fc_layer_params=(10, 5, 2),
        activation_fn=tf.keras.activations.tanh,
    )

    variables = network.variables
    self.assertEqual(10, len(variables))
    self.assertEqual(6, len(network.layers))

    # Validate first conv layer.
    config = network.layers[0].get_config()
    self.assertEqual('tanh', config['activation'])
    self.assertEqual((2, 2), config['kernel_size'])
    self.assertEqual(16, config['filters'])
    self.assertEqual((1, 1), config['strides'])
    self.assertTrue(config['trainable'])

    # Validate second conv layer.
    config = network.layers[1].get_config()
    self.assertEqual('tanh', config['activation'])
    self.assertEqual((2, 2), config['kernel_size'])
    self.assertEqual(15, config['filters'])
    self.assertEqual((1, 1), config['strides'])
    self.assertTrue(config['trainable'])

    # Validate flatten layer.
    config = network.layers[2].get_config()
    self.assertEqual('flatten', config['name'])

    # Validate dense layers.
    self.assertEqual(10, network.layers[3].get_config()['units'])
    self.assertEqual(5, network.layers[4].get_config()['units'])
    self.assertEqual(2, network.layers[5].get_config()['units'])

  def test_preprocessing_layer_no_combiner(self):
    network = encoding_network.EncodingNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([5], tf.float32),
        preprocessing_layers=tf.keras.layers.Lambda(lambda x: x),
        preprocessing_combiner=None,
        fc_layer_params=(2,))
    out, _ = network(tf.ones((3, 5)))
    self.assertAllEqual(out.shape.as_list(), [3, 2])

  def test_preprocessing_layers_no_combiner_error(self):
    with self.assertRaisesRegexp(ValueError, 'required'):
      encoding_network.EncodingNetwork(
          input_tensor_spec=[
              tensor_spec.TensorSpec([5], tf.float32),
              tensor_spec.TensorSpec([5], tf.float32)
          ],
          preprocessing_layers=[
              tf.keras.layers.Lambda(lambda x: x),
              tf.keras.layers.Lambda(lambda x: x)
          ],
          preprocessing_combiner=None,
          fc_layer_params=(2,))

  def test_error_raised_if_missing_preprocessing_layer(self):
    with self.assertRaisesRegexp(ValueError, 'sequence length'):
      encoding_network.EncodingNetwork(
          input_tensor_spec=[
              tensor_spec.TensorSpec([5], tf.float32),
              tensor_spec.TensorSpec([5], tf.float32)
          ],
          preprocessing_layers=[
              tf.keras.layers.Lambda(lambda x: x),
          ],
          preprocessing_combiner=None,
          fc_layer_params=(2,))

  def test_error_raised_extra_preprocessing_layer(self):
    with self.assertRaisesRegexp(ValueError, 'sequence length'):
      encoding_network.EncodingNetwork(
          input_tensor_spec=tensor_spec.TensorSpec([5], tf.float32),
          preprocessing_layers=[
              tf.keras.layers.Lambda(lambda x: x),
              tf.keras.layers.Lambda(lambda x: x)
          ],
          preprocessing_combiner=None,
          fc_layer_params=(2,))

  def test_dict_spec_and_pre_processing(self):
    input_spec = {
        'a': tensor_spec.TensorSpec((32, 32, 3), tf.float32),
        'b': tensor_spec.TensorSpec((32, 32, 3), tf.float32)
    }
    network = encoding_network.EncodingNetwork(
        input_spec,
        preprocessing_layers={
            'a': tf.keras.layers.Flatten(),
            'b': tf.keras.layers.Flatten()
        },
        fc_layer_params=(),
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        activation_fn=tf.keras.activations.tanh,
    )

    sample_input = tensor_spec.sample_spec_nest(input_spec)
    output, _ = network(sample_input)
    # 6144 is the shape from a concat of flat (32, 32, 3) x2.
    self.assertEqual((6144,), output.shape)


if __name__ == '__main__':
  tf.test.main()
