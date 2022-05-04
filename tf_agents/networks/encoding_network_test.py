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

"""Tests for tf_agents.networks.encoding_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.networks import encoding_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import test_utils

keras_preprocessing = tf.keras.layers.experimental.preprocessing


class EncodingNetworkTest(test_utils.TestCase, parameterized.TestCase):

  def test_empty_layers(self):
    input_spec = tensor_spec.TensorSpec((2, 3), tf.float32)
    network = encoding_network.EncodingNetwork(input_spec,)

    # Only one layer to flatten input.
    self.assertLen(network.layers, 1)
    config = network.layers[0].get_config()
    self.assertEqual('flatten', config['name'])

    out, _ = network(tf.ones((1, 2, 3)))
    self.assertAllEqual(out, [[1, 1, 1, 1, 1, 1]])
    self.assertEmpty(network.variables)

  def test_non_preprocessing_layers_2d(self):
    input_spec = tensor_spec.TensorSpec((32, 32, 3), tf.float32)
    network = encoding_network.EncodingNetwork(
        input_spec,
        conv_layer_params=((16, 2, 1), (15, 2, 1)),
        fc_layer_params=(10, 5, 2),
        activation_fn=tf.keras.activations.tanh,
    )

    network.create_variables()

    variables = network.variables
    self.assertLen(variables, 10)
    self.assertLen(network.layers, 6)

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

  def test_non_preprocessing_layers_1d(self):
    input_spec = tensor_spec.TensorSpec((32, 3), tf.float32)
    network = encoding_network.EncodingNetwork(
        input_spec,
        conv_layer_params=((16, 2, 1), (15, 2, 1)),
        fc_layer_params=(10, 5, 2),
        activation_fn=tf.keras.activations.tanh,
        conv_type='1d',
    )

    network.create_variables()

    variables = network.variables
    self.assertLen(variables, 10)
    self.assertLen(network.layers, 6)

    # Validate first conv layer.
    config = network.layers[0].get_config()
    self.assertEqual('tanh', config['activation'])
    self.assertEqual((2,), config['kernel_size'])
    self.assertEqual(16, config['filters'])
    self.assertEqual((1,), config['strides'])
    self.assertTrue(config['trainable'])

    # Validate second conv layer.
    config = network.layers[1].get_config()
    self.assertEqual('tanh', config['activation'])
    self.assertEqual((2,), config['kernel_size'])
    self.assertEqual(15, config['filters'])
    self.assertEqual((1,), config['strides'])
    self.assertTrue(config['trainable'])

  def test_conv_raise_error(self):
    input_spec = tensor_spec.TensorSpec((32, 3), tf.float32)
    with self.assertRaises(ValueError):
      _ = encoding_network.EncodingNetwork(
          input_spec,
          conv_layer_params=((16, 2, 1), (15, 2, 1)),
          fc_layer_params=(10, 5, 2),
          activation_fn=tf.keras.activations.tanh,
          conv_type='3d')

  def test_conv_dilation_params(self):
    with self.subTest(name='no dilations'):
      input_spec = tensor_spec.TensorSpec((32, 32, 3), tf.float32)
      network = encoding_network.EncodingNetwork(
          input_spec,
          conv_layer_params=((16, 2, 1), (15, 2, 1)),
      )

      network.create_variables()
      variables = network.variables

      self.assertLen(variables, 4)
      self.assertLen(network.layers, 3)

      # Validate dilation rates
      config = network.layers[0].get_config()
      self.assertEqual((1, 1), config['dilation_rate'])
      config = network.layers[1].get_config()
      self.assertEqual((1, 1), config['dilation_rate'])

    with self.subTest(name='dilations'):
      input_spec = tensor_spec.TensorSpec((32, 32, 3), tf.float32)
      network = encoding_network.EncodingNetwork(
          input_spec,
          conv_layer_params=((16, 2, 1, 2), (15, 2, 1, (2, 4))),
      )

      network.create_variables()
      variables = network.variables

      self.assertLen(variables, 4)
      self.assertLen(network.layers, 3)

      # Validate dilation rates
      config = network.layers[0].get_config()
      self.assertEqual((2, 2), config['dilation_rate'])
      config = network.layers[1].get_config()
      self.assertEqual((2, 4), config['dilation_rate'])

    with self.subTest(name='failing conv spec'):
      input_spec = tensor_spec.TensorSpec((32, 32, 3), tf.float32)
      with self.assertRaises(ValueError):
        network = encoding_network.EncodingNetwork(
            input_spec,
            conv_layer_params=((16, 2, 1, 2, 4), (15, 2, 1)),
            )
      with self.assertRaises(ValueError):
        network = encoding_network.EncodingNetwork(
            input_spec,
            conv_layer_params=((16, 2, 1), (15, 2)),
            )

  def test_preprocessing_layer_no_combiner(self):
    network = encoding_network.EncodingNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([5], tf.float32),
        preprocessing_layers=tf.keras.layers.Lambda(lambda x: x),
        preprocessing_combiner=None,
        fc_layer_params=(2,))
    out, _ = network(tf.ones((3, 5)))
    self.assertAllEqual(out.shape.as_list(), [3, 2])

  def test_preprocessing_layers_no_combiner_error(self):
    with self.assertRaisesRegex(ValueError, 'required'):
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
    with self.assertRaisesRegex(ValueError, 'sequence length'):
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
    with self.assertRaisesRegex(ValueError, 'sequence length'):
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
            'a':
                tf.keras.Sequential([
                    tf.keras.layers.Dense(4, activation='tanh'),
                    tf.keras.layers.Flatten()
                ]),
            'b':
                tf.keras.layers.Flatten()
        },
        fc_layer_params=(),
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        activation_fn=tf.keras.activations.tanh,
    )

    sample_input = tensor_spec.sample_spec_nest(input_spec)
    output, _ = network(sample_input)
    # 6144 is the shape from a concat of flat (32, 32, 3) x2.
    self.assertEqual((7168,), output.shape)

  def test_layers_buildable(self):
    input_spec = {
        'a': tensor_spec.TensorSpec((32, 32, 3), tf.float32),
        'b': tensor_spec.TensorSpec((32, 32, 3), tf.float32)
    }
    network = encoding_network.EncodingNetwork(
        input_spec,
        preprocessing_layers={
            'a':
                tf.keras.Sequential([
                    tf.keras.layers.Dense(4, activation='tanh'),
                    tf.keras.layers.Flatten()
                ]),
            'b':
                tf.keras.layers.Flatten()
        },
        fc_layer_params=(),
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        activation_fn=tf.keras.activations.tanh,
    )
    network.create_variables()
    self.assertNotEmpty(network.variables)

  def testDenseFeaturesV1RaisesError(self):
    key = 'feature_key'
    state_dims = 5
    column = tf.feature_column.numeric_column(key, [state_dims])
    input_spec = {key: tensor_spec.TensorSpec([state_dims], tf.int32)}
    dense_features = tf.compat.v1.keras.layers.DenseFeatures([column])
    with self.assertRaisesRegex(ValueError, 'DenseFeatures'):
      encoding_network.EncodingNetwork(
          input_spec, preprocessing_combiner=dense_features)

  def testNumericKerasInput(self):
    key = 'feature_key'
    batch_size = 3
    state_dims = 5
    input_shape = (batch_size, state_dims)
    keras_input = tf.keras.Input(shape=[state_dims], dtype=tf.int32, name=key)
    state = {key: tf.ones(input_shape, tf.int32)}
    input_spec = {key: tensor_spec.TensorSpec([state_dims], tf.int32)}

    network = encoding_network.EncodingNetwork(
        input_spec, preprocessing_combiner=tf.keras.Sequential([keras_input]))

    output, _ = network(state)
    self.assertEqual(input_shape, output.shape)

  def testKerasIntegerLookup(self):
    if not tf.executing_eagerly():
      self.skipTest('This test is TF2 only.')

    key = 'feature_key'
    vocab_list = [2, 3, 4]

    keras_input = tf.keras.Input(shape=(1,), name=key, dtype=tf.dtypes.int32)
    id_input = keras_preprocessing.IntegerLookup(
        vocabulary=vocab_list, num_oov_indices=0, output_mode='multi_hot')

    state_input = [3, 2, 2, 4, 3]
    state = {key: tf.expand_dims(state_input, -1)}
    input_spec = {key: tensor_spec.TensorSpec([1], tf.int32)}

    network = encoding_network.EncodingNetwork(
        input_spec,
        preprocessing_combiner=tf.keras.Sequential([keras_input, id_input]))

    output, _ = network(state)
    expected_shape = (len(state_input), len(vocab_list))
    self.assertEqual(expected_shape, output.shape)

  def testCombinedKerasPreprocessingLayers(self):
    if not tf.executing_eagerly():
      self.skipTest('This test is TF2 only.')

    inputs = {}
    features = {}
    tensors = {}
    specs = {}
    expected_dim = 0

    indicator_key = 'indicator_key'
    vocab_list = [2, 3, 4]
    inputs[indicator_key] = tf.keras.Input(
        shape=(1,), dtype=tf.dtypes.int32, name=indicator_key)
    features[indicator_key] = keras_preprocessing.IntegerLookup(
        vocabulary=vocab_list, num_oov_indices=0, output_mode='multi_hot')(
            inputs[indicator_key])
    state_input = [3, 2, 2, 4, 3]
    tensors[indicator_key] = tf.expand_dims(state_input, -1)
    specs[indicator_key] = tensor_spec.TensorSpec([1], tf.int32)
    expected_dim += len(vocab_list)

    embedding_key = 'embedding_key'
    embedding_dim = 3
    vocab_list = [2, 3, 4]
    inputs[embedding_key] = tf.keras.Input(
        shape=(1,), dtype=tf.dtypes.int32, name=embedding_key)
    id_input = keras_preprocessing.IntegerLookup(
        vocabulary=vocab_list, num_oov_indices=0)(
            inputs[embedding_key])
    embedding_input = tf.keras.layers.Embedding(
        input_dim=len(vocab_list),
        output_dim=embedding_dim)(id_input)
    features[embedding_key] = tf.reduce_sum(embedding_input, axis=-2)
    state_input = [3, 2, 2, 4, 3]
    tensors[embedding_key] = tf.expand_dims(state_input, -1)
    specs[embedding_key] = tensor_spec.TensorSpec([1], tf.int32)
    expected_dim += embedding_dim

    numeric_key = 'numeric_key'
    batch_size = 5
    state_dims = 3
    input_shape = (batch_size, state_dims)
    inputs[numeric_key] = tf.keras.Input(
        shape=[state_dims], dtype=tf.float32, name=numeric_key)
    features[numeric_key] = inputs[numeric_key]
    tensors[numeric_key] = tf.ones(input_shape, tf.float32)
    specs[numeric_key] = tensor_spec.TensorSpec([state_dims], tf.float32)
    expected_dim += state_dims
    features = tf.keras.layers.concatenate(features.values(), axis=-1)

    # TODO(b/170645185): Replace Model with FunctionalPreprocessingStage.
    network = encoding_network.EncodingNetwork(
        specs,
        preprocessing_combiner=tf.keras.Model(inputs=inputs, outputs=features))
    output, _ = network(tensors)
    expected_shape = (batch_size, expected_dim)
    self.assertEqual(expected_shape, output.shape)

  @parameterized.named_parameters(
      ('TrainingTrue', True,),
      ('TrainingFalse', False))
  def testDropoutFCLayers(self, training):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    network = encoding_network.EncodingNetwork(
        obs_spec,
        fc_layer_params=[20],
        dropout_layer_params=[0.5])
    obs = tf.random.uniform([batch_size, num_obs_dims])
    output1, _ = network(obs, training=training)
    output2, _ = network(obs, training=training)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    output1, output2 = self.evaluate([output1, output2])
    if training:
      self.assertGreater(np.linalg.norm(output1 - output2), 0)
    else:
      self.assertAllEqual(output1, output2)

  def testWeightDecay(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    network = encoding_network.EncodingNetwork(
        obs_spec,
        fc_layer_params=[20],
        weight_decay_params=[0.5])
    obs = tf.random.uniform([batch_size, num_obs_dims])
    network(obs)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    regularization_loss = self.evaluate(network.losses[0])
    self.assertGreater(regularization_loss, 0)


if __name__ == '__main__':
  tf.test.main()
