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

"""Sample Keras Value Network with LSTM cells .

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM   # lstm_cell_params
  [optional]: Dense  # output_fc_layer_params
  Dense -> 1         # Value output
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network


@gin.configurable
class ValueRnnNetwork(network.Network):
  """Recurrent value network. Reduces to 1 value output per batch item."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(75, 40),
               input_dropout_layer_params=None,
               lstm_size=(40,),
               output_fc_layer_params=(75, 40),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               name='ValueRnnNetwork'):
    """Creates an instance of `ValueRnnNetwork`.

    Network supports calls with shape outer_rank + input_tensor_shape.shape.
    Note outer_rank must be at least 1.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them.  Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        where each item is the fraction of input units to drop. The dropout
        layers are interleaved with the fully connected layers; there is a
        dropout layer after each fully connected layer, except if the entry in
        the list is None. This list must have the same length of
        input_fc_layer_params, or be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      dtype: The dtype to use by the convolution, LSTM, and fully connected
        layers.
      name: A string representing name of the network.
    """
    del input_dropout_layer_params

    lstm_encoder = lstm_encoding_network.LSTMEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        dtype=dtype,
        name=name)

    postprocessing_layers = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03))

    super(ValueRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        name=name)

    self._lstm_encoder = lstm_encoder
    self._postprocessing_layers = postprocessing_layers

  def call(self,
           observation,
           step_type=None,
           network_state=(),
           training=False):
    state, network_state = self._lstm_encoder(
        observation, step_type=step_type, network_state=network_state,
        training=training)
    value = self._postprocessing_layers(state, training=training)
    return tf.squeeze(value, -1), network_state
