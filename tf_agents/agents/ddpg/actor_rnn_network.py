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

"""Sample recurrent Actor network to use with DDPG agents.

Note: This network scales actions to fit the given spec by using `tanh`. Due to
the nature of the `tanh` function, actions near the spec bounds cannot be
returned.
"""

import gin
import tensorflow as tf
from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common
from tf_agents.utils import nest_utils


# TODO(kbanoop): Reduce code duplication with other actor networks.
@gin.configurable
class ActorRnnNetwork(network.Network):
  """Creates a recurrent actor network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(200, 100),
               input_dropout_layer_params=None,
               lstm_size=(40,),
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               name='ActorRnnNetwork'):
    """Creates an instance of `ActorRnnNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the actions.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
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
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of input_fc_layer_params, or
        be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.

    Returns:
      A nest of action tensors matching the action_spec.

    Raises:
      ValueError: If 'input_dropout_layer_params' is not None.
    """
    if input_dropout_layer_params:
      raise ValueError('Dropout layer is not supported.')

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

    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    action_layers = [
        tf.keras.layers.Dense(
            single_action_spec.shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='action') for single_action_spec in flat_action_spec
    ]

    super(ActorRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        name=name)

    self._output_tensor_spec = output_tensor_spec
    self._flat_action_spec = flat_action_spec

    self._lstm_encoder = lstm_encoder
    self._action_layers = action_layers


  # TODO(kbanoop): Standardize argument names across different networks.
  def call(self, observation, step_type, network_state=None):
    outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)  # Squash B, and T dims.

    observation, network_state = self._lstm_encoder(
      observation,
      step_type=step_type,
      network_state=network_state)

    states = batch_squash.flatten(observation)  # [B, T, ...] -> [BxT, ...]

    actions = []
    for layer, spec in zip(self._action_layers, self._flat_action_spec):
      action = layer(states)
      action = common.scale_to_spec(action, spec)
      action = batch_squash.unflatten(action)
      actions.append(action)

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, actions)
    return output_actions, network_state
