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

"""Sample recurrent Critic network to use with DDPG agents."""

import gin
import tensorflow as tf
from tf_agents.networks import encoding_network
from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.networks import sequential_layer
from tf_agents.utils import nest_utils


@gin.configurable
class CriticRnnNetwork(network.Network):
  """Creates a recurrent Critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_preprocessing_layers=None,
               observation_preprocessing_combiner=None,
               observation_conv_layer_params=None,
               observation_fc_layer_params=(200,),
               action_fc_layer_params=(200,),
               joint_fc_layer_params=(100),
               lstm_size=(40,),
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               name='CriticRnnNetwork'):
    """Creates an instance of `CriticRnnNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each of type
        `tensor_spec.TensorSpec` representing the inputs.
      observation_preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      observation_preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      observation_conv_layer_params: Optional list of convolution layers
        parameters to apply to the observations, where each item is a
        length-three tuple indicating (filters, kernel_size, stride).
      observation_fc_layer_params: Optional list of fully_connected parameters,
        where each item is the number of units in the layer. This is applied
        after the observation convultional layer.
      action_fc_layer_params: Optional list of parameters for a fully_connected
        layer to apply to the actions, where each item is the number of units
        in the layer.
      joint_fc_layer_params: Optional list of parameters for a fully_connected
        layer to apply after merging observations and actions, where each item
        is the number of units in the layer.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.

    Returns:
      A tf.float32 Tensor of q-values.

    Raises:
      ValueError: If `action_spec` contains more than one
        item.
    """
    observation_spec, action_spec = input_tensor_spec

    kernel_initializer = tf.compat.v1.variance_scaling_initializer(
      scale=2.0, mode='fan_in', distribution='truncated_normal')

    obs_encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=observation_preprocessing_layers,
        preprocessing_combiner=observation_preprocessing_combiner,
        conv_layer_params=observation_conv_layer_params,
        fc_layer_params=observation_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        dtype=dtype,
        name='obs_encoding')

    if len(tf.nest.flatten(action_spec)) > 1:
      raise ValueError('Only a single action is supported by this network.')

    action_layers = sequential_layer.SequentialLayer(utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='action_encoding'))

    obs_encoding_spec = tf.TensorSpec(
        shape=(observation_fc_layer_params[-1],), dtype=tf.float32)
    lstm_encoder = lstm_encoding_network.LSTMEncodingNetwork(
        input_tensor_spec=(obs_encoding_spec, action_spec),
        preprocessing_layers=(tf.keras.layers.Flatten(), action_layers),
        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
        input_fc_layer_params=joint_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        dtype=dtype,
        name='lstm')

    output_layers = [
        tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value')]

    super(CriticRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        name=name)

    self._obs_encoder = obs_encoder
    self._lstm_encoder = lstm_encoder
    self._output_layers = output_layers

  # TODO(kbanoop): Standardize argument names across different networks.
  def call(self, inputs, step_type, network_state=None):
    outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)
    batch_squash = utils.BatchSquash(outer_rank)

    observation, action = inputs
    observation, _ = self._obs_encoder(
        observation,
        step_type=step_type,
        network_state=network_state)

    observation = batch_squash.flatten(observation)
    action = tf.cast(tf.nest.flatten(action)[0], tf.float32)
    action = batch_squash.flatten(action)

    output, network_state = self._lstm_encoder(
        inputs=(observation, action),
        step_type=step_type,
        network_state=network_state)

    for layer in self._output_layers:
      output = layer(output)

    q_value = tf.reshape(output, [-1])
    q_value = batch_squash.unflatten(q_value)

    return q_value, network_state
