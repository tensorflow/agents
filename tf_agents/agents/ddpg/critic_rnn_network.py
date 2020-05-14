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
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

KERAS_LSTM_FUSED_IMPLEMENTATION = 2


@gin.configurable
class CriticRnnNetwork(network.Network):
  """Creates a recurrent Critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_conv_layer_params=None,
               observation_fc_layer_params=(200,),
               action_fc_layer_params=(200,),
               joint_fc_layer_params=(100,),
               lstm_size=None,
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               last_kernel_initializer=None,
               rnn_construction_fn=None,
               rnn_construction_kwargs=None,
               name='CriticRnnNetwork'):
    """Creates an instance of `CriticRnnNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each of type
        `tensor_spec.TensorSpec` representing the inputs.
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
      kernel_initializer: kernel initializer for all layers except for the value
        regression layer. If None, a VarianceScaling initializer will be used.
      last_kernel_initializer: kernel initializer for the value regression layer
        . If None, a RandomUniform initializer will be used.
      rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
        tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
        provide both rnn_construction_fn and lstm_size.
      rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
        rnn_construction_fn.

        The RNN will be constructed via:

        ```
        rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
        ```
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        item.
      ValueError: If neither `lstm_size` nor `rnn_construction_fn` are provided.
      ValueError: If both `lstm_size` and `rnn_construction_fn` are provided.
    """
    if lstm_size is None and rnn_construction_fn is None:
      raise ValueError('Need to provide either custom rnn_construction_fn or '
                       'lstm_size.')
    if lstm_size and rnn_construction_fn:
      raise ValueError('Cannot provide both custom rnn_construction_fn and '
                       'lstm_size.')

    observation_spec, action_spec = input_tensor_spec

    if len(tf.nest.flatten(observation_spec)) > 1:
      raise ValueError(
          'Only a single observation is supported by this network.')

    if len(tf.nest.flatten(action_spec)) > 1:
      raise ValueError('Only a single action is supported by this network.')

    if kernel_initializer is None:
      kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform')
    if last_kernel_initializer is None:
      last_kernel_initializer = tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003)

    observation_layers = utils.mlp_layers(
        observation_conv_layer_params,
        observation_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        name='observation_encoding')

    action_layers = utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        name='action_encoding')

    joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        name='joint_mlp')

    # Create RNN cell
    if rnn_construction_fn:
      rnn_construction_kwargs = rnn_construction_kwargs or {}
      lstm_network = rnn_construction_fn(**rnn_construction_kwargs)
    else:
      if len(lstm_size) == 1:
        cell = tf.keras.layers.LSTMCell(lstm_size[0])
      else:
        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(size) for size in lstm_size])
      lstm_network = dynamic_unroll_layer.DynamicUnroll(cell)

    counter = [-1]

    def create_spec(size):
      counter[0] += 1
      return tensor_spec.TensorSpec(
          size, dtype=tf.float32, name='network_state_%d' % counter[0])

    state_spec = tf.nest.map_structure(create_spec,
                                       lstm_network.cell.state_size)

    output_layers = utils.mlp_layers(fc_layer_params=output_fc_layer_params,
                                     name='output')

    output_layers.append(
        tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=last_kernel_initializer,
            name='value'))

    super(CriticRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name=name)

    self._observation_layers = observation_layers
    self._action_layers = action_layers
    self._joint_layers = joint_layers
    self._lstm_network = lstm_network
    self._output_layers = output_layers

  # TODO(kbanoop): Standardize argument names across different networks.
  def call(self, inputs, step_type, network_state=(), training=False):
    observation, action = inputs
    observation_spec, _ = self.input_tensor_spec
    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               observation_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          'Input observation must have a batch or batch x time outer shape.')

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                          observation)
      action = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), action)
      step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                        step_type)

    observation = tf.cast(tf.nest.flatten(observation)[0], tf.float32)
    action = tf.cast(tf.nest.flatten(action)[0], tf.float32)

    batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
    observation = batch_squash.flatten(observation)  # [B, T, ...] -> [BxT, ...]
    action = batch_squash.flatten(action)

    for layer in self._observation_layers:
      observation = layer(observation, training=training)

    for layer in self._action_layers:
      action = layer(action, training=training)

    joint = tf.concat([observation, action], -1)
    for layer in self._joint_layers:
      joint = layer(joint, training=training)

    joint = batch_squash.unflatten(joint)  # [B x T, ...] -> [B, T, ...]

    network_kwargs = {}
    if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
      network_kwargs['reset_mask'] = tf.equal(step_type,
                                              time_step.StepType.FIRST,
                                              name='mask')

    # Unroll over the time sequence.
    output = self._lstm_network(
        inputs=joint,
        initial_state=network_state,
        training=training,
        **network_kwargs)
    if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
      joint, network_state = output
    else:
      joint = output[0]
      network_state = tf.nest.pack_sequence_as(
          self._lstm_network.cell.state_size, tf.nest.flatten(output[1:]))

    output = batch_squash.flatten(joint)  # [B, T, ...] -> [B x T, ...]

    for layer in self._output_layers:
      output = layer(output, training=training)

    q_value = tf.reshape(output, [-1])
    q_value = batch_squash.unflatten(q_value)  # [B x T, ...] -> [B, T, ...]
    if not has_time_dim:
      q_value = tf.squeeze(q_value, axis=1)

    return q_value, network_state
