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

import functools
import gin
import tensorflow as tf
from tf_agents.networks import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils


@gin.configurable
class CriticRnnNetwork(network.Network):
  """Creates a recurrent Critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_conv_layer_params=None,
               observation_fc_layer_params=(200,),
               action_fc_layer_params=(200,),
               joint_fc_layer_params=(100),
               lstm_size=(40,),
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
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
      name: A string representing name of the network.

    Returns:
      A tf.float32 Tensor of q-values.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        item.
    """
    observation_spec, action_spec = input_tensor_spec

    if len(tf.nest.flatten(observation_spec)) > 1:
      raise ValueError(
          'Only a single observation is supported by this network.')

    if len(tf.nest.flatten(action_spec)) > 1:
      raise ValueError('Only a single action is supported by this network.')

    observation_layers = utils.mlp_layers(
        observation_conv_layer_params,
        observation_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='observation_encoding')

    action_layers = utils.mlp_layers(
        None,
        action_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='action_encoding')

    joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='joint_mlp')

    # Create RNN cell
    if len(lstm_size) == 1:
      cell = tf.keras.layers.LSTMCell(lstm_size[0])
    else:
      cell = tf.keras.layers.StackedRNNCells(
          [tf.keras.layers.LSTMCell(size) for size in lstm_size])

    state_spec = tf.nest.map_structure(
        functools.partial(
            tensor_spec.TensorSpec, dtype=tf.float32,
            name='network_state_spec'), list(cell.state_size))

    output_layers = utils.mlp_layers(fc_layer_params=output_fc_layer_params,
                                     name='output')

    output_layers.append(
        tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='value'))

    super(CriticRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name=name)

    self._observation_layers = observation_layers
    self._action_layers = action_layers
    self._joint_layers = joint_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._output_layers = output_layers

  # TODO(kbanoop): Standardize argument names across different networks.
  def call(self, inputs, step_type, network_state=None):
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
      observation = layer(observation)

    for layer in self._action_layers:
      action = layer(action)

    joint = tf.concat([observation, action], -1)
    for layer in self._joint_layers:
      joint = layer(joint)

    joint = batch_squash.unflatten(joint)  # [B x T, ...] -> [B, T, ...]

    with tf.name_scope('reset_mask'):
      reset_mask = tf.equal(step_type, time_step.StepType.FIRST)
    # Unroll over the time sequence.
    joint, network_state = self._dynamic_unroll(
        joint,
        reset_mask,
        initial_state=network_state)

    output = batch_squash.flatten(joint)  # [B, T, ...] -> [B x T, ...]

    for layer in self._output_layers:
      output = layer(output)

    q_value = tf.reshape(output, [-1])
    q_value = batch_squash.unflatten(q_value)  # [B x T, ...] -> [B, T, ...]
    if not has_time_dim:
      q_value = tf.squeeze(q_value, axis=1)

    return q_value, network_state
