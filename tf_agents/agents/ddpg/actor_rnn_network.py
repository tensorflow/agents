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

import functools
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import common
from tf_agents.utils import nest_utils


# TODO(kbanoop): Reduce code duplication with other actor networks.
@gin.configurable
class ActorRnnNetwork(network.Network):
  """Creates a recurrent actor network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               conv_layer_params=None,
               input_fc_layer_params=(200, 100),
               lstm_size=(40,),
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               name='ActorRnnNetwork'):
    """Creates an instance of `ActorRnnNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the actions.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Returns:
      A nest of action tensors matching the action_spec.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """
    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    input_layers = utils.mlp_layers(
        conv_layer_params,
        input_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        name='input_mlp')

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
        state_spec=state_spec,
        name=name)

    self._output_tensor_spec = output_tensor_spec
    self._flat_action_spec = flat_action_spec
    self._conv_layer_params = conv_layer_params
    self._input_layers = input_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._output_layers = output_layers
    self._action_layers = action_layers

  # TODO(kbanoop): Standardize argument names across different networks.
  def call(self, observation, step_type, network_state=(), training=False):
    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               self.input_tensor_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          'Input observation must have a batch or batch x time outer shape.')

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                          observation)
      step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                        step_type)

    states = tf.cast(tf.nest.flatten(observation)[0], tf.float32)
    batch_squash = utils.BatchSquash(2)  # Squash B, and T dims.
    states = batch_squash.flatten(states)  # [B, T, ...] -> [B x T, ...]

    for layer in self._input_layers:
      states = layer(states, training=training)

    states = batch_squash.unflatten(states)  # [B x T, ...] -> [B, T, ...]

    with tf.name_scope('reset_mask'):
      reset_mask = tf.equal(step_type, time_step.StepType.FIRST)
    # Unroll over the time sequence.
    states, network_state = self._dynamic_unroll(
        states,
        reset_mask,
        initial_state=network_state,
        training=training)

    states = batch_squash.flatten(states)  # [B, T, ...] -> [B x T, ...]

    for layer in self._output_layers:
      states = layer(states, training=training)

    actions = []
    for layer, spec in zip(self._action_layers, self._flat_action_spec):
      action = layer(states, training=training)
      action = common.scale_to_spec(action, spec)
      action = batch_squash.unflatten(action)  # [B x T, ...] -> [B, T, ...]
      if not has_time_dim:
        action = tf.squeeze(action, axis=1)
      actions.append(action)

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, actions)
    return output_actions, network_state
