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

"""Sample Keras Value Network with LSTM cells .

Implements a network that will generate the following layers:

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
class ValueRnnNetwork(network.Network):
  """Feed Forward value network. Reduces to 1 value output per batch item."""

  def __init__(self,
               input_tensor_spec,
               conv_layer_params=None,
               input_fc_layer_params=(75, 40),
               input_dropout_layer_params=None,
               lstm_size=(40,),
               output_fc_layer_params=(75, 40),
               activation_fn=tf.keras.activations.relu,
               name='ValueRnnNetwork'):
    """Creates an instance of `ValueRnnNetwork`.

    Network supports calls with shape outer_rank + input_tensor_shape.shape.
    Note outer_rank must be at least 1.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
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
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` contains more than one observation.
    """
    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError(
          'Network only supports observation_specs with a single observation.')

    input_layers = utils.mlp_layers(
        conv_layer_params,
        input_fc_layer_params,
        input_dropout_layer_params,
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
            name='network_state_spec'), cell.state_size)

    output_layers = []
    if output_fc_layer_params:
      output_layers = [
          tf.keras.layers.Dense(
              num_units,
              activation=activation_fn,
              kernel_initializer=tf.compat.v1.variance_scaling_initializer(
                  scale=2.0, mode='fan_in', distribution='truncated_normal'),
              name='output/dense') for num_units in output_fc_layer_params
      ]

    value_projection_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03),
    )

    state_spec = tf.nest.map_structure(
        functools.partial(
            tensor_spec.TensorSpec, dtype=tf.float32,
            name='network_state_spec'), list(cell.state_size))

    super(ValueRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name=name)

    self._conv_layer_params = conv_layer_params
    self._input_layers = input_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._output_layers = output_layers
    self._value_projection_layer = value_projection_layer

  def call(self, observation, step_type=None, network_state=None):
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
    states = batch_squash.flatten(states)

    for layer in self._input_layers:
      states = layer(states)

    states = batch_squash.unflatten(states)

    with tf.name_scope('reset_mask'):
      reset_mask = tf.equal(step_type, time_step.StepType.FIRST)
    # Unroll over the time sequence.
    states, network_state = self._dynamic_unroll(
        states,
        reset_mask,
        initial_state=network_state)

    states = batch_squash.flatten(states)

    for layer in self._output_layers:
      states = layer(states)

    value = self._value_projection_layer(states)
    value = tf.reshape(value, [-1])
    value = batch_squash.unflatten(value)
    return value, network_state
