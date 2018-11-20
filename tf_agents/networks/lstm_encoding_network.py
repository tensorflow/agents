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

"""Keras LSTM Encoding Network.

Implements a network that will generate the following layers:

  [optional]: Conv2D # input_conv_layer_params
  Flatten
  [optional]: Dense  # input_fc_layer_params
  [optional]: LSTM cell
  [optional]: Dense  # output_fc_layer_params
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from tf_agents.environments import time_step
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import rnn_utils

import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class LSTMEncodingNetwork(network.Network):
  """Recurrent network."""

  def __init__(
      self,
      observation_spec,
      conv_layer_params=None,
      input_fc_layer_params=(75, 40),
      lstm_size=(40,),
      output_fc_layer_params=(75, 40),
      activation_fn=tf.keras.activations.relu,
      name='LSTMEncodingNetwork',
  ):
    """Creates an instance of `LSTMEncodingNetwork`.

    Args:
      observation_spec: A nest of `tensor_spec.TensorSpec` representing the
        observations.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully connected parameters, where
        each item is the number of units in the layer. These feed into the
        recurrent layer.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully connected parameters, where
        each item is the number of units in the layer. These are applied on top
        of the recurrent layer.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      name: A string representing name of the network.
    """
    kernel_initializer = tf.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

    input_encoder = encoding_network.EncodingNetwork(
        observation_spec,
        conv_layer_params=conv_layer_params,
        fc_layer_params=input_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer)

    # Create RNN cell
    if len(lstm_size) == 1:
      cell = tf.keras.layers.LSTMCell(lstm_size[0])
    else:
      cell = tf.keras.layers.StackedRNNCells(
          [tf.keras.layers.LSTMCell(size) for size in lstm_size])

    output_encoder = ([
        tf.keras.layers.Dense(
            num_units,
            activation=activation_fn,
            kernel_initializer=kernel_initializer,
            name='/'.join([name, 'dense']))
        for num_units in output_fc_layer_params
    ])

    state_spec = nest.map_structure(
        functools.partial(
            tensor_spec.TensorSpec, dtype=tf.float32,
            name='network_state_spec'), list(cell.state_size))

    super(LSTMEncodingNetwork, self).__init__(
        observation_spec=observation_spec,
        action_spec=None,
        state_spec=state_spec,
        name=name)

    self._conv_layer_params = conv_layer_params
    self._input_encoder = input_encoder
    self._cell = cell
    self._output_encoder = output_encoder

  def call(self, observation, step_type, network_state=None):
    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               self._observation_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          'Input observation must have a batch or batch x time outer shape.')

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                       observation)
      step_type = nest.map_structure(lambda t: tf.expand_dims(t, 1), step_type)

    state = tf.to_float(nest.flatten(observation)[0])

    num_feature_dims = 3 if self._conv_layer_params else 1
    state.shape.with_rank_at_least(num_feature_dims)
    batch_squash = utils.BatchSquash(state.shape.ndims - num_feature_dims)

    state = batch_squash.flatten(state)
    state, network_state = self._input_encoder(state, step_type, network_state)
    state = batch_squash.unflatten(state)

    with tf.name_scope('reset_mask'):
      reset_mask = tf.equal(step_type, time_step.StepType.FIRST)
    # Unroll over the time sequence.
    state, network_state, _ = rnn_utils.dynamic_unroll(
        self._cell,
        state,
        reset_mask,
        initial_state=network_state,
        dtype=tf.float32)

    state = batch_squash.flatten(state)
    for layer in self._output_encoder:
      state = layer(state)
    state = batch_squash.unflatten(state)

    if not has_time_dim:
      # Remove time dimension from the state.
      state = tf.squeeze(state, [1])

    return state, network_state
