# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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

"""Sample Keras networks for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers as keras_layers

from tf_agents.networks import network
from tf_agents.networks import utils
import gin

nest = tf.contrib.framework.nest


def validate_specs(action_spec, observation_spec):
  """Validates the spec contains a single observation and action."""
  if len(nest.flatten(observation_spec)) > 1:
    raise ValueError(
        'Network only supports observation_specs with a single observation.')

  flat_action_spec = nest.flatten(action_spec)
  if len(flat_action_spec) > 1:
    raise ValueError('Network only supports action_specs with a single action.')

  if flat_action_spec[0].shape not in [(), (1,)]:
    raise ValueError(
        'Network only supports action_specs with shape in [(), (1,)])')


@gin.configurable
class QNetwork(network.Network):
  """Feed Forward network."""

  def __init__(self,
               observation_spec,
               action_spec,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               activation_fn=tf.keras.activations.relu,
               name='QNetwork'):
    """Creates an instance of `QNetwork`.

    Args:
      observation_spec: A nest of `tensor_spec.TensorSpec` representing the
        observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
    validate_specs(action_spec, observation_spec)
    action_spec = nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    layers = utils.mlp_layers(
        conv_layer_params,
        fc_layer_params,
        activation_fn=activation_fn,
        name='input_mlp')

    # TODO(kewa): consider create custom layer flattens/restores nested actions.
    layers.append(
        keras_layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(
                minval=-0.03, maxval=0.03),
            # TODO(kewa): double check if initialization is needed.
            bias_initializer=tf.constant_initializer(-0.2)))

    super(QNetwork, self).__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        state_spec=(),
        name=name)

    self._conv_layer_params = conv_layer_params
    self._fc_layer_params = fc_layer_params
    self._activation_fn = activation_fn
    self._layers = layers

  def call(self, observation, step_type=None, network_state=None):
    del step_type  # unused.
    state = tf.to_float(nest.flatten(observation)[0])
    for layer in self.layers:
      state = layer(state)
    return state, network_state
