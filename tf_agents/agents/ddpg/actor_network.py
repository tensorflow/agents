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

"""Sample Actor network to use with DDPG agents."""

import tensorflow as tf
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils


import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class ActorNetwork(network.Network):
  """Creates an actor network."""

  def __init__(self,
               observation_spec,
               action_spec,
               fc_layer_params=None,
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               name='ActorNetwork'):
    """Creates an instance of `ActorNetwork`.

    Args:
      observation_spec: A nest of `tensor_spec.TensorSpec` representing the
        observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        item, or if the action data type is not `float`.
    """
    super(ActorNetwork, self).__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        state_spec=(),
        name=name)

    if len(nest.flatten(observation_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    # TODO(kbanoop): Replace mlp_layers with encoding networks.
    self._mlp_layers = utils.mlp_layers(
        conv_layer_params,
        fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_in', distribution='uniform'),
        name='input_mlp')

    self._mlp_layers.append(
        tf.keras.layers.Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='action'))

  def call(self, observations, step_type=(), network_state=()):
    del step_type  # unused.
    observations = nest.flatten(observations)
    output = tf.to_float(observations[0])
    for layer in self._mlp_layers:
      output = layer(output)

    actions = common_utils.scale_to_spec(output, self._single_action_spec)
    return nest.pack_sequence_as(self._action_spec, [actions]), network_state
