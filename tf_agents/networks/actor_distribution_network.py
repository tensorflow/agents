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

"""Sample Keras actor network that generates distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

import gin.tf

nest = tf.contrib.framework.nest


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_initializer_value=std_initializer_value)


@gin.configurable
class ActorDistributionNetwork(network.Network):
  """Creates an actor producing either Normal or Categorical distribution."""

  def __init__(self,
               observation_spec,
               action_spec,
               fc_layer_params=(200, 100),
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               categorical_projection_net=_categorical_projection_net,
               normal_projection_net=_normal_projection_net,
               name='ActorDistributionNetwork'):
    """Creates an instance of `ActorDistributionNetwork`.

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
      categorical_projection_net: Callable that generates a categorical
        projection network to be called with some hidden state and the
        outer_rank of the state.
      normal_projection_net: Callable that generates a normal projection network
        to be called with some hidden state and the outer_rank of the state.
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` contains more than one observation.
    """
    super(ActorDistributionNetwork, self).__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        state_spec=(),
        name=name)

    if len(nest.flatten(observation_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    self._mlp_layers = utils.mlp_layers(
        conv_layer_params,
        fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.keras.initializers.glorot_uniform(),
        name='input_mlp')

    self._projection_networks = []
    for single_output_spec in nest.flatten(action_spec):
      if single_output_spec.is_discrete():
        self._projection_networks.append(
            categorical_projection_net(single_output_spec))
      else:
        self._projection_networks.append(
            normal_projection_net(single_output_spec))

  def call(self, observations, step_type, network_state):
    del step_type  # unused.
    outer_rank = nest_utils.get_outer_rank(observations, self._observation_spec)
    observations = nest.flatten(observations)
    states = tf.to_float(observations[0])

    # Reshape to only a single batch dimension for neural network functions.
    batch_squash = utils.BatchSquash(outer_rank)
    states = batch_squash.flatten(states)

    for layer in self._mlp_layers:
      states = layer(states)

    # TODO(oars): Can we avoid unflattening to flatten again
    states = batch_squash.unflatten(states)
    outputs = [
        projection(states, outer_rank)
        for projection in self._projection_networks
    ]

    return nest.pack_sequence_as(self._action_spec, outputs), network_state
