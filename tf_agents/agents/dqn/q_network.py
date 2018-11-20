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

"""Sample Keras networks for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import encoding_network
from tf_agents.networks import network

import gin.tf

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
               kernel_initializer=None,
               batch_squash=True,
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
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
    validate_specs(action_spec, observation_spec)
    action_spec = nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    encoder = encoding_network.EncodingNetwork(
        observation_spec,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash)

    # TODO(kewa): consider create custom layer flattens/restores nested actions.
    q_value_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03),
        # TODO(kewa): double check if initialization is needed.
        bias_initializer=tf.constant_initializer(-0.2))

    super(QNetwork, self).__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        state_spec=(),
        name=name)

    self._encoder = encoder
    self._q_value_layer = q_value_layer

  def call(self, observation, step_type=None, network_state=()):
    state, network_state = self._encoder(
        observation, step_type=step_type, network_state=network_state)
    return self._q_value_layer(state), network_state
