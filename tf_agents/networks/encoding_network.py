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

"""Keras Encoding Network.

Implements a network that will generate the following layers:

  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class EncodingNetwork(network.Network):
  """Feed Forward network with CNN and FNN layers.."""

  def __init__(self,
               observation_spec,
               conv_layer_params=None,
               fc_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               batch_squash=True,
               name='EncodingNetwork'):
    """Creates an instance of `EncodingNetwork`.

    Network supports calls with shape outer_rank + observation_spec.shape. Note
    outer_rank must be at least 1.

    For example an observation spec with shape (2, 3) will require observations
    with at least a batch size making it shape (1, 2, 3).

    Args:
      observation_spec: A nest of `tensor_spec.TensorSpec` representing the
        observations.
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
      ValueError: If `observation_spec` contains more than one observation.
    """
    if len(nest.flatten(observation_spec)) > 1:
      raise ValueError('EncodingNetwork only supports observation_specs with '
                       'a single observation.')

    if not (conv_layer_params or fc_layer_params):
      raise ValueError('At least one conv_layer or fc_layer should be setup.')

    if not kernel_initializer:
      kernel_initializer = tf.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal')

    layers = []

    if conv_layer_params:
      layers.extend([
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=kernel_size,
              strides=strides,
              activation=activation_fn,
              kernel_initializer=kernel_initializer,
              name='%s/conv2d' % name)
          for (filters, kernel_size, strides) in conv_layer_params
      ])

    layers.append(tf.keras.layers.Flatten())

    if fc_layer_params:
      layers.extend([
          tf.keras.layers.Dense(
              num_units,
              activation=activation_fn,
              kernel_initializer=kernel_initializer,
              name='%s/dense' % name) for num_units in fc_layer_params
      ])

    super(EncodingNetwork, self).__init__(
        observation_spec=observation_spec,
        action_spec=None,
        state_spec=(),
        name=name)

    self._layers = layers
    self._batch_squash = batch_squash

  def call(self, observation, step_type=None, network_state=()):
    del step_type  # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(observation, self.observation_spec)
      batch_squash = utils.BatchSquash(outer_rank)

    # Get single observation out regardless of nesting.
    states = tf.to_float(nest.flatten(observation)[0])

    if self._batch_squash:
      states = batch_squash.flatten(states)

    for layer in self.layers:
      states = layer(states)

    if self._batch_squash:
      states = batch_squash.unflatten(states)
    return states, network_state
