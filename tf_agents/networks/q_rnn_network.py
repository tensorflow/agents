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

"""Sample recurrent Keras network for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers

from tf_agents.agents.dqn import q_network
from tf_agents.networks import lstm_encoding_network

import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class QRnnNetwork(lstm_encoding_network.LSTMEncodingNetwork):
  """Recurrent network."""

  def __init__(
      self,
      observation_spec,
      action_spec,
      conv_layer_params=None,
      input_fc_layer_params=(75, 40),
      lstm_size=(40,),
      output_fc_layer_params=(75, 40),
      activation_fn=tf.keras.activations.relu,
      name='QRnnNetwork',
  ):
    """Creates an instance of `QRnnNetwork`.

    Args:
      observation_spec: A nest of `tensor_spec.TensorSpec` representing the
        observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
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

    Raises:
      ValueError: If `observation_spec` contains more than one observation. Or
        If `action_spec` contains more than one action.
    """
    q_network.validate_specs(action_spec, observation_spec)
    action_spec = nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    q_projection = layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.constant_initializer(-0.2),
        name='num_action_project/dense')

    super(QRnnNetwork, self).__init__(
        observation_spec=observation_spec,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        name=name)

    self._output_encoder.append(q_projection)
