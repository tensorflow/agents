# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Keras networks for Dueling (D)DQN.

Implements a TF-Agents Network from

"Dueling Network Architectures for Deep Reinforcement Learning"
 Wang et al., 2016
 https://arxiv.org/abs/1511.06581
"""

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import q_network


@gin.configurable
class DuelingQNetwork(q_network.QNetwork):
  """Extensions to the classic (D)DQN networks."""

  def __init__(
      self,
      input_tensor_spec,
      action_spec,
      preprocessing_layers=None,
      preprocessing_combiner=None,
      conv_layer_params=None,
      fc_layer_params=(75, 40),
      dropout_layer_params=None,
      activation_fn=tf.keras.activations.relu,
      kernel_initializer=None,
      batch_squash=True,
      dtype=tf.float32,
      q_layer_activation_fn=None,
      name='DuelingQNetwork',
  ):
    """Creates an instance of `DuelingQNetwork` as a subclass of QNetwork.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations. All of these
        layers must not be already built. For more details see the documentation
        of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include `tf.keras.layers.Add`
        and `tf.keras.layers.Concatenate(axis=-1)`. This layer must not be
        already built. For more details see the documentation of
        `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, where
        each item is the fraction of input units to drop. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of fc_layer_params, or be
        None.
      activation_fn: Activation function, e.g. tf.keras.activations.relu.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allows encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      q_layer_activation_fn: Activation function for the Q layer.
      name: A string representing the name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
    super(DuelingQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        action_spec=action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype,
        q_layer_activation_fn=q_layer_activation_fn,
        name=name,
    )

    # Add a dense layer to the encoding network, in parallel to the
    # q_value_layer. This 'dueling' layer estimates the state value for the
    # input.
    dueling_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03
        ),
        bias_initializer=tf.constant_initializer(-0.2),
        dtype=dtype,
    )

    self._dueling_layer = dueling_layer
    self.layers.append(self._dueling_layer)  # state value
    self.layers.append(self._q_value_layer)  # action advantage

  def call(self, observation, step_type=None, network_state=(), training=False):
    """Runs the given observation through the network.

    Args:
      observation: The observation to provide to the network.
      step_type: The step type for the given observation. See `StepType` in
        time_step.py.
      network_state: A state tuple to pass to the network, mainly used by RNNs.
      training: Whether the output is being used for training.

    Returns:
      A tuple `(logits, network_state)`.
    """
    state, network_state = self._encoder(
        observation,
        step_type=step_type,
        network_state=network_state,
        training=training,
    )

    q_values = self._q_value_layer(state)
    state_value = self._dueling_layer(state)
    advantage = state_value + (
        q_values - tf.reduce_mean(q_values, axis=1, keepdims=True)
    )
    return (advantage, q_values), network_state
