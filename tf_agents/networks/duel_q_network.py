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

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.networks import utils

@gin.configurable
class DuelQNetwork(network.Network):
  """Feed Forward network."""

  def __init__(self,
               input_tensor_spec,
               action_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=None,
               dropout_layer_params=None,
               a_fc_layer_params=None,
               a_weight_decay_params=None,
               a_dropout_layer_params=None,
               v_fc_layer_params=None,
               v_weight_decay_params=None,
               v_dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               av_combine_fn=None,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='DuelQNetwork'):
    """Creates an instance of `DuelQNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride), used in shared encoder.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer, used in shared encoder
      *_fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer, corresponding to each branch:
        a_fc_layer_params designed for the advantage branch,
        v_fc_layer_params designed for the state branch
      *_weight_decay_params: Optional list of L2 weight decay params, where each
        item is the L2-regularization strength applied to corresponding
        fully_connected layer.The weight decay parameters are interleaved with
        the fully connected layer, except if the list is None.
        Crresponding to each branch:
        a_weight_decay_params designed for the advantage branch,
        v_weight_decay_params designed for the state branch,
      *_dropout_layer_params: Optional list of dropout layer parameters, where
        each item is the fraction of input units to drop. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None.
        Corresponding to each branch:
        a_dropout_layer_params for the advantage branch,
                               same length of a_fc_layer_params
        v_dropout_layer_params for the state branch.
                               same length of v_fc_layer_params
      activation_fn: Activation function, e.g. tf.keras.activations.relu.
      av_combine_fn: Function to produce q-value from advantage and state value
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing the name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
    q_network.validate_specs(action_spec, input_tensor_spec)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1
    encoder_input_tensor_spec = input_tensor_spec

    # Shared encoder to convert observation to shared state tensor
    # which is fed to advantage branch and state branch
    encoder = encoding_network.EncodingNetwork(
        encoder_input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=batch_squash,
        dtype=dtype,
        name='shared_encoder'
    )

    # Advantage branch

    # Advantage intermediate fully connected layers
    a_encode_layers = self.create_branch_layers(
        a_fc_layer_params,
        a_dropout_layer_params,
        a_weight_decay_params,
        activation_fn,
        kernel_initializer,
        dtype,
        name='a_branch_layer'
    )

    # Advantage dense layer to project to action space
    a_value_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.compat.v1.initializers.constant(-0.2),
        dtype=dtype,
        name='a_value_layer'
    )

    # State branch

    # State intermediate fully connected layers
    v_encoder_layers = self.create_branch_layers(
        v_fc_layer_params,
        v_dropout_layer_params,
        v_weight_decay_params,
        activation_fn,
        kernel_initializer,
        dtype,
        name='v_branch_layer'
    )

    # State dense layer to project to a single scalar state value
    v_value_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.compat.v1.initializers.constant(-0.2),
        dtype=dtype,
        name='v_value_layer'
    )

    super().__init__(
      input_tensor_spec=input_tensor_spec,
      state_spec=(),
      name=name)

    self._encoder = encoder
    self._a_encode_layers = a_encode_layers
    self._a_value_layer = a_value_layer
    self._v_encode_layers = v_encoder_layers
    self._v_value_layer = v_value_layer

    if av_combine_fn is None:
      av_combine_fn = self.av_combine_f

    self._av_combine_fn = av_combine_fn

  def create_branch_layers(self,
                          fc_layer_params,
                          dropout_layer_params,
                          weight_decay_params,
                          activation_fn,
                          kernel_initializer,
                          dtype,
                          name=None):
    """Creates fully-connected layers for advantage or state branch`.

    Args:
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout parameters, where each
        item is the fraction of input units to drop. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except that if the entry in the
        list is None.
      weight_decay_params: Optional list of L2 weight decay parameters,
        where each item is the L2 regularization strength applied on the
        corresponding fully_connected layer.The weight decay parameters are
        interleaved with the fully connected layer, except if the list is None.
      activation_fn: Activation function, the same as the activation function
        used in the shared encoder.
      kernel_initializer: Initializer to use for the kernels of fc layers,
        the same as the kernel_initializer used in the shared encoder.
      dtype: The dtype to use by the fully connected layers.
      name: A string representing the name of the network.

    Returns:
      A tensor representing the state after fully connected layers
    """
    layers = []

    if not fc_layer_params:
      return layers

    if dropout_layer_params is None:
      dropout_layer_params = [None] * len(fc_layer_params)
    else:
      if len(dropout_layer_params) != len(fc_layer_params):
        raise ValueError(
            'Dropout and fully connected layer parameter lists'
            'have different lengths %s (%d vs. %d.)' %
            (name, len(dropout_layer_params), len(fc_layer_params)))
    if weight_decay_params is None:
      weight_decay_params = [None] * len(fc_layer_params)
    else:
      if len(weight_decay_params) != len(fc_layer_params):
        raise ValueError(
            'Weight decay and fully connected layer parameter '
            'lists have different lengths %s (%d vs. %d.)' %
            (name, len(weight_decay_params), len(fc_layer_params)))

    for layer_idx, (num_units, dropout_params, weight_decay) in enumerate(
            zip(fc_layer_params, dropout_layer_params, weight_decay_params)):
      kernal_regularizer = None
      if weight_decay is not None:
        kernal_regularizer = tf.keras.regularizers.l2(weight_decay)
      layers.append(
        tf.keras.layers.Dense(
          num_units,
          activation=activation_fn,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernal_regularizer,
          dtype=dtype,
          name='{}_dense_{}'.format(name, layer_idx)))
      if not isinstance(dropout_params, dict):
        dropout_params = {
            'rate': dropout_params} if dropout_params else None

      if dropout_params is not None:
        layers.append(
            utils.maybe_permanent_dropout(**dropout_params))

    return layers

  def call(self,
           observation,
           step_type=None,
           network_state=(),
           training=False):
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
    state, network_state = self._encoder(observation,
                                         step_type=step_type,
                                         network_state=network_state,
                                         training=training)

    a_state = tf.identity(state, name='advantage_in')
    for layer in self._a_encode_layers:
      a_state = layer(a_state, training=training)
    a_value = self._a_value_layer(a_state, training=training)

    v_state = tf.identity(state, name='state_value_in')
    for layer in self._v_encode_layers:
      v_state = layer(v_state, training=training)
    v_value = self._v_value_layer(v_state, training=training)

    q_value = self._av_combine_fn(a_value, v_value)
    return q_value, network_state

  def av_combine_f(self, a_value, v_value):
    """Combine advantage value and state value to produce Q-value

    Args:
      a_value (tensor): The advantage value from the advantage network.
      v_value (tensor): The state value from the state network.

    Returns:
      q_value tensor.
    """
    a_value_mean = tf.reduce_mean(a_value, -1, keepdims=True)
    a_value_cor = a_value - a_value_mean
    return a_value_cor + v_value
