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

"""Sample Keras actor network  with LSTM cells that generates distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import tensorflow as tf

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import dynamic_unroll_layer
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value)


@gin.configurable
class ActorDistributionRnnNetwork(network.DistributionNetwork):
  """Creates an actor producing either Normal or Categorical distribution.

  Note: By default, this network uses `NormalProjectionNetwork` for continuous
  projection which by default uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               input_fc_layer_params=(200, 100),
               input_dropout_layer_params=None,
               output_fc_layer_params=(200, 100),
               conv_layer_params=None,
               lstm_size=(40,),
               activation_fn=tf.keras.activations.relu,
               categorical_projection_net=_categorical_projection_net,
               normal_projection_net=_normal_projection_net,
               name='ActorDistributionRnnNetwork'):
    """Creates an instance of `ActorDistributionRnnNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of input_fc_layer_params, or
        be None.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      categorical_projection_net: Callable that generates a categorical
        projection network to be called with some hidden state and the
        outer_rank of the state.
      normal_projection_net: Callable that generates a normal projection network
        to be called with some hidden state and the outer_rank of the state.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """
    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

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

    output_layers = utils.mlp_layers(
        fc_layer_params=output_fc_layer_params, name='output')

    projection_networks = []
    for single_output_spec in tf.nest.flatten(output_tensor_spec):
      if tensor_spec.is_discrete(single_output_spec):
        projection_networks.append(
            categorical_projection_net(single_output_spec))
      else:
        projection_networks.append(normal_projection_net(single_output_spec))

    projection_distribution_specs = [
        proj_net.output_spec for proj_net in projection_networks
    ]
    output_spec = tf.nest.pack_sequence_as(output_tensor_spec,
                                           projection_distribution_specs)

    super(ActorDistributionRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        output_spec=output_spec,
        name=name)

    self._conv_layer_params = conv_layer_params
    self._input_layers = input_layers
    self._dynamic_unroll = dynamic_unroll_layer.DynamicUnroll(cell)
    self._output_layers = output_layers
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observation, step_type, network_state=None):
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

    states = batch_squash.unflatten(states)
    outputs = [
        projection(states, num_outer_dims)
        for projection in self._projection_networks
    ]

    output_actions = tf.nest.pack_sequence_as(self._output_tensor_spec, outputs)
    return output_actions, network_state
