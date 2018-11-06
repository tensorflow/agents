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

"""Sample Actor and State-Value function network for PPO agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf

from tf_agents.distributions import layers as distribution_layers
from tf_agents.distributions import utils as distribution_utils
from tf_agents.environments import time_step as ts
from tf_agents.networks import utils as network_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import rnn_utils
import gin

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


# TODO(oars): Figure out a way to properly generate this so that it always
# matches the cell below.
def get_state_spec(lstm_size=40, dtype=tf.float32):
  """Make a spec that matches policy state for actor_net."""
  state_tuple = tf.contrib.rnn.LSTMStateTuple(lstm_size, lstm_size)
  spec = nest.map_structure(
      functools.partial(
          tensor_spec.TensorSpec, dtype=dtype, name='network_state'),
      state_tuple)
  return spec


def _make_lstm_architecture(inputs,
                            reset_mask,
                            outer_rank,
                            network_state=None,
                            conv_layers=None,
                            fc_layers=None,
                            lstm_size=(40,),
                            output_fc_layers=(40,),
                            normalizer_fn=None,
                            activation_fn=tf.nn.relu):
  """Construct architecture with encoder, LSTM cells, and decoder."""
  has_time_dim = outer_rank == 2
  if not has_time_dim:
    inputs = tf.expand_dims(inputs, 1)
    reset_mask = tf.expand_dims(reset_mask, 1)

  output = tf.to_float(nest.flatten(inputs)[0])

  encoder_arg_scope_fn = functools.partial(
      slim.arg_scope, [slim.conv2d, slim.fully_connected],
      activation_fn=activation_fn, normalizer_fn=normalizer_fn,)

  with tf.variable_scope('input_network'), encoder_arg_scope_fn():
    output = network_utils.encode_state(
        output, conv_layers=conv_layers, fc_layers=fc_layers)

  # Create RNN cell
  with tf.variable_scope('LSTM'):
    if len(lstm_size) == 1:
      cell = tf.contrib.rnn.BasicLSTMCell(lstm_size[0])
    else:
      cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.BasicLSTMCell(cell_size) for cell_size in lstm_size])

    # Unroll over the time sequence.
    output, network_state, _ = rnn_utils.dynamic_unroll(
        cell, output, reset_mask, initial_state=network_state, dtype=tf.float32)

  # Final pass through fc layers.
  with encoder_arg_scope_fn():
    output = network_utils.encode_state(output, fc_layers=output_fc_layers)

  if not has_time_dim:
    # Remove time dimension from the state.
    output = tf.squeeze(output, [1])

  return output, network_state


@gin.configurable
def actor_network(time_steps,
                  action_spec,
                  network_state,
                  conv_layers=None,
                  fc_layers=None,
                  lstm_size=(40,),
                  output_fc_layers=None,
                  time_step_spec=None,
                  normalizer_fn=None,
                  activation_fn=tf.nn.relu,
                  init_means_output_factor=0.1,
                  init_action_stddev=0.35):
  """Creates an actor that parameterizes a Gaussian policy (means, stdevs).

  Args:
    time_steps: A batched TimeStep tuple containing the observation.
    action_spec: A BoundedTensorSpec indicating the shape and range of actions.
    network_state: Optional policy state to use when evaluating RNN cells.
      If not provided the cell's zero state is used instead.
    conv_layers: Optional list of convolution layers parameters, where each item
      is a length-three tuple indicating (num_units, kernel_size, stride).
    fc_layers: Optional list of fully connected parameters, where each item is
      the number of units in the layer.
    lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
    output_fc_layers: Like fc_layers, but to be applied after LSTM.
    time_step_spec: A spec for the time_steps, used for inferring outer_dims.
    normalizer_fn: Normalizer function, e.g. slim.layer_norm,
    activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
    init_means_output_factor: Output factor for initializing action means
      weights.
    init_action_stddev: Initializer for pre-softmax action stddevs.

  Returns:
    A nest of distributions over a batch of actions.
    A network_state tensor.
  Raises:
    ValueError:
      If observations contain more than one item.
      If the action_spec's dtype is not float32 or float64.
  """
  if len(nest.flatten(time_steps.observation)) > 1:
    raise ValueError('Only a single observation is supported by this network.')
  if time_step_spec is None:
    raise ValueError('Actor network requires that observation_spec be '
                     'provided.')
  outer_rank = nest_utils.get_outer_rank(time_steps, time_step_spec)
  if outer_rank not in (1, 2):
    raise ValueError(
        'Input time_steps must have a (batch,) or (batch, time) outer shape.')
  states, network_state = _make_lstm_architecture(
      time_steps.observation, time_steps.is_first(), outer_rank,
      network_state=network_state, conv_layers=conv_layers, fc_layers=fc_layers,
      lstm_size=lstm_size, output_fc_layers=output_fc_layers,
      normalizer_fn=normalizer_fn, activation_fn=activation_fn)

  def _projection_layer(inputs, num_elements, scope=None):
    return tf.contrib.layers.fully_connected(
        inputs, num_elements,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=init_means_output_factor),
        activation_fn=None,
        normalizer_fn=None,
        scope=scope)

  init_action_log_stddev = np.log(np.exp(init_action_stddev) - 1)
  def _project_to_continuous(inputs, output_spec, outer_rank=1):
    return distribution_layers.normal(
        inputs, output_spec, outer_rank=outer_rank,
        projection_layer=_projection_layer,
        std_initializer=tf.constant_initializer(init_action_log_stddev),
        std_transform=tf.nn.softplus)

  action_distribution = distribution_utils.project_to_output_distributions(
      states, action_spec, project_to_continuous=_project_to_continuous,
      outer_rank=outer_rank)

  return action_distribution, network_state


@gin.configurable
def value_network(observations,
                  step_types,
                  network_state,
                  conv_layers=None,
                  fc_layers=None,
                  lstm_size=(40,),
                  output_fc_layers=None,
                  observation_spec=None,
                  normalizer_fn=None,
                  activation_fn=tf.nn.relu):
  """Creates a state-value estimation network.

  The value network returns state-values for a given batch of states. This
  network supports only a single observation tensor.

  Args:
    observations: A batched observation (potentially nested) tensor. May either
      have outer_dims (batch_size,) or (batch_size, num_time_steps).
    step_types: A batched tensor of step_types. Same outer dims as observations.
    network_state: Optional rnn state to use when evaluating RNN cells. If not
      provided the cell's zero state is used instead.
    conv_layers: Optional list of convolution layers parameters for
      observations, where each item is a length-three tuple indicating
      (num_units, kernel_size, stride).
    fc_layers: Optional list of fully connected parameters, where each item is
      the number of units in the layer.
    lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
    output_fc_layers: Like fc_layers, but to be applied after LSTM.
    observation_spec: A spec for the time_steps, used for inferring outer_dims.
    normalizer_fn: Normalizer function, e.g. slim.layer_norm.
    activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu.
  Returns:
    A Tensor of tf.float32 [batch_size] of q-values.

  Raises:
    ValueError: if multiple observations or actions are given.
  """
  if len(nest.flatten(observations)) > 1:
    raise ValueError('Only a single observation is supported by this network.')
  if observation_spec is None:
    raise ValueError('Actor network requires that observation_spec be '
                     'provided.')
  outer_rank = nest_utils.get_outer_rank(observations, observation_spec)
  if outer_rank not in (1, 2):
    raise ValueError(
        'Input time_steps must have a (batch,) or (batch, time) outer shape.')

  states, network_state = _make_lstm_architecture(
      observations, tf.equal(step_types, ts.StepType.FIRST), outer_rank,
      network_state=network_state, conv_layers=conv_layers, fc_layers=fc_layers,
      lstm_size=lstm_size, output_fc_layers=output_fc_layers,
      normalizer_fn=normalizer_fn, activation_fn=activation_fn)

  # Reshape to only a single batch dimension for neural network functions.
  batch_squash = network_utils.BatchSquash(outer_rank)
  states = batch_squash.flatten(states)

  with slim.arg_scope([slim.fully_connected],
                      weights_initializer=tf.random_uniform_initializer(
                          minval=-0.003, maxval=0.003)):
    value = slim.fully_connected(states, 1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 scope='state_value')
    value = tf.reshape(value, [-1])

  # Reshape back to full batch dimensions if there are any.
  value = batch_squash.unflatten(value)

  return value, network_state
