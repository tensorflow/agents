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

"""Sample Actor and Critic/Q network to use with DDPG agents."""

import tensorflow as tf

import gin

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


@gin.configurable
def critic_network(time_steps,
                   actions,
                   observation_conv_layers=None,
                   observation_fc_layers=None,
                   action_fc_layers=None,
                   joint_fc_layers=None,
                   normalizer_fn=None,
                   activation_fn=tf.nn.relu):
  """Creates a critic/q network.

  The critic returns the q values for a given batch of states and actions.
  This network supports only a single observation tensor and a single action
  tensor.

  Args:
    time_steps: A batched TimeStep tuple containing the observation.
    actions: A nest representing a batch of actions.
    observation_conv_layers: Optional list of convolution layers parameters for
      observations, where each item is a length-three tuple indicating
      (num_units, kernel_size, stride).
    observation_fc_layers: Optional list of fully connected parameters for
      observations, where each item is the number of units in the layer.
    action_fc_layers: Optional list of fully connected parameters for
      actions, where each item is the number of units in the layer.
    joint_fc_layers: Optional list of fully connected parameters after merging
      observations and actions, where each item is the number of units in the
      layer.
    normalizer_fn: Normalizer function, e.g. slim.layer_norm.
    activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu.
  Returns:
    A Tensor of tf.float32 [batch_size] of q-values.

  Raises:
    ValueError: if multiple observations or actions are given.
  """
  observations = nest.flatten(time_steps.observation)
  if len(observations) > 1:
    raise ValueError('Only a single observation is supported by this network.')
  states = tf.to_float(observations[0])

  actions = nest.flatten(actions)
  if len(actions) > 1:
    raise ValueError('Only a single action is supported by this network.')
  actions = tf.to_float(actions[0])

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):

    if observation_conv_layers:
      states.shape.assert_has_rank(4)
      states = slim.stack(states, slim.conv2d, observation_conv_layers,
                          scope='conv_states')

    states = slim.flatten(states)
    actions = slim.flatten(actions)

    if observation_fc_layers:
      states.shape.assert_has_rank(2)
      states = slim.stack(
          states,
          slim.fully_connected,
          observation_fc_layers,
          scope='fc_states')

    if action_fc_layers:
      actions.shape.assert_has_rank(2)
      actions = slim.stack(
          actions, slim.fully_connected, action_fc_layers, scope='fc_actions')

    joint = tf.concat([states, actions], 1)
    if joint_fc_layers:
      joint.shape.assert_has_rank(2)
      joint = slim.stack(joint, slim.fully_connected, joint_fc_layers,
                         scope='joint')

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      value = slim.fully_connected(joint, 1,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='q_value')
      value = tf.reshape(value, [-1])
  return value


@gin.configurable
def actor_network(time_steps,
                  action_spec,
                  conv_layers=None,
                  fc_layers=None,
                  normalizer_fn=None,
                  activation_fn=tf.nn.relu):
  """Creates an actor that returns actions for the given states.

  Args:
    time_steps: A batched TimeStep tuple containing the observation.
    action_spec: A BoundedTensorSpec indicating the shape and range of actions.
    conv_layers: Optional list of convolution layers parameters, where each item
      is a length-three tuple indicating (num_units, kernel_size, stride).
    fc_layers: Optional list of fully connected parameters, where each item is
      the number of units in the layer.
    normalizer_fn: Normalizer function, i.e. slim.layer_norm,
    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...
  Returns:
    A nest representing a batch of actions.
  Raises:
    ValueError:
      If observations or action_spec contain more than one item.
      If the action_spec's dtype is not float32 or float64.
  """
  observations = nest.flatten(time_steps.observation)
  if len(observations) > 1:
    raise ValueError('Only a single observation is supported by this network.')
  states = tf.to_float(observations[0])

  flat_action_spec = nest.flatten(action_spec)
  if len(flat_action_spec) > 1:
    raise ValueError('Only a single action is supported by this network.')
  single_action_spec = flat_action_spec[0]

  if single_action_spec.dtype not in [tf.float32, tf.float64]:
    raise ValueError('Only float actions are supported by this network.')

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0 / 3.0, mode='FAN_IN', uniform=True)):

    states = tf.to_float(states)
    if conv_layers:
      states.shape.assert_has_rank(4)
      states = slim.stack(states, slim.conv2d, conv_layers,
                          scope='conv_states')

    states = slim.flatten(states)
    if fc_layers:
      states.shape.assert_has_rank(2)
      states = slim.stack(
          states, slim.fully_connected, fc_layers, scope='fc_states')

    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      actions = slim.fully_connected(states,
                                     single_action_spec.shape.num_elements(),
                                     scope='actions',
                                     normalizer_fn=None,
                                     activation_fn=tf.nn.tanh)

  actions = tf.reshape(actions, [-1] + single_action_spec.shape.as_list())
  action_means = (
      single_action_spec.maximum + single_action_spec.minimum) / 2.0
  action_magnitudes = (
      single_action_spec.maximum - single_action_spec.minimum) / 2.0
  actions = action_means + action_magnitudes * actions
  actions = tf.cast(actions, single_action_spec.dtype)
  actions = nest.pack_sequence_as(action_spec, [actions])
  return actions
