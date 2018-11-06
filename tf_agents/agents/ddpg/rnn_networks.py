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
import functools
import tensorflow as tf

from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import rnn_utils
import gin

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


# TODO(oars): Figure out a way to properly generate this so that it always
# matches the cell below.
def get_state_spec(lstm_size=40, dtype=tf.float32):
  return nest.map_structure(
      functools.partial(
          tensor_spec.TensorSpec, dtype=dtype, name='policy_state'),
      tf.contrib.rnn.LSTMStateTuple(lstm_size, lstm_size))


def add_time_dimension(batch, spec, name):
  """Adds time dimension to a batch.

  Args:
    batch: A batch of tensors.
    spec: A TensorSpec indicating the shape of tensors in the batch.
    name: Name of the batch for debugging purposes.

  Returns:
    batch: A batch of tensors with time dimesion added.
    add_time_dim: A boolean indicating if time dimension was added.
  Raises:
    ValueError: if the batch doesn't have a batch dimension.
  """
  num_outer_dims = nest_utils.get_outer_rank(batch, spec)
  if num_outer_dims not in (1, 2):
    raise ValueError(
        'Input %s must have a batch or batch x time outer shape.' % name)
  add_time_dim = num_outer_dims != 2
  if add_time_dim:
    # Add a time dimension to the inputs. We need to do it on the whole
    # time_step so that time_step.is_first() has the right dim for the dynamic
    # unroll.
    batch = nest.map_structure(lambda t: tf.expand_dims(t, 1), batch)
  return batch, add_time_dim


@gin.configurable
def critic_network(time_steps,
                   actions,
                   time_step_spec,
                   action_spec,
                   policy_state=None,
                   observation_conv_layers=None,
                   observation_fc_layers=None,
                   action_fc_layers=None,
                   joint_fc_layers=None,
                   lstm_size=(40,),
                   output_fc_layers=None,
                   normalizer_fn=None,
                   activation_fn=tf.nn.relu):
  """Creates a critic/q network.

  The critic returns the q values for a given batch of states and actions.
  This network supports only a single observation tensor and a single action
  tensor.

  Args:
    time_steps: A batched TimeStep tuple containing the observation.
    actions: A nest representing a batch of actions.
    time_step_spec: A `TimeStep` spec of the input time_steps.
    action_spec: A BoundedTensorSpec indicating the shape and range of actions.
    policy_state: Optional policy state to use when evaluating RNN cells. If not
      provided the cell's zero state is used instead.
    observation_conv_layers: Optional list of convolution layers parameters for
      observations, where each item is a length-three tuple indicating
      (num_units, kernel_size, stride).
    observation_fc_layers: Optional list of fully connected parameters for
      observations, where each item is the number of units in the layer.
    action_fc_layers: Optional list of fully connected parameters for actions,
      where each item is the number of units in the layer.
    joint_fc_layers: Optional list of fully connected parameters after merging
      observations and actions, where each item is the number of units in the
      layer.
    lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
    output_fc_layers: Optional list of fully connected parameters, where each
      item is the number of units in the layer. These are applied on top of the
      recurrent layer.
    normalizer_fn: Normalizer function, e.g. slim.layer_norm.
    activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu.

  Returns:
    A Tensor of tf.float32 [batch_size] of q-values.

  Raises:
    ValueError: if multiple observations or actions are given.
    ValueError: if the time_steps don't have at least 1 batch dimension.
    ValueError: if the actions don't have at least 1 batch dimension.
  """
  if len(nest.flatten(time_steps.observation)) > 1:
    raise ValueError('Only a single observation is supported by this network.')

  if len(nest.flatten(actions)) > 1:
    raise ValueError('Only a single action is supported by this network.')

  # TODO(oars): consider moving logic to add time_dim to the policy.
  time_steps, ts_add_time_dim = add_time_dimension(time_steps, time_step_spec,
                                                   'time_steps')
  actions, _ = add_time_dimension(actions, action_spec, 'actions')

  states = tf.to_float(nest.flatten(time_steps.observation)[0])
  actions = tf.to_float(nest.flatten(actions)[0])

  slim_arg_scope = functools.partial(
      slim.arg_scope, [slim.conv2d, slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1. / 3, mode='FAN_IN', uniform=True))

  # Pass through observation conv and fc_layers
  with tf.variable_scope('observation_network'), slim_arg_scope():
    states = utils.encode_state(
        states, observation_conv_layers, fc_layers=observation_fc_layers)

  # Pass through actions fc_layers
  if action_fc_layers:
    with tf.variable_scope('actions_network'), slim_arg_scope():
      actions = utils.encode_state(actions, fc_layers=action_fc_layers)

  joint = tf.concat([states, actions], 2)
  # Pass through actions fc_layers
  if joint_fc_layers:
    with tf.variable_scope('joint_network'), slim_arg_scope():
      joint = utils.encode_state(joint, fc_layers=joint_fc_layers)

  # Create RNN cell
  reset_mask = time_steps.is_first()
  if len(lstm_size) == 1:
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size[0])
  else:
    cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(cell_size) for cell_size in lstm_size])

  # Unroll over the time sequence.
  joint, policy_state, _ = rnn_utils.dynamic_unroll(
      cell, joint, reset_mask, initial_state=policy_state, dtype=tf.float32)

  # Final pass through fc layers.
  if output_fc_layers:
    with tf.variable_scope('output_network'), slim_arg_scope():
      joint = utils.encode_state(joint, fc_layers=output_fc_layers)

  if ts_add_time_dim:
    # Remove time dimension from the state.
    joint = tf.squeeze(joint, [1])

  with slim.arg_scope([slim.fully_connected],
                      weights_regularizer=None,
                      weights_initializer=tf.random_uniform_initializer(
                          minval=-0.003, maxval=0.003)):
    value = slim.fully_connected(
        joint, 1, activation_fn=None, normalizer_fn=None, scope='q_value')
    # Remove dimensions of size 1.
    value = tf.squeeze(value)
  return value, policy_state


@gin.configurable
def actor_network(time_steps,
                  action_spec,
                  time_step_spec,
                  policy_state=None,
                  conv_layers=None,
                  fc_layers=None,
                  lstm_size=(40,),
                  output_fc_layers=None,
                  normalizer_fn=None,
                  activation_fn=tf.nn.relu):
  """Creates an actor that returns actions for the given states.

  Args:
    time_steps: A batched TimeStep tuple containing the observation.
    action_spec: A BoundedTensorSpec indicating the shape and range of actions.
    time_step_spec: A `TimeStep` spec of the input time_steps.
    policy_state: Optional policy state to use when evaluating RNN cells. If not
      provided the cell's zero state is used instead.
    conv_layers: Optional list of convolution layers parameters, where each item
      is a length-three tuple indicating (num_units, kernel_size, stride).
    fc_layers: Optional list of fully connected parameters, where each item is
      the number of units in the layer.
    lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
    output_fc_layers: Optional list of fully connected parameters, where each
      item is the number of units in the layer. These are applied on top of the
      recurrent layer.
    normalizer_fn: Normalizer function, i.e. slim.layer_norm,
    activation_fn: Activation function, i.e. tf.nn.relu, slim.leaky_relu, ...

  Returns:
    A nest representing a batch of actions.
  Raises:
    ValueError:
      If observations or action_spec contain more than one item.
      If the action_spec's dtype is not float32 or float64.
  """
  if len(nest.flatten(time_steps.observation)) > 1:
    raise ValueError('Only a single observation is supported by this network.')

  flat_action_spec = nest.flatten(action_spec)
  if len(flat_action_spec) > 1:
    raise ValueError('Only a single action is supported by this network.')
  single_action_spec = flat_action_spec[0]

  if single_action_spec.dtype not in [tf.float32, tf.float64]:
    raise ValueError('Only float actions are supported by this network.')

  # TODO(oars): consider moving logic to add time_dim to the policy.
  num_outer_dims = nest_utils.get_outer_rank(time_steps, time_step_spec)
  if num_outer_dims not in (1, 2):
    raise ValueError(
        'Input time_steps must have a batch or batch x time outer shape.')

  has_time_dim = num_outer_dims == 2
  if not has_time_dim:
    # Add a time dimension to the inputs. We need to do it on the whole
    # time_step so that time_step.is_first() has the right dim for the dynamic
    # unroll.
    time_steps = nest.map_structure(lambda t: tf.expand_dims(t, 1), time_steps)

  states = tf.to_float(nest.flatten(time_steps.observation)[0])

  slim_arg_scope = functools.partial(
      slim.arg_scope, [slim.conv2d, slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1. / 3, mode='FAN_IN', uniform=True))

  # Pass through observation conv and fc_layers
  with tf.variable_scope('input_network'), slim_arg_scope():
    states = utils.encode_state(states, conv_layers, fc_layers)

  # Create RNN cell
  reset_mask = time_steps.is_first()
  if len(lstm_size) == 1:
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size[0])
  else:
    cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.BasicLSTMCell(cell_size) for cell_size in lstm_size])

  # Unroll over the time sequence.
  states, policy_state, _ = rnn_utils.dynamic_unroll(
      cell, states, reset_mask, initial_state=policy_state, dtype=tf.float32)

  # Final pass through fc layers.
  if output_fc_layers:
    with tf.variable_scope('output_network'), slim_arg_scope():
      states = utils.encode_state(states, fc_layers=output_fc_layers)

  if not has_time_dim:
    # Remove time dimension from the state.
    states = tf.squeeze(states, [1])

  with slim.arg_scope([slim.fully_connected],
                      weights_initializer=tf.random_uniform_initializer(
                          minval=-0.003, maxval=0.003)):
    actions = slim.fully_connected(
        states,
        single_action_spec.shape.num_elements(),
        scope='actions',
        normalizer_fn=None,
        activation_fn=tf.nn.tanh)

  action_means = (single_action_spec.maximum + single_action_spec.minimum) / 2.0
  action_magnitudes = (
      single_action_spec.maximum - single_action_spec.minimum) / 2.0
  actions = action_means + action_magnitudes * actions
  actions = tf.cast(actions, single_action_spec.dtype)
  actions = nest.pack_sequence_as(action_spec, [actions])
  return actions, policy_state
