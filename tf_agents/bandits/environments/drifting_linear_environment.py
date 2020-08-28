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

"""Bandit drifting linear environment."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import non_stationary_stochastic_environment as nsse
from tf_agents.specs import tensor_spec
from tf_agents.typing import types


def _raise_batch_shape_error(tensor_name, batch_shape):
  raise ValueError('`{tensor_name}` must have batch shape with length 1; '
                   'got {batch_shape}.'.format(
                       tensor_name=tensor_name,
                       batch_shape=batch_shape))


def _update_row(input_x, updates, row_index):
  """Updates the i-th row of tensor `x` with the values given in `updates`.

  Args:
    input_x: the input tensor.
    updates: the values to place on the i-th row of `x`.
    row_index: which row to update.

  Returns:
    The updated tensor (same shape as `x`).
  """
  n = tf.compat.dimension_value(input_x.shape[1]) or tf.shape(input_x)[1]
  indices = tf.concat(
      [row_index * tf.ones([n, 1], dtype=tf.int32),
       tf.reshape(tf.range(n, dtype=tf.int32), [n, 1])], axis=-1)
  return tf.tensor_scatter_nd_update(
      tensor=input_x, indices=indices, updates=tf.squeeze(updates))


def _apply_givens_rotation(cosa, sina, axis_i, axis_j, input_x):
  """Applies a Givens rotation on tensor `x`.

  Reference on Givens rotations:
  https://en.wikipedia.org/wiki/Givens_rotation

  Args:
    cosa: the cosine of the angle.
    sina: the sine of the angle.
    axis_i: the first axis of rotation.
    axis_j: the second axis of rotation.
    input_x: the input tensor.

  Returns:
    The rotated tensor (same shape as `x`).
  """
  output = _update_row(
      input_x, cosa * input_x[axis_i, :] - sina * input_x[axis_j, :], axis_i)
  output = _update_row(
      output, sina * input_x[axis_i, :] + cosa * input_x[axis_j, :], axis_j)
  return output


@gin.configurable
class DriftingLinearDynamics(nsse.EnvironmentDynamics):
  """A drifting linear environment dynamics.

  This is a drifting linear environment which computes rewards as:

  rewards(t) = observation(t) * observation_to_reward(t) + additive_reward(t)

  where `t` is the environment time. `observation_to_reward` slowly rotates over
  time. The environment time is incremented in the base class after the reward
  is computed. The parameters `observation_to_reward` and `additive_reward` are
  updated at each time step.
  In order to preserve the norm of the `observation_to_reward` (and the range
  of values of the reward) the drift is applied in form of rotations, i.e.,

  observation_to_reward(t) = R(theta(t)) * observation_to_reward(t - 1)

  where `theta` is the angle of the rotation. The angle is sampled from a
  provided input distribution.
  """

  def __init__(self,
               observation_distribution: types.Distribution,
               observation_to_reward_distribution: types.Distribution,
               drift_distribution: types.Distribution,
               additive_reward_distribution: types.Distribution):
    """Initialize the parameters of the drifting linear dynamics.

    Args:
      observation_distribution: A distribution from tfp.distributions with shape
        `[batch_size, observation_dim]` Note that the values of `batch_size` and
        `observation_dim` are deduced from the distribution.
      observation_to_reward_distribution: A distribution from
        `tfp.distributions` with shape `[observation_dim, num_actions]`. The
        value `observation_dim` must match the second dimension of
        `observation_distribution`.
      drift_distribution: A scalar distribution from `tfp.distributions` of
        type tf.float32. It represents the angle of rotation.
      additive_reward_distribution: A distribution from `tfp.distributions` with
        shape `[num_actions]`. This models the non-contextual behavior of the
        bandit.
    """
    self._observation_distribution = observation_distribution
    self._drift_distribution = drift_distribution
    self._observation_to_reward_distribution = (
        observation_to_reward_distribution)
    self._additive_reward_distribution = additive_reward_distribution

    observation_batch_shape = observation_distribution.batch_shape
    reward_batch_shape = additive_reward_distribution.batch_shape
    if observation_batch_shape.rank != 2:
      _raise_batch_shape_error(
          'observation_distribution', observation_batch_shape)
    if reward_batch_shape.rank != 1:
      _raise_batch_shape_error(
          'additive_reward_distribution', reward_batch_shape)
    if additive_reward_distribution.dtype != tf.float32:
      raise ValueError('Reward  must have dtype float32; got {}'.format(
          self._reward.dtype))
    self._observation_dim = self._observation_distribution.batch_shape[1]

    expected_observation_to_reward_shape = [
        tf.compat.dimension_value(
            self._observation_distribution.batch_shape[1:]),
        tf.compat.dimension_value(
            self._additive_reward_distribution.batch_shape[0])]
    observation_to_reward_shape = [
        tf.compat.dimension_value(x)
        for x in observation_to_reward_distribution.batch_shape]
    if (observation_to_reward_shape !=
        expected_observation_to_reward_shape):
      raise ValueError(
          'Observation to reward has {} as expected shape; got {}'.format(
              expected_observation_to_reward_shape,
              observation_to_reward_shape))

    self._current_observation_to_reward = tf.compat.v2.Variable(
        observation_to_reward_distribution.sample(),
        dtype=tf.float32,
        name='observation_to_reward')
    self._current_additive_reward = tf.compat.v2.Variable(
        additive_reward_distribution.sample(),
        dtype=tf.float32,
        name='additive_reward')

  @property
  def batch_size(self) -> types.Int:
    return tf.compat.dimension_value(
        self._observation_distribution.batch_shape[0])

  @property
  def observation_spec(self) -> types.TensorSpec:
    return tensor_spec.TensorSpec(
        shape=self._observation_distribution.batch_shape[1:],
        dtype=self._observation_distribution.dtype,
        name='observation_spec')

  @property
  def action_spec(self) -> types.BoundedTensorSpec:
    return tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=tf.compat.dimension_value(
            self._additive_reward_distribution.batch_shape[0]) - 1,
        name='action')

  def observation(self, unused_t) -> types.NestedTensor:
    return self._observation_distribution.sample()

  def reward(self,
             observation: types.NestedTensor,
             t: types.Int) -> types.NestedTensor:
    # Apply the drift.
    theta = self._drift_distribution.sample()
    random_i = tf.random.uniform(
        [], minval=0, maxval=self._observation_dim - 1, dtype=tf.int32)
    random_j = tf.math.mod(random_i + 1, self._observation_dim)
    tf.compat.v1.assign(
        self._current_observation_to_reward,
        _apply_givens_rotation(
            tf.cos(theta), tf.sin(theta), random_i, random_j,
            self._current_observation_to_reward))
    tf.compat.v1.assign(self._current_additive_reward,
                        self._additive_reward_distribution.sample())

    reward = (tf.matmul(observation, self._current_observation_to_reward) +
              self._current_additive_reward)
    return reward

  @gin.configurable
  def compute_optimal_reward(
      self, observation: types.NestedTensor) -> types.NestedTensor:
    deterministic_reward = tf.matmul(
        observation, self._current_observation_to_reward)
    optimal_action_reward = tf.reduce_max(deterministic_reward, axis=-1)
    return optimal_action_reward

  @gin.configurable
  def compute_optimal_action(
      self, observation: types.NestedTensor) -> types.NestedTensor:
    deterministic_reward = tf.matmul(
        observation, self._current_observation_to_reward)
    optimal_action = tf.argmax(
        deterministic_reward, axis=-1, output_type=tf.int32)
    return optimal_action


@gin.configurable
class DriftingLinearEnvironment(nsse.NonStationaryStochasticEnvironment):
  """Implements a drifting linear environment."""

  def __init__(self,
               observation_distribution: types.Distribution,
               observation_to_reward_distribution: types.Distribution,
               drift_distribution: types.Distribution,
               additive_reward_distribution: types.Distribution):
    """Initialize the environment with the dynamics parameters.

    Args:
      observation_distribution: A distribution from `tfp.distributions` with
        shape `[batch_size, observation_dim]`. Note that the values of
        `batch_size` and `observation_dim` are deduced from the distribution.
      observation_to_reward_distribution: A distribution from
        `tfp.distributions` with shape `[observation_dim, num_actions]`. The
        value `observation_dim` must match the second dimension of
        `observation_distribution`.
      drift_distribution: A scalar distribution from `tfp.distributions` of
        type tf.float32. It represents the angle of rotation.
      additive_reward_distribution: A distribution from `tfp.distributions` with
        shape `[num_actions]`. This models the non-contextual behavior of the
        bandit.
    """
    super(DriftingLinearEnvironment, self).__init__(
        DriftingLinearDynamics(
            observation_distribution,
            observation_to_reward_distribution,
            drift_distribution,
            additive_reward_distribution))
