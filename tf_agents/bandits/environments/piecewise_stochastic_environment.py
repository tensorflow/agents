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

"""Bandit piecewise linear stationary environment."""

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


@gin.configurable
class PiecewiseStationaryDynamics(nsse.EnvironmentDynamics):
  """A piecewise stationary environment dynamics.

  This is a piecewise stationary environment which computes rewards as:

  rewards(t) = observation(t) * observation_to_reward(i) + additive_reward(i)

  where t is the environment time (env_time) and i is the index of each piece.
  The environment time is incremented after the reward is computed while the
  piece index is incremented at the end of the time interval. The parameters
  observation_to_reward(i), additive_reward(i), and the length of interval, are
  drawn from given distributions at the beginning of each temporal interval.
  """

  def __init__(self,
               observation_distribution: types.Distribution,
               interval_distribution: types.Distribution,
               observation_to_reward_distribution: types.Distribution,
               additive_reward_distribution: types.Distribution):
    """Initialize the parameters of the piecewise dynamics.

    Args:
      observation_distribution: A distribution from tfp.distributions with shape
        `[batch_size, observation_dim]` Note that the values of `batch_size` and
        `observation_dim` are deduced from the distribution.
      interval_distribution: A scalar distribution from `tfp.distributions`. The
        value is casted to `int64` to update the time range.
      observation_to_reward_distribution: A distribution from
        `tfp.distributions` with shape `[observation_dim, num_actions]`. The
        value `observation_dim` must match the second dimension of
        `observation_distribution`.
      additive_reward_distribution: A distribution from `tfp.distributions` with
        shape `[num_actions]`. This models the non-contextual behavior of the
        bandit.
    """
    self._observation_distribution = observation_distribution
    self._interval_distribution = interval_distribution
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
              observation_to_reward_shape,
              expected_observation_to_reward_shape))

    self._current_interval = tf.compat.v2.Variable(
        tf.cast(interval_distribution.sample(), dtype=tf.int64),
        dtype=tf.int64, name='interval')
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
    def same_interval_parameters():
      """Returns the parameters of the current piece.

      Returns:
        The pair of `tf.Tensor` `(observation_to_reward, additive_reward)`.
      """
      return [self._current_observation_to_reward,
              self._current_additive_reward]

    def new_interval_parameters():
      """Update and returns the piece parameters.

      Returns:
        The pair of `tf.Tensor` `(observation_to_reward, additive_reward)`.
      """
      tf.compat.v1.assign_add(
          self._current_interval,
          tf.cast(self._interval_distribution.sample(), dtype=tf.int64))
      tf.compat.v1.assign(self._current_observation_to_reward,
                          self._observation_to_reward_distribution.sample())
      tf.compat.v1.assign(self._current_additive_reward,
                          self._additive_reward_distribution.sample())

      return [self._current_observation_to_reward,
              self._current_additive_reward]

    observation_to_reward, additive_reward = tf.cond(
        t < self._current_interval,
        same_interval_parameters,
        new_interval_parameters)

    reward = (tf.matmul(observation, observation_to_reward) +
              tf.reshape(additive_reward, [1, -1]))
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
class PiecewiseStochasticEnvironment(nsse.NonStationaryStochasticEnvironment):
  """Implements a piecewise stationary linear environment."""

  def __init__(self,
               observation_distribution: types.Distribution,
               interval_distribution: types.Distribution,
               observation_to_reward_distribution: types.Distribution,
               additive_reward_distribution: types.Distribution):
    """Initialize the environment with the dynamics parameters.

    Args:
      observation_distribution: A distribution from `tfp.distributions` with
        shape `[batch_size, observation_dim]`. Note that the values of
        `batch_size` and `observation_dim` are deduced from the distribution.
      interval_distribution: A scalar distribution from `tfp.distributions`. The
        value is casted to `int64` to update the time range.
      observation_to_reward_distribution: A distribution from
        `tfp.distributions` with shape `[observation_dim, num_actions]`. The
        value `observation_dim` must match the second dimension of
        `observation_distribution`.
      additive_reward_distribution: A distribution from `tfp.distributions` with
        shape `[num_actions]`. This models the non-contextual behavior of the
        bandit.
    """
    super(PiecewiseStochasticEnvironment, self).__init__(
        PiecewiseStationaryDynamics(
            observation_distribution,
            interval_distribution,
            observation_to_reward_distribution,
            additive_reward_distribution))
