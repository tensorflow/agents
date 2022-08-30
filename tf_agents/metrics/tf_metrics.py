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

"""TF metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.replay_buffers import table
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class TFDeque(object):
  """Deque backed by tf.Variable storage."""

  def __init__(self, max_len, dtype, shape=(), name='TFDeque'):
    self._max_len = tf.convert_to_tensor(max_len, dtype=tf.int32)
    self._spec = tf.TensorSpec(shape, dtype, name='Buffer')
    self._buffer = table.Table(self._spec, capacity=max_len)

    self._head = common.create_variable(
        initial_value=0, dtype=tf.int32, shape=(), name=name + 'Head')

  @property
  def data(self):
    return self._buffer.read(tf.range(self.length))

  @common.function(autograph=True)
  def extend(self, value):
    for v in value:
      self.add(v)

  @common.function(autograph=True)
  def add(self, value):
    position = tf.math.mod(self._head, self._max_len)
    self._buffer.write(position, value)
    self._head.assign_add(1)

  @property
  def length(self):
    return tf.minimum(self._head, self._max_len)

  @common.function
  def clear(self):
    self._head.assign(0)

  @common.function(autograph=True)
  def mean(self):
    if tf.equal(self._head, 0):
      return tf.zeros(self._spec.shape, self._spec.dtype)
    return tf.math.reduce_mean(self.data, axis=0)

  @common.function(autograph=True)
  def max(self):
    if tf.equal(self._head, 0):
      return tf.fill(self._spec.shape, self._spec.dtype.min)
    return tf.math.reduce_max(self.data, axis=0)

  @common.function(autograph=True)
  def min(self):
    if tf.equal(self._head, 0):
      return tf.fill(self._spec.shape, self._spec.dtype.max)
    return tf.math.reduce_min(self.data, axis=0)


@gin.configurable(module='tf_agents')
class EnvironmentSteps(tf_metric.TFStepMetric):
  """Counts the number of steps taken in the environment."""

  def __init__(self, name='EnvironmentSteps', prefix='Metrics', dtype=tf.int64):
    super(EnvironmentSteps, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self.environment_steps = common.create_variable(
        initial_value=0, dtype=self.dtype, shape=(), name='environment_steps')

  def call(self, trajectory):
    """Increase the number of environment_steps according to trajectory.

    Step count is not increased on trajectory.boundary() since that step
    is not part of any episode.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    # The __call__ will execute this.
    num_steps = tf.cast(~trajectory.is_boundary(), self.dtype)
    num_steps = tf.reduce_sum(input_tensor=num_steps)
    self.environment_steps.assign_add(num_steps)
    return trajectory

  def result(self):
    return tf.identity(self.environment_steps, name=self.name)

  @common.function
  def reset(self):
    self.environment_steps.assign(0)


@gin.configurable(module='tf_agents')
class NumberOfEpisodes(tf_metric.TFStepMetric):
  """Counts the number of episodes in the environment."""

  def __init__(self, name='NumberOfEpisodes', prefix='Metrics', dtype=tf.int64):
    super(NumberOfEpisodes, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self.number_episodes = common.create_variable(
        initial_value=0, dtype=self.dtype, shape=(), name='number_episodes')

  def call(self, trajectory):
    """Increase the number of number_episodes according to trajectory.

    It would increase for all trajectory.is_last().

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    # The __call__ will execute this.
    num_episodes = tf.cast(trajectory.is_last(), self.dtype)
    num_episodes = tf.reduce_sum(input_tensor=num_episodes)
    self.number_episodes.assign_add(num_episodes)
    return trajectory

  def result(self):
    return tf.identity(self.number_episodes, name=self.name)

  @common.function
  def reset(self):
    self.number_episodes.assign(0)


@gin.configurable(module='tf_agents')
class AverageReturnMetric(tf_metric.TFStepMetric):
  """Metric to compute the average return."""

  def __init__(self,
               name='AverageReturn',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AverageReturnMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._return_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._return_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                 self._return_accumulator))

    # Update accumulator with received rewards. We are summing over all
    # non-batch dimensions in case the reward is a vector.
    self._return_accumulator.assign_add(
        tf.reduce_sum(
            trajectory.reward, axis=range(1, len(trajectory.reward.shape))))

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._return_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


@gin.configurable(module='tf_agents')
class MaxReturnMetric(tf_metric.TFStepMetric):
  """Metric to compute the max return."""

  def __init__(self,
               name='MaxReturn',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(MaxReturnMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._return_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._return_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                 self._return_accumulator))

    # Update accumulator with received rewards.
    self._return_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._return_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.max()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


@gin.configurable(module='tf_agents')
class MinReturnMetric(tf_metric.TFStepMetric):
  """Metric to compute the min return."""

  def __init__(self,
               name='MinReturn',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(MinReturnMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._return_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._return_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                 self._return_accumulator))

    # Update accumulator with received rewards.
    self._return_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._return_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.min()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


@gin.configurable(module='tf_agents')
class AverageEpisodeLengthMetric(tf_metric.TFStepMetric):
  """Metric to compute the average episode length."""

  def __init__(self,
               name='AverageEpisodeLength',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AverageEpisodeLengthMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._length_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Each non-boundary trajectory (first, mid or last) represents a step.
    non_boundary_indices = tf.squeeze(
        tf.where(tf.logical_not(trajectory.is_boundary())), axis=-1)
    self._length_accumulator.scatter_add(
        tf.IndexedSlices(
            tf.ones_like(
                non_boundary_indices, dtype=self._length_accumulator.dtype),
            non_boundary_indices))

    # Add lengths to buffer when we hit end of episode
    last_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_indices:
      self._buffer.add(self._length_accumulator[indx])

    # Clear length accumulator at the end of episodes.
    self._length_accumulator.scatter_update(
        tf.IndexedSlices(
            tf.zeros_like(last_indices, dtype=self._dtype), last_indices))

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._length_accumulator.assign(tf.zeros_like(self._length_accumulator))


@gin.configurable(module='tf_agents')
class ChosenActionHistogram(tf_metric.TFHistogramStepMetric):
  """Metric to compute the frequency of each action chosen."""

  def __init__(self,
               name='ChosenActionHistogram',
               dtype=tf.int32,
               buffer_size=100):
    super(ChosenActionHistogram, self).__init__(name=name)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype

  @common.function
  def call(self, trajectory):
    self._buffer.extend(trajectory.action)
    return trajectory

  @common.function
  def result(self):
    return self._buffer.data

  @common.function
  def reset(self):
    self._buffer.clear()


@gin.configurable(module='tf_agents')
class AverageReturnMultiMetric(tf_metric.TFMultiMetricStepMetric):
  """Metric to compute the average return for multiple metrics."""

  def __init__(self,
               reward_spec,
               name='AverageReturnMultiMetric',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    self._batch_size = batch_size
    self._buffer = tf.nest.map_structure(
        lambda r: TFDeque(buffer_size, r.dtype, r.shape), reward_spec)
    metric_names = _get_metric_names_from_spec(reward_spec)
    self._dtype = dtype
    def create_acc(spec):
      return common.create_variable(
          initial_value=np.zeros((batch_size,) + spec.shape),
          shape=(batch_size,) + spec.shape,
          dtype=spec.dtype,
          name='Accumulator/' + spec.name)
    self._return_accumulator = tf.nest.map_structure(create_acc, reward_spec)
    self._reward_spec = reward_spec
    super(AverageReturnMultiMetric, self).__init__(
        name=name, prefix=prefix, metric_names=metric_names)

  @common.function(autograph=True)
  def call(self, trajectory):
    nest_utils.assert_same_structure(trajectory.reward, self._reward_spec)
    for buf, return_acc, reward in zip(
        tf.nest.flatten(self._buffer),
        tf.nest.flatten(self._return_accumulator),
        tf.nest.flatten(trajectory.reward)):
      # Zero out batch indices where a new episode is starting.
      is_start = trajectory.is_first()
      if reward.shape.rank > 1:
        is_start = tf.broadcast_to(tf.reshape(trajectory.is_first(), [-1, 1]),
                                   tf.shape(return_acc))
      return_acc.assign(
          tf.where(is_start, tf.zeros_like(return_acc),
                   return_acc))

      # Update accumulator with received rewards.
      return_acc.assign_add(reward)

      # Add final returns to buffer.
      last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
      for indx in last_episode_indices:
        buf.add(return_acc[indx])

    return trajectory

  def result(self):
    return tf.nest.map_structure(lambda b: b.mean(), self._buffer)

  @common.function
  def reset(self):
    tf.nest.map_structure(lambda b: b.clear(), self._buffer)
    tf.nest.map_structure(lambda acc: acc.assign(tf.zeros_like(acc)),
                          self._return_accumulator)


def log_metrics(metrics, prefix=''):
  log = ['{0} = {1}'.format(m.name, m.log().numpy()) for m in metrics]
  logging.info('%s', '{0} \n\t\t {1}'.format(prefix, '\n\t\t '.join(log)))


def _get_metric_names_from_spec(reward_spec):
  reward_spec_flat = tf.nest.flatten(reward_spec)
  metric_names_list = tf.nest.map_structure(lambda r: r.name, reward_spec_flat)
  return metric_names_list
