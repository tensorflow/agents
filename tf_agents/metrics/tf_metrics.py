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

"""TF metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_py_metric
from tf_agents.utils import common


# TODO(kbanoop): Add reset to TF metrics for eval in eager mode.


class EnvironmentSteps(tf_metric.TFStepMetric):
  """Counts the number of steps taken in the environment."""

  def __init__(self, name='EnvironmentSteps', dtype=tf.int64):
    super(EnvironmentSteps, self).__init__(name=name)
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
    return tf.identity(
        self.environment_steps, name=self.name)


class NumberOfEpisodes(tf_metric.TFStepMetric):
  """Counts the number of episodes in the environment."""

  def __init__(self, name='NumberOfEpisodes', dtype=tf.int64):
    super(NumberOfEpisodes, self).__init__(name=name)
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
    return tf.identity(
        self.number_episodes, name=self.name)


class AverageReturnMetric(tf_py_metric.TFPyMetric):
  """Metric to compute the average return."""

  def __init__(self, name='AverageReturn', dtype=tf.float32, buffer_size=10):
    py_metric = py_metrics.AverageReturnMetric(buffer_size=buffer_size)

    super(AverageReturnMetric, self).__init__(
        py_metric=py_metric, name=name, dtype=dtype)


class AverageEpisodeLengthMetric(tf_py_metric.TFPyMetric):
  """Metric to compute the average episode length."""

  def __init__(self,
               name='AverageEpisodeLength',
               dtype=tf.float32,
               buffer_size=10):

    py_metric = py_metrics.AverageEpisodeLengthMetric(
        buffer_size=buffer_size)

    super(AverageEpisodeLengthMetric, self).__init__(
        py_metric=py_metric, name=name, dtype=dtype)


def log_metrics(metrics, prefix=''):
  log = ['{0} = {1}'.format(m.name, m.log().numpy()) for m in metrics]
  logging.info('%s', '{0} \n\t\t {1}'.format(prefix, '\n\t\t '.join(log)))
