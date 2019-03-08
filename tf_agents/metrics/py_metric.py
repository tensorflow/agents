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

"""Base class for Python metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging

import numpy as np
import six
import tensorflow as tf

from tf_agents.metrics import tf_metric
from tf_agents.utils import common


def run_summaries(metrics, session=None):
  """Execute summary ops for py_metrics.

  Args:
    metrics: A list of py_metric.Base objects.
    session: A TensorFlow session-like object. If it is not provided, it will
      use the current TensorFlow session context manager.

  Raises:
    RuntimeError: If .tf_summaries() was not previously called on any of the
      `metrics`.
    AttributeError: If session is not provided and there is no default session
      provided by a context manager.
  """
  if session is None:
    default_session = tf.compat.v1.get_default_session()
    if default_session is None:
      raise AttributeError(
          'No TensorFlow session-like object was provided, and none '
          'could be retrieved using \'tf.get_default_session()\'.')
    session = default_session

  for metric in metrics:
    if metric.summary_op is None:
      raise RuntimeError('metric.tf_summaries() must be called on py_metric '
                         '{} before attempting to run '
                         'summaries.'.format(metric.name))
  summary_ops = [metric.summary_op for metric in metrics]
  feed_dict = dict(
      (metric.summary_placeholder, metric.result()) for metric in metrics)
  session.run(summary_ops, feed_dict=feed_dict)


@six.add_metaclass(abc.ABCMeta)
class PyMetric(tf.Module):
  """Defines the interface for metrics."""

  def __init__(self, name, prefix='Metrics'):
    """Creates a metric."""
    super(PyMetric, self).__init__(name)
    self._prefix = prefix
    self._summary_placeholder = None
    self._summary_op = None

  @property
  def prefix(self):
    """Prefix for the metric."""
    return self._prefix

  @abc.abstractmethod
  def reset(self):
    """Resets internal stat gathering variables used to compute the metric."""

  @abc.abstractmethod
  def result(self):
    """Evaluates the current value of the metric."""

  def log(self):
    tag = common.join_scope(self.prefix, self.name)
    logging.info('%s', '{0} = {1}'.format(tag, self.result()))

  def tf_summaries(self, train_step=None, step_metrics=()):
    """Build TF summary op and placeholder for this metric.

    To execute the op, call py_metric.run_summaries.

    Args:
      train_step: Step counter for training iterations. If None, no metric is
        generated against the global step.
      step_metrics: Step values to plot as X axis in addition to global_step.

    Returns:
      The summary op.

    Raises:
      RuntimeError: If this method has already been called (it can only be
        called once).
      ValueError: If any item in step_metrics is not of type PyMetric or
        tf_metric.TFStepMetric.
    """
    if self.summary_op is not None:
      raise RuntimeError('metric.tf_summaries() can only be called once.')

    tag = common.join_scope(self.prefix, self.name)
    summaries = []
    summaries.append(tf.compat.v2.summary.scalar(
        name=tag, data=self.summary_placeholder, step=train_step))
    prefix = self.prefix
    if prefix:
      prefix += '_'
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
      if isinstance(step_metric, PyMetric):
        step_tensor = step_metric.summary_placeholder
      elif isinstance(step_metric, tf_metric.TFStepMetric):
        step_tensor = step_metric.result()
      else:
        raise ValueError('step_metric is not PyMetric or TFStepMetric: '
                         '{}'.format(step_metric))
      summaries.append(tf.compat.v2.summary.scalar(
          name=step_tag,
          data=self.summary_placeholder,
          step=step_tensor))

    self._summary_op = tf.group(*summaries)
    return self._summary_op

  @property
  def summary_placeholder(self):
    """TF placeholder to be used for the result of this metric."""
    if self._summary_placeholder is None:
      result = self.result()
      if not isinstance(result, (np.ndarray, np.generic)):
        result = np.array(result)
      dtype = tf.as_dtype(result.dtype)
      shape = result.shape
      self._summary_placeholder = tf.compat.v1.placeholder(
          dtype, shape=shape, name='{}_ph'.format(self.name))
    return self._summary_placeholder

  @property
  def summary_op(self):
    """TF summary op for this metric."""
    return self._summary_op

  @staticmethod
  def aggregate(metrics):
    """Aggregates a list of metrics.

    The default behaviour is to return the average of the metrics.

    Args:
      metrics: a list of metrics, of the same class.
    Returns:
      The result of aggregating this metric.
    """
    return np.mean([metric.result() for metric in metrics])

  def __call__(self, *args):
    """Method to update the metric contents.

    To change the behavior of this function, override the call method.

    Different subclasses might use this differently. For instance, the
    PyStepMetric takes in a trajectory, while the CounterMetric takes no
    parameters.

    Args:
      *args: See call method of subclass for specific arguments.
    """
    self.call(*args)


class PyStepMetric(PyMetric):
  """Defines the interface for metrics that operate on trajectories."""

  @abc.abstractmethod
  def call(self, trajectory):
    """Processes a trajectory to update the metric.

    Args:
      trajectory: A trajectory.Trajectory.
    """
