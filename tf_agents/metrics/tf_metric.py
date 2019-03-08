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

"""Base class for TensorFlow metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.utils import common


class TFStepMetric(tf.Module):
  """Defines the interface for TF metrics."""

  def __init__(self, name, prefix='Metrics'):
    super(TFStepMetric, self).__init__(name)
    self._prefix = prefix

  def call(self, *args, **kwargs):
    """Accumulates statistics for the metric. Users should use __call__ instead.

    Note: This function is executed as a graph function in graph mode.
    This means:
    a) Operations on the same resource are executed in textual order.
       This should make it easier to do things like add the updated
       value of a variable to another, for example.
    b) You don't need to worry about collecting the update ops to execute.
       All update ops added to the graph by this function will be executed.
    As a result, code should generally work the same way with graph or
    eager execution.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric, as passed to
        `__call__()`.
    """
    raise NotImplementedError('Metrics must define a call() member function')

  def result(self):
    """Computes and returns a final value for the metric."""
    raise NotImplementedError('Metrics must define a result() member function')

  def init_variables(self):
    """Initializes this Metric's variables.

    Should be called after variables are created in the first execution
    of `__call__()`. If using graph execution, the return value should be
    `run()` in a session before running the op returned by `__call__()`.
    (See example above.)

    Returns:
      If using graph execution, this returns an op to perform the
      initialization. Under eager execution, the variables are reset to their
      initial values as a side effect and this function returns None.
    """
    if not tf.executing_eagerly():
      return tf.compat.v1.group([v.initializer for v in self.variables])

  @common.function
  def _update_state(self, *arg, **kwargs):
    """A function wrapping the implementor-defined call method."""
    return self.call(*arg, **kwargs)

  def __call__(self, *args, **kwargs):
    """Returns op to execute to update this metric for these inputs.

    Returns None if eager execution is enabled.
    Returns a graph-mode function if graph execution is enabled.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric, passed on to `call()`.
    """
    return self._update_state(*args, **kwargs)

  def tf_summaries(self, train_step=None, step_metrics=()):
    """Generates summaries against train_step and all step_metrics.

    Args:
      train_step: (Optional) Step counter for training iterations. If None, no
        metric is generated against the global step.
      step_metrics: (Optional) Iterable of step metrics to generate summaries
        against.

    Returns:
      A list of summaries.
    """
    summaries = []
    prefix = self._prefix
    tag = common.join_scope(prefix, self.name)
    result = self.result()
    if train_step is not None:
      summaries.append(
          tf.compat.v2.summary.scalar(name=tag, data=result, step=train_step))
    if prefix:
      prefix += '_'
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
      step = step_metric.result()
      summaries.append(tf.compat.v2.summary.scalar(
          name=step_tag,
          data=result,
          step=step))
    return summaries
