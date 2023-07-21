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

"""Wraps a python metric as a TF metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import tf_metric


@contextlib.contextmanager
def _check_not_called_concurrently(lock):
  """Checks the returned context is not executed concurrently with any other."""
  if not lock.acquire(False):  # Non-blocking.
    raise RuntimeError(
        'Detected concurrent execution of TFPyMetric ops.')
  try:
    yield
  finally:
    lock.release()


class TFPyMetric(tf_metric.TFStepMetric):
  """Wraps a python metric as a TF metric."""

  def __init__(self, py_metric, name=None, dtype=tf.float32):
    """Creates a TF metric given a py metric to wrap.

    Args:
      py_metric: A batched python metric to wrap.
      name: Name of the metric.
      dtype: Data type of the metric.
    """
    name = name or py_metric.name
    super(TFPyMetric, self).__init__(name=name, prefix=py_metric.prefix)
    self._py_metric = py_metric
    self._dtype = dtype
    self._lock = threading.Lock()

  def call(self, trajectory):
    """Update the value of the metric using trajectory.

    The trajectory can be either batched or un-batched depending on
    the expected inputs for the py_metric being wrapped.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory.

    Returns:
      The arguments, for easy chaining.
    """
    def _call(*flattened_trajectories):
      with _check_not_called_concurrently(self._lock):
        flat_sequence = [x.numpy() for x in flattened_trajectories]
        packed_trajectories = tf.nest.pack_sequence_as(
            structure=(trajectory), flat_sequence=flat_sequence)
        return self._py_metric(packed_trajectories)

    flattened_trajectories = tf.nest.flatten(trajectory)
    metric_op = tf.py_function(
        _call,
        flattened_trajectories,
        [],
        name='metric_call_py_func')

    with tf.control_dependencies([metric_op]):
      return tf.nest.map_structure(tf.identity, trajectory)

  def result(self):
    def _result():
      with _check_not_called_concurrently(self._lock):
        return self._py_metric.result()

    result_value = tf.py_function(
        _result,
        [],
        self._dtype,
        name='metric_result_py_func')
    if not tf.executing_eagerly():
      result_value.set_shape(())
    return result_value

  def reset(self):
    def _reset():
      with _check_not_called_concurrently(self._lock):
        return self._py_metric.reset()

    return tf.py_function(
        _reset, [], [],
        name='metric_reset_py_func')
