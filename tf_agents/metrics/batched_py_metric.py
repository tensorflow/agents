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

"""A python metric that can be called with batches of trajectories."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import numpy as np
from tf_agents.metrics import py_metric
from tf_agents.trajectories import trajectory as traj
from tf_agents.typing import types
from tf_agents.utils import nest_utils

from typing import Any, Optional, Text


class BatchedPyMetric(py_metric.PyStepMetric):
  """Wrapper for batching metrics.

  This can be used to wrap any python metric that takes a single trajectory to
  produce a batched version of the metric that takes a batch of trajectories.
  """

  def __init__(self,
               metric_class: py_metric.PyMetric.__class__,
               metric_args: Optional[Any] = None,
               name: Optional[Text] = None,
               batch_size: Optional[types.Int] = None,
               dtype: np.dtype = np.float32):
    """Creates a BatchedPyMetric metric."""
    self._metric_class = metric_class
    if metric_args is None:
      self._metric_args = {}
    else:
      self._metric_args = metric_args

    if not name:
      name = self._metric_class(**self._metric_args).name
    super(BatchedPyMetric, self).__init__(name)

    self._built = False
    self._dtype = dtype
    if batch_size is not None:
      self.build(batch_size)

  def build(self, batch_size: types.Int):
    self._metrics = [self._metric_class(**self._metric_args)
                     for _ in range(batch_size)]
    for metric in self._metrics:
      metric.reset()
    self._built = True

  def call(self, batched_trajectory: traj.Trajectory):
    """Processes the batched_trajectory to update the metric.

    Args:
      batched_trajectory: A Trajectory containing batches of experience.

    Raises:
      ValueError: If the batch size is an unexpected value.
    """
    trajectories = nest_utils.unstack_nested_arrays(batched_trajectory)
    batch_size = len(trajectories)
    if not self._built:
      self.build(batch_size)
    if batch_size != len(self._metrics):
      raise ValueError('Batch size {} does not match previously set batch '
                       'size {}. Make sure your batch size is set correctly '
                       'in BatchedPyMetric initialization and that the batch '
                       'size remains constant.'.format(batch_size,
                                                       len(self._metrics)))

    for metric, trajectory in zip(self._metrics, trajectories):
      metric(trajectory)

  def reset(self):
    """Resets internal stat gathering variables used to compute the metric."""
    if self._built:
      for metric in self._metrics:
        metric.reset()

  def result(self) -> Any:
    """Evaluates the current value of the metric."""
    if self._built:
      return self._metric_class.aggregate(self._metrics)
    else:
      return np.array(0.0, dtype=self._dtype)

  @staticmethod
  def aggregate(metrics):
    raise NotImplementedError(
        'aggregate() is not implemented for BatchedPyMetric.')
