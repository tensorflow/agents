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

"""Tests for tf_agents.metrics.batched_py_metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import batched_py_metric
from tf_agents.metrics import py_metrics
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils


class BatchedPyMetricTest(tf.test.TestCase):

  def setUp(self):
    super(BatchedPyMetricTest, self).setUp()
    # Order of args for trajectory methods:
    # (observation, action, policy_info, reward, discount)
    self._ts0 = nest_utils.stack_nested_arrays([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.boundary((), (), (), 0., 1.)
    ])
    self._ts1 = nest_utils.stack_nested_arrays([
        trajectory.first((), (), (), 1., 1.),
        trajectory.first((), (), (), 2., 1.)
    ])
    self._ts2 = nest_utils.stack_nested_arrays([
        trajectory.last((), (), (), 3., 1.),
        trajectory.last((), (), (), 4., 1.)
    ])
    self._ts3 = nest_utils.stack_nested_arrays([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.boundary((), (), (), 0., 1.)
    ])
    self._ts4 = nest_utils.stack_nested_arrays([
        trajectory.first((), (), (), 5., 1.),
        trajectory.first((), (), (), 6., 1.)
    ])
    self._ts5 = nest_utils.stack_nested_arrays([
        trajectory.last((), (), (), 7., 1.),
        trajectory.last((), (), (), 8., 1.)
    ])

  def testMetricIsComputedCorrectlyNoSteps(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)
    self.assertEqual(batched_avg_return_metric.result(), 0)

  def testMetricIsComputedCorrectlyPartialEpisode(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)

    batched_avg_return_metric(self._ts0)
    batched_avg_return_metric(self._ts1)
    self.assertEqual(batched_avg_return_metric.result(), 0)

  def testMetricIsComputedCorrectlyOneEpisode(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)

    batched_avg_return_metric(self._ts0)
    batched_avg_return_metric(self._ts1)
    batched_avg_return_metric(self._ts2)

    self.assertEqual(batched_avg_return_metric.result(), 5)

  def testMetricIsComputedCorrectlyOneAndPartialEpisode(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)
    batched_avg_return_metric(self._ts0)
    batched_avg_return_metric(self._ts1)
    batched_avg_return_metric(self._ts2)
    batched_avg_return_metric(self._ts3)
    batched_avg_return_metric(self._ts4)

    self.assertEqual(batched_avg_return_metric.result(), 5)

  def testMetricIsComputedCorrectlyTwoEpisodes(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)
    batched_avg_return_metric(self._ts0)
    batched_avg_return_metric(self._ts1)
    batched_avg_return_metric(self._ts2)
    batched_avg_return_metric(self._ts3)
    batched_avg_return_metric(self._ts4)
    batched_avg_return_metric(self._ts5)
    self.assertEqual(batched_avg_return_metric.result(), 9)

  def testReset(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)
    batched_avg_return_metric(self._ts0)
    batched_avg_return_metric(self._ts1)
    batched_avg_return_metric(self._ts2)
    batched_avg_return_metric.reset()
    batched_avg_return_metric(self._ts3)
    batched_avg_return_metric(self._ts4)
    batched_avg_return_metric(self._ts5)
    self.assertEqual(batched_avg_return_metric.result(), 13)


if __name__ == '__main__':
  tf.test.main()
