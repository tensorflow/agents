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

"""Tests for tf_agents.metrics.tf_py_metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import batched_py_metric
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils


class BatchedPyMetricTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(BatchedPyMetricTest, self).setUp()
    # Order of args for trajectory methods:
    # observation, action, policy_info, reward, discount
    ts0 = nest_utils.stack_nested_tensors([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.boundary((), (), (), 0., 1.)
    ])
    ts1 = nest_utils.stack_nested_tensors([
        trajectory.first((), (), (), 1., 1.),
        trajectory.first((), (), (), 2., 1.)
    ])
    ts2 = nest_utils.stack_nested_tensors([
        trajectory.last((), (), (), 3., 1.),
        trajectory.last((), (), (), 4., 1.)
    ])
    ts3 = nest_utils.stack_nested_tensors([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.boundary((), (), (), 0., 1.)
    ])
    ts4 = nest_utils.stack_nested_tensors([
        trajectory.first((), (), (), 5., 1.),
        trajectory.first((), (), (), 6., 1.)
    ])
    ts5 = nest_utils.stack_nested_tensors([
        trajectory.last((), (), (), 7., 1.),
        trajectory.last((), (), (), 8., 1.)
    ])

    self._ts = [ts0, ts1, ts2, ts3, ts4, ts5]

  @parameterized.named_parameters(
      [('testMetricIsComputedCorrectlyNoSteps', 0, 0),
       ('testMetricIsComputedCorrectlyPartialEpisode', 2, 0),
       ('testMetricIsComputedCorrectlyOneEpisode', 3, 5),
       ('testMetricIsComputedCorrectlyOneAndPartialEpisode', 5, 5),
       ('testMetricIsComputedCorrectlyTwoEpisodes', 6, 9),
      ])
  def testMetricIsComputedCorrectly(self, num_time_steps, expected_reward):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)
    tf_avg_return_metric = tf_py_metric.TFPyMetric(batched_avg_return_metric)
    deps = []
    for i in range(num_time_steps):
      with tf.control_dependencies(deps):
        traj = tf_avg_return_metric(self._ts[i])
        deps = tf.nest.flatten(traj)
    with tf.control_dependencies(deps):
      result = tf_avg_return_metric.result()
    result_ = self.evaluate(result)
    self.assertEqual(result_, expected_reward)

  def testMetricPrefix(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric, prefix='CustomPrefix')
    self.assertEqual(batched_avg_return_metric.prefix, 'CustomPrefix')

    tf_avg_return_metric = tf_py_metric.TFPyMetric(batched_avg_return_metric)
    self.assertEqual(tf_avg_return_metric._prefix, 'CustomPrefix')

  def testReset(self):
    batched_avg_return_metric = batched_py_metric.BatchedPyMetric(
        py_metrics.AverageReturnMetric)
    tf_avg_return_metric = tf_py_metric.TFPyMetric(batched_avg_return_metric)

    deps = []
    # run one episode
    for i in range(3):
      with tf.control_dependencies(deps):
        traj = tf_avg_return_metric(self._ts[i])
        deps = tf.nest.flatten(traj)

    # reset
    with tf.control_dependencies(deps):
      reset_op = tf_avg_return_metric.reset()
      deps = [reset_op]

    # run second episode
    for i in range(3, 6):
      with tf.control_dependencies(deps):
        traj = tf_avg_return_metric(self._ts[i])
        deps = tf.nest.flatten(traj)

    # Test result is the reward for the second episode.
    with tf.control_dependencies(deps):
      result = tf_avg_return_metric.result()

    result_ = self.evaluate(result)
    self.assertEqual(result_, 13)


if __name__ == '__main__':
  tf.test.main()
