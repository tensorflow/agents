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

"""Test for tf_agents.bandits.metrics.tf_metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tensorflow.python.eager import context  # pylint: disable=g-direct-tensorflow-import  # TF internal


def compute_optimal_reward(unused_observation):
  return tf.constant(10.0)


def compute_optimal_action(unused_observation):
  return tf.constant(5, dtype=tf.int32)


class TFMetricsTest(parameterized.TestCase, tf.test.TestCase):

  def _create_trajectory(self):
    return trajectory.Trajectory(observation=(),
                                 action=(tf.constant(1)),
                                 policy_info=(),
                                 reward=tf.constant(1.0),
                                 discount=tf.constant(1.0),
                                 step_type=ts.StepType.FIRST,
                                 next_step_type=ts.StepType.LAST)

  def _create_batched_trajectory(self, batch_size):
    return trajectory.Trajectory(observation=(),
                                 action=tf.range(batch_size, dtype=tf.int32),
                                 policy_info=(),
                                 reward=tf.range(batch_size, dtype=tf.float32),
                                 discount=tf.ones(batch_size),
                                 step_type=ts.StepType.FIRST,
                                 next_step_type=ts.StepType.LAST)

  @parameterized.named_parameters(
      ('RegretMetricName', tf_metrics.RegretMetric, compute_optimal_reward,
       'RegretMetric'),
      ('SuboptimalArmsMetricName', tf_metrics.SuboptimalArmsMetric,
       compute_optimal_action, 'SuboptimalArmsMetric')
  )
  def testName(self, metric_class, fn, expected_name):
    metric = metric_class(fn)
    self.assertEqual(expected_name, metric.name)

  @parameterized.named_parameters([
      ('TestRegretGraph', context.graph_mode, tf_metrics.RegretMetric,
       compute_optimal_reward, 9),
      ('TestRegretEager', context.eager_mode, tf_metrics.RegretMetric,
       compute_optimal_reward, 9),
      ('TestSuboptimalArmsGraph', context.graph_mode,
       tf_metrics.SuboptimalArmsMetric, compute_optimal_action, 1),
      ('TestSuboptimalArmsEager', context.eager_mode,
       tf_metrics.SuboptimalArmsMetric, compute_optimal_action, 1),
  ])
  def testRegretMetric(self, run_mode, metric_class, fn, expected_result):
    with run_mode():
      traj = self._create_trajectory()
      metric = metric_class(fn)
      self.evaluate(metric.init_variables())
      traj_out = metric(traj)
      deps = tf.nest.flatten(traj_out)
      with tf.control_dependencies(deps):
        result = metric.result()
      result_ = self.evaluate(result)
      self.assertEqual(result_, expected_result)

  @parameterized.named_parameters([
      ('TestRegretGraphBatched', context.graph_mode, tf_metrics.RegretMetric,
       compute_optimal_reward, 8, 6.5),
      ('TestRegretEagerBatched', context.eager_mode, tf_metrics.RegretMetric,
       compute_optimal_reward, 8, 6.5),
      ('TestSuboptimalArmsGraphBatched', context.graph_mode,
       tf_metrics.SuboptimalArmsMetric, compute_optimal_action, 8, 7.0 / 8.0),
      ('TestSuboptimalArmsEagerBatched', context.eager_mode,
       tf_metrics.SuboptimalArmsMetric, compute_optimal_action, 8, 7.0 / 8.0),
  ])
  def testRegretMetricBatched(self, run_mode, metric_class, fn, batch_size,
                              expected_result):
    with run_mode():
      traj = self._create_batched_trajectory(batch_size)
      metric = metric_class(fn)
      self.evaluate(metric.init_variables())
      traj_out = metric(traj)
      deps = tf.nest.flatten(traj_out)
      with tf.control_dependencies(deps):
        result = metric.result()
      result_ = self.evaluate(result)
      self.assertEqual(result_, expected_result)


if __name__ == '__main__':
  tf.test.main()
