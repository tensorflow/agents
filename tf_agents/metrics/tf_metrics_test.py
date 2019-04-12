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

"""Test for tf_agents.train.tf_metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils

from tensorflow.python.eager import context  # TF internal


class TFMetricsTest(parameterized.TestCase, tf.test.TestCase):

  def _create_trajectories(self):
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

    return [ts0, ts1, ts2, ts3, ts4, ts5]

  @parameterized.named_parameters([
      ('testEnvironmentStepsGraph', context.graph_mode,
       tf_metrics.EnvironmentSteps, 5, 6),
      ('testNumberOfEpisodesGraph', context.graph_mode,
       tf_metrics.NumberOfEpisodes, 4, 2),
      ('testAverageReturnGraph', context.graph_mode,
       tf_metrics.AverageReturnMetric, 6, 9.0),
      ('testAverageEpisodeLengthGraph', context.graph_mode,
       tf_metrics.AverageEpisodeLengthMetric, 6, 2.0),
      ('testEnvironmentStepsEager', context.eager_mode,
       tf_metrics.EnvironmentSteps, 5, 6),
      ('testNumberOfEpisodesEager', context.eager_mode,
       tf_metrics.NumberOfEpisodes, 4, 2),
      ('testAverageReturnEager', context.eager_mode,
       tf_metrics.AverageReturnMetric, 6, 9.0),
      ('testAverageEpisodeLengthEager', context.eager_mode,
       tf_metrics.AverageEpisodeLengthMetric, 6, 2.0),
  ])
  def testMetric(self, run_mode, metric_class, num_trajectories,
                 expected_result):
    with run_mode():
      trajectories = self._create_trajectories()
      metric = metric_class()
      deps = []
      self.evaluate(metric.init_variables())
      for i in range(num_trajectories):
        with tf.control_dependencies(deps):
          traj = metric(trajectories[i])
          deps = tf.nest.flatten(traj)
      with tf.control_dependencies(deps):
        result = metric.result()
      result_ = self.evaluate(result)
      self.assertEqual(result_, expected_result)

if __name__ == '__main__':
  tf.test.main()
