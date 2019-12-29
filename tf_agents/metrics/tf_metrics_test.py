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

from tensorflow.python.eager import context  # TF internal


class TFDequeTest(tf.test.TestCase):

  def test_data_is_zero(self):
    d = tf_metrics.TFDeque(3, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual([], self.evaluate(d.data))

  def test_rolls_over(self):
    d = tf_metrics.TFDeque(3, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add(1))
    self.evaluate(d.add(2))
    self.evaluate(d.add(3))
    self.assertAllEqual([1, 2, 3], self.evaluate(d.data))

    self.evaluate(d.add(4))
    self.assertAllEqual([4, 2, 3], self.evaluate(d.data))

  def test_clear(self):
    d = tf_metrics.TFDeque(3, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add(1))
    self.evaluate(d.add(2))
    self.evaluate(d.add(3))
    self.assertAllEqual([1, 2, 3], self.evaluate(d.data))

    self.evaluate(d.clear())
    self.assertAllEqual([], self.evaluate(d.data))

    self.evaluate(d.add(4))
    self.evaluate(d.add(5))
    self.assertAllEqual([4, 5], self.evaluate(d.data))

  def test_mean_not_full(self):
    d = tf_metrics.TFDeque(3, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add(2))
    self.evaluate(d.add(4))
    self.assertEqual(3.0, self.evaluate(d.mean()))

  def test_mean_empty(self):
    d = tf_metrics.TFDeque(3, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertEqual(0, self.evaluate(d.mean()))

  def test_mean_roll_over(self):
    d = tf_metrics.TFDeque(3, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add(1))
    self.evaluate(d.add(2))
    self.evaluate(d.add(3))
    self.evaluate(d.add(4))
    self.assertEqual(3.0, self.evaluate(d.mean()))
    self.assertEqual(tf.float32, d.mean().dtype)

  def test_extend(self):
    d = tf_metrics.TFDeque(3, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.extend([1, 2, 3, 4]))
    self.assertEqual(3.0, self.evaluate(d.mean()))
    self.assertEqual(tf.float32, d.mean().dtype)


class TFMetricsTest(parameterized.TestCase, tf.test.TestCase):

  def _create_trajectories(self):

    def _concat_nested_tensors(nest1, nest2):
      return tf.nest.map_structure(lambda t1, t2: tf.concat([t1, t2], axis=0),
                                   nest1, nest2)

    # Order of args for trajectory methods:
    # observation, action, policy_info, reward, discount
    ts0 = _concat_nested_tensors(
        trajectory.boundary((), tf.constant([1]), (),
                            tf.constant([0.], dtype=tf.float32), [1.]),
        trajectory.boundary((), tf.constant([2]), (),
                            tf.constant([0.], dtype=tf.float32), [1.]))
    ts1 = _concat_nested_tensors(
        trajectory.first((), tf.constant([2]), (),
                         tf.constant([1.], dtype=tf.float32), [1.]),
        trajectory.first((), tf.constant([1]), (),
                         tf.constant([2.], dtype=tf.float32), [1.]))
    ts2 = _concat_nested_tensors(
        trajectory.last((), tf.constant([1]), (),
                        tf.constant([3.], dtype=tf.float32), [1.]),
        trajectory.last((), tf.constant([1]), (),
                        tf.constant([4.], dtype=tf.float32), [1.]))
    ts3 = _concat_nested_tensors(
        trajectory.boundary((), tf.constant([2]), (),
                            tf.constant([0.], dtype=tf.float32), [1.]),
        trajectory.boundary((), tf.constant([0]), (),
                            tf.constant([0.], dtype=tf.float32), [1.]))
    ts4 = _concat_nested_tensors(
        trajectory.first((), tf.constant([1]), (),
                         tf.constant([5.], dtype=tf.float32), [1.]),
        trajectory.first((), tf.constant([1]), (),
                         tf.constant([6.], dtype=tf.float32), [1.]))
    ts5 = _concat_nested_tensors(
        trajectory.last((), tf.constant([1]), (),
                        tf.constant([7.], dtype=tf.float32), [1.]),
        trajectory.last((), tf.constant([1]), (),
                        tf.constant([8.], dtype=tf.float32), [1.]))

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
      if metric_class in [tf_metrics.AverageReturnMetric,
                          tf_metrics.AverageEpisodeLengthMetric]:
        metric = metric_class(batch_size=2)
      else:
        metric = metric_class()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(metric.init_variables())
      for i in range(num_trajectories):
        self.evaluate(metric(trajectories[i]))

      self.assertEqual(expected_result, self.evaluate(metric.result()))
      self.evaluate(metric.reset())
      self.assertEqual(0.0, self.evaluate(metric.result()))

  @parameterized.named_parameters([
      ('testActionRelativeFreqGraph', context.graph_mode),
      ('testActionRelativeFreqEager', context.eager_mode),
  ])
  def testChosenActionHistogram(self, run_mode):
    with run_mode():
      trajectories = self._create_trajectories()
      num_trajectories = 5
      expected_result = [1, 2, 2, 1, 1, 1, 2, 0, 1, 1]
      metric = tf_metrics.ChosenActionHistogram(buffer_size=10)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(metric.init_variables())
      for i in range(num_trajectories):
        self.evaluate(metric(trajectories[i]))

      self.assertAllEqual(expected_result, self.evaluate(metric.result()))
      self.evaluate(metric.reset())
      self.assertEmpty(self.evaluate(metric.result()))


if __name__ == '__main__':
  tf.test.main()
