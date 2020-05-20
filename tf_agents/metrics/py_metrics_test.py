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

"""Tests for tf_agents.metrics.py_metrics."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import py_metrics
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils


class PyMetricsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric, 'AverageReturn'),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric,
       'AverageEpisodeLength'),
      ('EnvironmentSteps', py_metrics.EnvironmentSteps, 'EnvironmentSteps'),
      ('NumberOfEpisodes', py_metrics.NumberOfEpisodes, 'NumberOfEpisodes'),
      ('CounterMetric', py_metrics.CounterMetric, 'Counter'))
  def testName(self, metric_class, expected_name):
    metric = metric_class()
    self.assertEqual(expected_name, metric.name)

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric),
      ('EnvironmentSteps', py_metrics.EnvironmentSteps),
      ('NumberOfEpisodes', py_metrics.NumberOfEpisodes),
      ('CounterMetric', py_metrics.NumberOfEpisodes))
  def testChangeName(self, metric_class):
    name = 'SomeMetric'
    metric = metric_class(name)
    self.assertEqual(metric.name, name)

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric, 0.0),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric,
       0.0),
      ('EnvironmentSteps', py_metrics.EnvironmentSteps, 1.0),
      ('NumberOfEpisodes', py_metrics.NumberOfEpisodes, 0.0))
  def testZeroEpisodes(self, metric_class, expected_result):
    metric = metric_class()
    # Order of args for trajectory methods:
    # observation, action, policy_info, reward, discount
    metric(trajectory.boundary((), (), (), 0., 1.))
    metric(trajectory.first((), (), (), 1., 1.))
    self.assertEqual(expected_result, metric.result())

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric, 6.0),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric,
       3.0),
      ('EnvironmentSteps', py_metrics.EnvironmentSteps, 3.0),
      ('NumberOfEpisodes', py_metrics.NumberOfEpisodes, 1.0))
  def testAverageOneEpisode(self, metric_class, expected_result):
    metric = metric_class()

    metric(trajectory.boundary((), (), (), 0., 1.))
    metric(trajectory.mid((), (), (), 1., 1.))
    metric(trajectory.mid((), (), (), 2., 1.))
    metric(trajectory.last((), (), (), 3., 0.))
    self.assertEqual(expected_result, metric.result())

  @parameterized.named_parameters(('AverageReturnMetric',
                                   py_metrics.AverageReturnMetric, 7.0))
  def testAverageOneEpisodeWithReset(self, metric_class, expected_result):
    metric = metric_class()

    metric(trajectory.first((), (), (), 0., 1.))
    metric(trajectory.mid((), (), (), 1., 1.))
    metric(trajectory.mid((), (), (), 2., 1.))
    # The episode is reset.
    #
    # This could happen when using the dynamic_episode_driver with
    # parallel_py_environment. When the parallel episodes are of different
    # lengths and num_episodes is reached, some episodes would be left in "MID".
    # When the driver runs again, all environments are reset at the beginning
    # of the tf.while_loop and the unfinished episodes would get "FIRST" without
    # seeing "LAST".
    metric(trajectory.first((), (), (), 3., 1.))
    metric(trajectory.last((), (), (), 4., 1.))
    self.assertEqual(expected_result, metric.result())

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric, 0.0),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric,
       2.0),
      ('EnvironmentSteps', py_metrics.EnvironmentSteps, 4.0),
      ('NumberOfEpisodes', py_metrics.NumberOfEpisodes, 2.0))
  def testAverageTwoEpisode(self, metric_class, expected_result):
    metric = metric_class()

    metric(trajectory.boundary((), (), (), 0., 1.))
    metric(trajectory.first((), (), (), 1., 1.))
    metric(trajectory.mid((), (), (), 2., 1.))
    metric(trajectory.last((), (), (), 3., 0.))
    metric(trajectory.boundary((), (), (), 0., 1.))

    # TODO(kbanoop): Add optional next_step_type arg to trajectory.first. Or
    # implement trajectory.first_last().
    metric(
        trajectory.Trajectory(ts.StepType.FIRST, (), (), (), ts.StepType.LAST,
                              -6., 1.))

    self.assertEqual(expected_result, metric.result())

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric, 5.0),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric,
       2.5))
  def testBatch(self, metric_class, expected_result):
    metric = metric_class()

    metric(nest_utils.stack_nested_arrays([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.boundary((), (), (), 0., 1.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.first((), (), (), 1., 1.),
        trajectory.first((), (), (), 1., 1.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.mid((), (), (), 2., 1.),
        trajectory.last((), (), (), 3., 0.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.last((), (), (), 3., 0.),
        trajectory.boundary((), (), (), 0., 1.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.first((), (), (), 1., 1.)]))
    self.assertEqual(expected_result, metric.result(), 5.0)

  @parameterized.named_parameters(
      ('AverageReturnMetric', py_metrics.AverageReturnMetric, 5.0),
      ('AverageEpisodeLengthMetric', py_metrics.AverageEpisodeLengthMetric,
       2.5))
  def testBatchSizeProvided(self, metric_class, expected_result):
    metric = metric_class(batch_size=2)

    metric(nest_utils.stack_nested_arrays([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.boundary((), (), (), 0., 1.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.first((), (), (), 1., 1.),
        trajectory.first((), (), (), 1., 1.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.mid((), (), (), 2., 1.),
        trajectory.last((), (), (), 3., 0.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.last((), (), (), 3., 0.),
        trajectory.boundary((), (), (), 0., 1.)]))
    metric(nest_utils.stack_nested_arrays([
        trajectory.boundary((), (), (), 0., 1.),
        trajectory.first((), (), (), 1., 1.)]))
    self.assertEqual(metric.result(), expected_result)

  def testCounterMetricIncrements(self):
    counter = py_metrics.CounterMetric()

    self.assertEqual(0, counter.result())
    counter()
    self.assertEqual(1, counter.result())
    counter()
    self.assertEqual(2, counter.result())
    counter.reset()
    self.assertEqual(0, counter.result())
    counter()
    self.assertEqual(1, counter.result())

  def testSaveRestore(self):
    metrics = [
        py_metrics.AverageReturnMetric(),
        py_metrics.AverageEpisodeLengthMetric(),
        py_metrics.EnvironmentSteps(),
        py_metrics.NumberOfEpisodes()
    ]

    for metric in metrics:
      metric(trajectory.boundary((), (), (), 0., 1.))
      metric(trajectory.mid((), (), (), 1., 1.))
      metric(trajectory.mid((), (), (), 2., 1.))
      metric(trajectory.last((), (), (), 3., 0.))

    checkpoint = tf.train.Checkpoint(**{m.name: m for m in metrics})
    prefix = self.get_temp_dir() + '/ckpt'
    save_path = checkpoint.save(prefix)
    for metric in metrics:
      metric.reset()
      self.assertEqual(0, metric.result())
    checkpoint.restore(save_path).assert_consumed()
    for metric in metrics:
      self.assertGreater(metric.result(), 0)


class NumpyDequeTest(tf.test.TestCase):

  def testSimple(self):
    buf = py_metrics.NumpyDeque(maxlen=10, dtype=np.float64)
    buf.add(2)
    buf.add(3)
    buf.add(5)
    buf.add(6)
    self.assertEqual(4, buf.mean())

  def testFullLength(self):
    buf = py_metrics.NumpyDeque(maxlen=4, dtype=np.float64)
    buf.add(2)
    buf.add(3)
    buf.add(5)
    buf.add(6)
    self.assertEqual(4, buf.mean())

  def testPastMaxLen(self):
    buf = py_metrics.NumpyDeque(maxlen=4, dtype=np.float64)
    buf.add(2)
    buf.add(3)
    buf.add(5)
    buf.add(6)
    buf.add(8)
    buf.add(9)
    self.assertEqual(7, buf.mean())

  def testClear(self):
    buf = py_metrics.NumpyDeque(maxlen=4, dtype=np.float64)
    buf.add(2)
    buf.add(3)
    buf.clear()
    buf.add(5)
    self.assertEqual(5, buf.mean())

  def testUnbounded(self):
    buf = py_metrics.NumpyDeque(maxlen=np.inf, dtype=np.float64)
    for i in range(101):
      buf.add(i)
    self.assertEqual(50, buf.mean())

  def testUnboundedClear(self):
    buf = py_metrics.NumpyDeque(maxlen=np.inf, dtype=np.float64)
    for i in range(101):
      buf.add(i)
    buf.clear()
    buf.add(4)
    buf.add(6)
    self.assertEqual(5, buf.mean())


if __name__ == '__main__':
  tf.test.main()
