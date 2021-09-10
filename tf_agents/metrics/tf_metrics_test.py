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

"""Test for tf_agents.train.tf_metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import tf_metrics
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

from tensorflow.python.eager import context  # pylint: disable=g-direct-tensorflow-import  # TF internal


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
    self.assertEqual(3, self.evaluate(d.mean()))
    self.assertEqual(tf.int32, d.mean().dtype)

  def test_queue_empty(self):
    d = tf_metrics.TFDeque(3, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertEqual(0, self.evaluate(d.mean()))
    self.assertEqual(tf.int32.min, self.evaluate(d.max()))
    self.assertEqual(tf.int32.max, self.evaluate(d.min()))
    self.assertEqual(tf.int32, d.mean().dtype)
    self.assertEqual(tf.int32, d.min().dtype)
    self.assertEqual(tf.int32, d.max().dtype)

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

  def test_min(self):
    d = tf_metrics.TFDeque(4, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(d.extend([1, 2, 3, 4]))
    self.assertEqual(1.0, self.evaluate(d.min()))

  def test_max(self):
    d = tf_metrics.TFDeque(4, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(d.extend([1, 2, 3, 4]))
    self.assertEqual(4.0, self.evaluate(d.max()))


class TFShapedDequeTest(tf.test.TestCase):

  def test_data_is_zero(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    values = self.evaluate(d.data)
    self.assertAllEqual((0, 2), values.shape)

  def test_rolls_over(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add([1, 1]))
    self.evaluate(d.add([2, 2]))
    self.evaluate(d.add([3, 3]))
    self.assertAllEqual([[1, 1], [2, 2], [3, 3]], self.evaluate(d.data))

    self.evaluate(d.add([4, 4]))
    self.assertAllEqual([[4, 4], [2, 2], [3, 3]], self.evaluate(d.data))

  def test_clear(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add([1, 1]))

    self.evaluate(d.clear())
    self.assertAllEqual((0, 2), self.evaluate(d.data).shape)

    self.evaluate(d.add([2, 2]))
    self.evaluate(d.add([3, 3]))
    self.assertAllEqual([[2, 2], [3, 3]], self.evaluate(d.data))

  def test_mean_full(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add([1, 1]))
    self.evaluate(d.add([2, 2]))
    self.evaluate(d.add([3, 3]))
    self.assertAllEqual([2, 2], self.evaluate(d.mean()))
    self.assertEqual(tf.int32, d.mean().dtype)

  def test_mean_not_full(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add([1, 1]))
    self.evaluate(d.add([3, 3]))
    self.assertAllEqual([2, 2], self.evaluate(d.mean()))
    self.assertEqual(tf.int32, d.mean().dtype)

  def test_mean_empty(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllEqual([0, 0], self.evaluate(d.mean()))
    self.assertEqual(tf.int32, d.mean().dtype)

  def test_mean_roll_over(self):
    d = tf_metrics.TFDeque(3, tf.float32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.add([1, 1]))
    self.evaluate(d.add([2, 2]))
    self.evaluate(d.add([3, 3]))
    self.evaluate(d.add([4, 4]))
    self.assertAllEqual([3.0, 3.0], self.evaluate(d.mean()))
    self.assertEqual(tf.float32, d.mean().dtype)

  def test_extend(self):
    d = tf_metrics.TFDeque(3, tf.int32, shape=(2,))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.evaluate(d.extend([[1, 1], [2, 2], [3, 3], [4, 4]]))
    self.assertAllEqual([[4, 4], [2, 2], [3, 3]], self.evaluate(d.data))
    self.assertAllEqual([3, 3], self.evaluate(d.mean()))


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

  def _create_misaligned_trajectories(self):

    def _concat_nested_tensors(nest1, nest2):
      return tf.nest.map_structure(lambda t1, t2: tf.concat([t1, t2], axis=0),
                                   nest1, nest2)

    # Order of args for trajectory methods:
    # observation, action, policy_info, reward, discount
    ts1 = _concat_nested_tensors(
        trajectory.first((), tf.constant([2]), (),
                         tf.constant([1.], dtype=tf.float32), [1.]),
        trajectory.boundary((), tf.constant([1]), (),
                            tf.constant([0.], dtype=tf.float32), [1.]))
    ts2 = _concat_nested_tensors(
        trajectory.last((), tf.constant([1]), (),
                        tf.constant([3.], dtype=tf.float32), [1.]),
        trajectory.first((), tf.constant([1]), (),
                         tf.constant([2.], dtype=tf.float32), [1.]))
    ts3 = _concat_nested_tensors(
        trajectory.boundary((), tf.constant([2]), (),
                            tf.constant([0.], dtype=tf.float32), [1.]),
        trajectory.last((), tf.constant([1]), (),
                        tf.constant([4.], dtype=tf.float32), [1.]))

    return [ts1, ts2, ts3]

  @parameterized.named_parameters([
      ('testEnvironmentStepsGraph', context.graph_mode,
       tf_metrics.EnvironmentSteps, 5, 6, 0.0),
      ('testNumberOfEpisodesGraph', context.graph_mode,
       tf_metrics.NumberOfEpisodes, 4, 2, 0.0),
      ('testAverageReturnGraph', context.graph_mode,
       tf_metrics.AverageReturnMetric, 6, 9.0, 0.0),
      ('testMaxReturnGraph', context.graph_mode,
       tf_metrics.MaxReturnMetric, 6, 14.0, tf.float32.min),
      ('testMinReturnGraph', context.graph_mode,
       tf_metrics.MinReturnMetric, 6, 4.0, tf.float32.max),
      ('testAverageEpisodeLengthGraph', context.graph_mode,
       tf_metrics.AverageEpisodeLengthMetric, 6, 2.0, 0.0),
      ('testEnvironmentStepsEager', context.eager_mode,
       tf_metrics.EnvironmentSteps, 5, 6, 0.0),
      ('testNumberOfEpisodesEager', context.eager_mode,
       tf_metrics.NumberOfEpisodes, 4, 2, 0.0),
      ('testAverageReturnEager', context.eager_mode,
       tf_metrics.AverageReturnMetric, 6, 9.0, 0.0),
      ('testMaxReturnEager', context.eager_mode,
       tf_metrics.MaxReturnMetric, 6, 14.0, tf.float32.min),
      ('testMinReturnEager', context.eager_mode,
       tf_metrics.MinReturnMetric, 6, 4.0, tf.float32.max),
      ('testAverageEpisodeLengthEager', context.eager_mode,
       tf_metrics.AverageEpisodeLengthMetric, 6, 2.0, 0.0),
  ])
  def testMetric(self, run_mode, metric_class, num_trajectories,
                 expected_result, empty_queue_expected_result):
    with run_mode():
      trajectories = self._create_trajectories()
      if metric_class in [tf_metrics.AverageReturnMetric,
                          tf_metrics.MaxReturnMetric,
                          tf_metrics.MinReturnMetric,
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
      self.assertEqual(empty_queue_expected_result,
                       self.evaluate(metric.result()))

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

  @parameterized.named_parameters([
      ('testAverageReturnMultiMetricGraph', context.graph_mode, 6,
       tensor_spec.TensorSpec((2,), tf.float32, 'r'), [9.0, 9.0]),
      ('testAverageReturnMultiMetricEager', context.eager_mode, 6,
       tensor_spec.TensorSpec((2,), tf.float32, 'r'), [9.0, 9.0]),
      ('testAverageReturnMultiMetricRewardSpecListGraph', context.graph_mode, 6,
       [tensor_spec.TensorSpec((), tf.float32, 'r1'),
        tensor_spec.TensorSpec((), tf.float32, 'r2')], [9.0, 9.0]),
      ('testAverageReturnMultiMetricRewardSpecListEager', context.eager_mode, 6,
       [tensor_spec.TensorSpec((), tf.float32, 'r1'),
        tensor_spec.TensorSpec((), tf.float32, 'r2')], [9.0, 9.0]),
      ('testAverageReturnMultiMetricRewardSpecDictEager', context.eager_mode, 6,
       {'a': tensor_spec.TensorSpec((), tf.float32, 'r1'),
        'b': tensor_spec.TensorSpec((), tf.float32, 'r2')},
       {'a': 9.0, 'b': 9.0})
  ])
  def testAverageReturnMultiMetric(self, run_mode, num_trajectories,
                                   reward_spec, expected_result):
    with run_mode():
      trajectories = self._create_trajectories()
      multi_trajectories = []
      for traj in trajectories:
        if isinstance(reward_spec, list):
          new_reward = [traj.reward, traj.reward]
        elif isinstance(reward_spec, dict):
          new_reward = {'a': traj.reward, 'b': traj.reward}
        else:
          new_reward = tf.stack([traj.reward, traj.reward], axis=1)
        new_traj = trajectory.Trajectory(
            step_type=traj.step_type,
            observation=traj.observation,
            action=traj.action,
            policy_info=traj.policy_info,
            next_step_type=traj.next_step_type,
            reward=new_reward,
            discount=traj.discount)
        multi_trajectories.append(new_traj)

      metric = tf_metrics.AverageReturnMultiMetric(reward_spec, batch_size=2)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(metric.init_variables())
      for i in range(num_trajectories):
        self.evaluate(metric(multi_trajectories[i]))

      self.assertAllClose(expected_result, self.evaluate(metric.result()))
      self.evaluate(metric.reset())
      reset_result = tf.nest.map_structure(tf.zeros_like, expected_result)
      self.assertAllClose(reset_result, self.evaluate(metric.result()))

  @parameterized.named_parameters([
      ('testAverageReturnMultiMetricTimeMisalignedGraph', context.graph_mode, 3,
       tensor_spec.TensorSpec((2,), tf.float32, 'r'), [5.0, 5.0]),
      ('testAverageReturnMultiMetricTimeMisalignedEager', context.eager_mode, 3,
       tensor_spec.TensorSpec((2,), tf.float32, 'r'), [5.0, 5.0]),
      ('testAverageReturnMultiMetricRewardSpecListTimeMisalignedGraph',
       context.graph_mode, 3,
       [tensor_spec.TensorSpec((), tf.float32, 'r1'),
        tensor_spec.TensorSpec((), tf.float32, 'r2')], [5.0, 5.0]),
      ('testAverageReturnMultiMetricRewardSpecListTimeMisalignedEager',
       context.eager_mode, 3,
       [tensor_spec.TensorSpec((), tf.float32, 'r1'),
        tensor_spec.TensorSpec((), tf.float32, 'r2')], [5.0, 5.0])
  ])
  def testAverageReturnMultiMetricTimeMisalignment(
      self, run_mode, num_trajectories, reward_spec, expected_result):
    with run_mode():
      trajectories = self._create_misaligned_trajectories()
      multi_trajectories = []
      for traj in trajectories:
        if isinstance(reward_spec, list):
          new_reward = [traj.reward, traj.reward]
        else:
          new_reward = tf.stack([traj.reward, traj.reward], axis=1)
        new_traj = trajectory.Trajectory(
            step_type=traj.step_type,
            observation=traj.observation,
            action=traj.action,
            policy_info=traj.policy_info,
            next_step_type=traj.next_step_type,
            reward=new_reward,
            discount=traj.discount)
        multi_trajectories.append(new_traj)

      metric = tf_metrics.AverageReturnMultiMetric(reward_spec, batch_size=2)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(metric.init_variables())
      for i in range(num_trajectories):
        self.evaluate(metric(multi_trajectories[i]))

      self.assertAllEqual(expected_result, self.evaluate(metric.result()))
      self.evaluate(metric.reset())
      self.assertAllEqual([0.0, 0.0], self.evaluate(metric.result()))

if __name__ == '__main__':
  tf.test.main()
