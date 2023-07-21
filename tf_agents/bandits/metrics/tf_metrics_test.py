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

"""Test for tf_agents.bandits.metrics.tf_metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.metrics import tf_metrics
from tf_agents.bandits.policies import constraints
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


def compute_optimal_reward(unused_observation):
  return tf.constant(10.0)


def compute_optimal_action(unused_observation):
  return tf.constant(5, dtype=tf.int32)


class SimpleThresholdConstraint(constraints.BaseConstraint):

  def __init__(self, time_step_spec, action_spec, batch_size, threshold,
               name=None):
    self.batch_size = batch_size
    self.threshold = threshold
    super(SimpleThresholdConstraint, self).__init__(
        time_step_spec, action_spec, name='SimpleThresholdConstraint')

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    if actions is None:
      actions = tf.range(
          self._action_spec.minimum, self._action_spec.maximum + 1)
      actions = tf.reshape(actions, [1, -1])
      actions = tf.tile(actions, [self.batch_size, 1])
    feasibility_prob = tf.cast(tf.greater(actions, self.threshold), tf.float32)
    return feasibility_prob


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

  def _create_test_trajectory(self, batch_size):
    num_actions = tf.cast(batch_size / 2, dtype=tf.int32)
    action_tensor = tf.concat([
        tf.range(num_actions, dtype=tf.int32),
        tf.range(num_actions, dtype=tf.int32)], axis=-1)
    return trajectory.Trajectory(observation=tf.ones(batch_size),
                                 action=action_tensor,
                                 policy_info=(),
                                 reward=tf.range(batch_size, dtype=tf.float32),
                                 discount=tf.ones(batch_size),
                                 step_type=ts.StepType.FIRST,
                                 next_step_type=ts.StepType.LAST)

  def _create_batched_trajectory_with_reward_dict(self, batch_size):
    reward_dict = {
        'reward': tf.range(batch_size, dtype=tf.float32),
        'constraint': tf.range(batch_size, dtype=tf.float32),
    }
    return trajectory.Trajectory(observation=(),
                                 action=tf.range(batch_size, dtype=tf.int32),
                                 policy_info=(),
                                 reward=reward_dict,
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
      ('TestRegret', tf_metrics.RegretMetric,
       compute_optimal_reward, 9),
      ('TestSuboptimalArms',
       tf_metrics.SuboptimalArmsMetric, compute_optimal_action, 1),
  ])
  def testRegretMetric(self, metric_class, fn, expected_result):
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
      ('TestRegretBatched', tf_metrics.RegretMetric,
       compute_optimal_reward, 8, 6.5),
      ('TestSuboptimalArmsBatched',
       tf_metrics.SuboptimalArmsMetric, compute_optimal_action, 8, 7.0 / 8.0),
  ])
  def testRegretMetricBatched(self, metric_class, fn, batch_size,
                              expected_result):
    traj = self._create_batched_trajectory(batch_size)
    metric = metric_class(fn)
    self.evaluate(metric.init_variables())
    traj_out = metric(traj)
    deps = tf.nest.flatten(traj_out)
    with tf.control_dependencies(deps):
      result = metric.result()
    result_ = self.evaluate(result)
    self.assertEqual(result_, expected_result)

  def testRegretMetricWithRewardDict(
      self, metric_class=tf_metrics.RegretMetric, fn=compute_optimal_reward,
      batch_size=8, expected_result=6.5):
    traj = self._create_batched_trajectory_with_reward_dict(batch_size)
    metric = metric_class(fn)
    self.evaluate(metric.init_variables())
    traj_out = metric(traj)
    deps = tf.nest.flatten(traj_out)
    with tf.control_dependencies(deps):
      result = metric.result()
    result_ = self.evaluate(result)
    self.assertEqual(result_, expected_result)

  @parameterized.named_parameters([
      ('TestConstraintViolationTh1', 8, 1, 0.5),
      ('TestConstraintViolationTh2', 8, 2, 0.75),
  ])
  def testConstraintViolationMetric(
      self, batch_size, threshold, expected_result):
    traj = self._create_test_trajectory(batch_size)
    num_actions = batch_size / 2

    obs_spec = tensor_spec.TensorSpec([], tf.float32)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions-1)
    stc = SimpleThresholdConstraint(
        time_step_spec, action_spec, batch_size=batch_size,
        threshold=threshold)
    metric = tf_metrics.ConstraintViolationsMetric(constraint=stc)
    self.evaluate(metric.init_variables())
    traj_out = metric(traj)
    deps = tf.nest.flatten(traj_out)
    with tf.control_dependencies(deps):
      result = metric.result()
    result_ = self.evaluate(result)
    self.assertEqual(result_, expected_result)

  def testDistanceFromGreedyMetric(self):
    batch_size = 11
    num_actions = 12
    traj = self._create_batched_trajectory(batch_size)
    def estimated_reward_fn(unused_observation):
      return tf.stack([tf.range(num_actions, dtype=tf.float32)] * batch_size)

    metric = tf_metrics.DistanceFromGreedyMetric(estimated_reward_fn)
    self.evaluate(metric.init_variables())
    traj_out = metric(traj)
    deps = tf.nest.flatten(traj_out)
    with tf.control_dependencies(deps):
      result = metric.result()
    result_ = self.evaluate(result)
    self.assertEqual(result_, 6)


if __name__ == '__main__':
  tf.test.main()
