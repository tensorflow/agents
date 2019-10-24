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

"""Tests for tf_agents.bandits.policies.lin_ucb_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.bandits.policies import linear_thompson_sampling_policy as lin_ts
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': 'batch1',
          'batch_size': 1,
          'num_actions': 2,
      }, {
          'testcase_name': 'batch4',
          'batch_size': 4,
          'num_actions': 5,
      })


@test_util.run_all_in_graph_and_eager_modes
class LinearThompsonSamplingPolicyTest(parameterized.TestCase,
                                       test_utils.TestCase):

  def setUp(self):
    super(LinearThompsonSamplingPolicyTest, self).setUp()
    self._obs_dim = 2
    self._obs_spec = tensor_spec.TensorSpec((self._obs_dim), tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)

  def _weight_covariance_matrices(self, num_actions):
    return [
        tf.constant(
            list(range(4 * i, 4 * (i + 1))), shape=[2, 2], dtype=tf.float32)
        for i in range(num_actions)
    ]

  def _parameter_estimators(self, num_actions):
    return [
        tf.constant(
            list(range(2 * i, 2 * (i + 1))), shape=[2], dtype=tf.float32)
        for i in list(range(num_actions))
    ]

  def _time_step_batch(self, batch_size, num_actions):
    observation = tf.constant(
        np.array(range(batch_size * self._obs_dim)),
        dtype=tf.float32,
        shape=[batch_size, self._obs_dim],
        name='observation')
    return ts.restart(observation, batch_size=batch_size)

  def _time_step_batch_with_action_mask(self, batch_size, num_actions):
    mask = tf.eye(batch_size, num_columns=num_actions, dtype=tf.int32)
    observation = (tf.constant(
        np.array(range(batch_size * self._obs_dim)),
        dtype=tf.float32,
        shape=[batch_size, self._obs_dim]), mask)
    return ts.restart(observation, batch_size=batch_size)

  @test_cases()
  def testActionBatch(self, batch_size, num_actions):
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        minimum=0,
        maximum=num_actions - 1,
        dtype=tf.int32,
        name='action')
    policy = lin_ts.LinearThompsonSamplingPolicy(
        action_spec, self._time_step_spec,
        self._weight_covariance_matrices(num_actions),
        self._parameter_estimators(num_actions))

    action_step = policy.action(self._time_step_batch(batch_size, num_actions))

    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    actions = self.evaluate(action_step.action)
    self.assertAllGreaterEqual(actions, 0)
    self.assertAllLessEqual(actions, num_actions - 1)

  def testCorrectEstimates(self):
    parameter_estimators = tf.unstack(
        tf.constant([[1, 2], [30, 40]], dtype=tf.float32))
    weight_covariance_matrices = tf.unstack(
        tf.constant([[[1, 0], [0, 1]], [[.5, 0], [0, .5]]], dtype=tf.float32))
    batch_size = 7
    observation = tf.constant(
        [6, 7] * batch_size,
        dtype=tf.float32,
        shape=[batch_size, 2],
        name='observation')
    expected_means = [[20, 920]] * batch_size
    means, variances = lin_ts._get_means_and_variances(
        parameter_estimators, weight_covariance_matrices, observation)
    self.assertAllEqual(self.evaluate(tf.stack(means, axis=-1)), expected_means)
    expected_variances = [[85, 170]] * batch_size
    self.assertAllEqual(
        self.evaluate(tf.stack(variances, axis=-1)), expected_variances)

  @test_cases()
  def testMaskedActions(self, batch_size, num_actions):
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        minimum=0,
        maximum=num_actions - 1,
        dtype=tf.int32,
        name='action')
    obs_spec = (tensor_spec.TensorSpec(self._obs_dim, tf.float32),
                tensor_spec.TensorSpec(num_actions, tf.int32))
    policy = lin_ts.LinearThompsonSamplingPolicy(
        action_spec,
        ts.time_step_spec(obs_spec),
        self._weight_covariance_matrices(num_actions),
        self._parameter_estimators(num_actions),
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]))

    action_step = policy.action(
        self._time_step_batch_with_action_mask(batch_size, num_actions))

    self.assertEqual(action_step.action.shape.as_list(), [batch_size])
    self.assertEqual(action_step.action.dtype, tf.int32)
    actions = self.evaluate(action_step.action)
    print(actions)
    self.assertAllEqual(actions, range(batch_size))


if __name__ == '__main__':
  tf.test.main()
