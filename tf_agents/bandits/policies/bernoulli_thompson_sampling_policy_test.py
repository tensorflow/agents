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

"""Test for bernoulli_thomlson_sampling_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.policies import bernoulli_thompson_sampling_policy as bern_thompson_sampling_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal


def _prepare_alphas(action_spec, dtype=tf.float32):
  num_actions = action_spec.maximum - action_spec.minimum + 1
  alphas = [tf.compat.v2.Variable(
      tf.ones([], dtype=dtype), name='alpha_{}'.format(k)) for k in range(
          num_actions)]
  return alphas


def _prepare_betas(action_spec, dtype=tf.float32):
  num_actions = action_spec.maximum - action_spec.minimum + 1
  betas = [tf.compat.v2.Variable(
      tf.ones([], dtype=dtype), name='beta_{}'.format(k)) for k in range(
          num_actions)]
  return betas


@test_util.run_all_in_graph_and_eager_modes
class BernoulliThompsonSamplingPolicyTest(
    parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(BernoulliThompsonSamplingPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)

  def testBuild(self):
    policy = bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
        self._time_step_spec,
        self._action_spec,
        alpha=_prepare_alphas(self._action_spec),
        beta=_prepare_betas(self._action_spec))

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  def testMultipleActionsRaiseError(self):
    action_spec = [tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)] * 2
    with self.assertRaisesRegex(
        NotImplementedError,
        'action_spec can only contain a single BoundedTensorSpec'):
      bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
          self._time_step_spec,
          action_spec,
          alpha=_prepare_alphas(self._action_spec),
          beta=_prepare_betas(self._action_spec))

  def testWrongActionsRaiseError(self):
    action_spec = tensor_spec.BoundedTensorSpec((5, 6, 7), tf.float32, 0, 2)
    with self.assertRaisesRegex(
        NotImplementedError,
        'action_spec must be a BoundedTensorSpec of integer type.*'):
      bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
          self._time_step_spec,
          action_spec,
          alpha=_prepare_alphas(self._action_spec),
          beta=_prepare_betas(self._action_spec))

  def testWrongAlphaParamsSize(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 20)
    alphas = [tf.compat.v2.Variable(
        tf.zeros([], dtype=tf.float32),
        name='alpha_{}'.format(k)) for k in range(5)]
    betas = [tf.compat.v2.Variable(
        tf.zeros([], dtype=tf.float32),
        name='beta_{}'.format(k)) for k in range(5)]
    with self.assertRaisesRegex(
        ValueError,
        r'The size of alpha parameters is expected to be equal to the number'
        r' of actions, but found to be 5'):
      bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
          self._time_step_spec,
          action_spec,
          alpha=alphas,
          beta=betas)

  def testWrongBetaParamsSize(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 10, 20)
    alphas = [tf.compat.v2.Variable(
        tf.zeros([], dtype=tf.float32),
        name='alpha_{}'.format(k)) for k in range(11)]
    betas = [tf.compat.v2.Variable(
        tf.zeros([], dtype=tf.float32),
        name='beta_{}'.format(k)) for k in range(5)]
    with self.assertRaisesRegex(
        ValueError,
        r'The size of alpha parameters is expected to be equal to the size of'
        r' beta parameters'):
      bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
          self._time_step_spec,
          action_spec,
          alpha=alphas,
          beta=betas)

  def testAction(self):
    tf.compat.v1.set_random_seed(1)
    policy = bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
        self._time_step_spec,
        self._action_spec,
        alpha=_prepare_alphas(self._action_spec),
        beta=_prepare_betas(self._action_spec))
    time_step = ts.TimeStep(ts.StepType.FIRST, 0.0, 0.0, observation=1.0)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [1])
    self.assertEqual(action_step.action.dtype, tf.int32)

  def testActionBatchSizeHigherThanOne(self):
    tf.compat.v1.set_random_seed(1)
    policy = bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
        self._time_step_spec,
        self._action_spec,
        alpha=_prepare_alphas(self._action_spec),
        beta=_prepare_betas(self._action_spec))
    observations = tf.constant([[1], [1]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)

  def testMaskedAction(self):
    tf.compat.v1.set_random_seed(1)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 2)
    observation_spec = (tensor_spec.TensorSpec([], tf.float32),
                        tensor_spec.TensorSpec([3], tf.int32))
    time_step_spec = ts.time_step_spec(observation_spec)

    def split_fn(obs):
      return obs[0], obs[1]

    policy = bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
        time_step_spec,
        action_spec,
        alpha=_prepare_alphas(self._action_spec),
        beta=_prepare_betas(self._action_spec),
        observation_and_action_constraint_splitter=split_fn)

    observations = (tf.constant([[1], [1]], dtype=tf.float32),
                    tf.constant([[0, 0, 1], [1, 0, 0]], dtype=tf.int32))
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [2, 0])

  @parameterized.named_parameters([
      ('_tf.float32', tf.float32),
      ('_tf.float64', tf.float64)
    ])
  def testPredictedRewards(self, dtype):
    tf.compat.v1.set_random_seed(1)
    policy = bern_thompson_sampling_policy.BernoulliThompsonSamplingPolicy(
        self._time_step_spec,
        self._action_spec,
        alpha=_prepare_alphas(self._action_spec, dtype=dtype),
        beta=_prepare_betas(self._action_spec, dtype=dtype),
        emit_policy_info=(
            'predicted_rewards_mean', 'predicted_rewards_sampled',))
    time_step = ts.TimeStep(ts.StepType.FIRST, 0.0, 0.0, observation=1.0)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [1])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    predicted_rewards_expected_array = np.array([[0.5, 0.5, 0.5]])
    p_info = self.evaluate(action_step.info)
    self.assertAllClose(p_info.predicted_rewards_mean,
                        predicted_rewards_expected_array)
    self.assertEqual(list(p_info.predicted_rewards_sampled.shape), [1, 3])


if __name__ == '__main__':
  tf.test.main()
