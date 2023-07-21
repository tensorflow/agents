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

"""Tests for tf_agents.utils.generalized_advantage_estimation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.utils import value_ops


def _naive_gae_as_ground_truth(discounts, rewards, values, final_value,
                               td_lambda):
  """A naive GAE closely resembles equation (16) in the paper.

  Slow, for testing purpose only.
  For full paper see https://arxiv.org/abs/1506.02438.pdf
  Args:
    discounts: `np.array` with shape [T, B].
    rewards: `np.array` with shape [T, B].
    values: `np.array` with shape [T, B].
    final_value: `np.array` with shape [B].
    td_lambda: A float scalar.

  Returns:
    A `np.array` with shape[T, B] representing the advantages.
  """

  episode_length = len(values)
  values_t_puls_1 = np.concatenate([values, final_value[None, :]], axis=0)

  delta_v = [
      (rewards[t] + discounts[t] * values_t_puls_1[t + 1] - values_t_puls_1[t])
      for t in range(episode_length)
  ]
  weighted_discounts = discounts * td_lambda
  advantages = []
  for s in range(episode_length):
    advantage = np.copy(delta_v[s])
    for t in range(s + 1, episode_length):
      advantage += np.prod(weighted_discounts[s:t], axis=0) * delta_v[t]
    advantages.append(advantage)

  return np.array(advantages)


def _numpy_discounted_return(rewards, discounts, final_value):
  """A naive reward to do implemented in python.

  Slow, for testing purpose only.
  Args:
    rewards: `np.array` with shape [T, B].
    discounts: `np.array` with shape [T, B].
    final_value: `np.array` with shape [B].

  Returns:
    A `np.array` with shape[T, B] representing the target values.
  """
  if final_value is None:
    final_value = np.zeros_like(rewards[-1])

  discounted_returns = np.zeros_like(rewards)
  accumulated_rewards = final_value
  for t in reversed(range(len(rewards))):
    discounted_returns[t] = rewards[t] + discounts[t] * accumulated_rewards
    accumulated_rewards = discounted_returns[t]
  return discounted_returns


class DiscountedReturnTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_batch_single_step_without_final_value', 1, 1, False),
      ('single_batch_single_step_with_final_value', 1, 1, True),
      ('multiple_batch_multiple_step_without_final_value', 7, 9, False),
      ('multiple_batch_multiple_step_with_final_value', 7, 9, True),
  )
  def testDiscountedReturnIsCorrectlyComputed(self,
                                              num_time_steps,
                                              batch_size,
                                              with_final_value):
    rewards = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    discounts = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    final_value = np.random.rand(batch_size).astype(
        np.float32) if with_final_value else None

    discounted_return = value_ops.discounted_return(
        rewards=rewards, discounts=discounts, final_value=final_value)

    single_discounted_return = value_ops.discounted_return(
        rewards=rewards, discounts=discounts, final_value=final_value,
        provide_all_returns=False)

    expected = _numpy_discounted_return(
        rewards=rewards, discounts=discounts, final_value=final_value)

    self.assertAllClose(discounted_return, expected)
    self.assertAllClose(single_discounted_return, expected[0])

  @parameterized.named_parameters(
      ('single_batch_single_step_without_final_value', 1, 1, False),
      ('single_batch_single_step_with_final_value', 1, 1, True),
      ('multiple_batch_multiple_step_without_final_value', 7, 9, False),
      ('multiple_batch_multiple_step_with_final_value', 7, 9, True),
  )
  def testTimeMajorBatchMajorDiscountedReturnsAreSame(self,
                                                      num_time_steps,
                                                      batch_size,
                                                      with_final_value):
    rewards = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    discounts = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    final_value = np.random.rand(batch_size).astype(
        np.float32) if with_final_value else None

    time_major_discounted_return = value_ops.discounted_return(
        rewards=rewards,
        discounts=discounts,
        final_value=final_value)

    batch_major_discounted_return = value_ops.discounted_return(
        rewards=tf.transpose(rewards),
        discounts=tf.transpose(discounts),
        final_value=final_value,
        time_major=False)

    self.assertAllClose(time_major_discounted_return,
                        tf.transpose(batch_major_discounted_return))

    single_time_major_discounted_return = value_ops.discounted_return(
        rewards=rewards,
        discounts=discounts,
        final_value=final_value,
        provide_all_returns=False)

    single_batch_major_discounted_return = value_ops.discounted_return(
        rewards=tf.transpose(rewards),
        discounts=tf.transpose(discounts),
        final_value=final_value,
        time_major=False,
        provide_all_returns=False)

    self.assertAllClose(single_time_major_discounted_return,
                        time_major_discounted_return[0])
    self.assertAllClose(single_batch_major_discounted_return,
                        time_major_discounted_return[0])

  def testDiscountedReturnWithFinalValueMatchPrecomputedResult(self):
    discounted_return = value_ops.discounted_return(
        rewards=tf.constant([1] * 9, dtype=tf.float32),
        discounts=tf.constant(
            [1, 1, 1, 1, 0, 0.9, 0.9, 0.9, 0.9], dtype=tf.float32),
        final_value=tf.constant(8, dtype=tf.float32))

    expected = [
        5, 4, 3, 2, 1, 8 * 0.9**4 + 3.439, 8 * 0.9**3 + 2.71, 8 * 0.9**2 + 1.9,
        8 * 0.9 + 1
    ]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(discounted_return, expected)


class GeneralizedAdvantageEstimationTest(tf.test.TestCase,
                                         parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_batch_single_step', 1, 1, 0.7),
      ('multiple_batch_multiple_step', 7, 9, 0.7),
      ('multiple_batch_multiple_step_lambda_0', 7, 9, 0.),
      ('multiple_batch_multiple_step_lambda_1', 7, 9, 1.),
  )
  def testAdvantagesAreCorrectlyComputed(self,
                                         batch_size,
                                         num_time_steps,
                                         td_lambda):
    rewards = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    discounts = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    values = np.random.rand(num_time_steps, batch_size).astype(np.float32)
    final_value = np.random.rand(batch_size).astype(np.float32)
    ground_truth = _naive_gae_as_ground_truth(
        discounts=discounts,
        rewards=rewards,
        values=values,
        final_value=final_value,
        td_lambda=td_lambda)

    advantages = value_ops.generalized_advantage_estimation(
        discounts=discounts,
        rewards=rewards,
        values=values,
        final_value=final_value,
        td_lambda=td_lambda)

    self.assertAllClose(advantages, ground_truth)

  def testAdvantagesMatchPrecomputedResult(self):
    advantages = value_ops.generalized_advantage_estimation(
        discounts=tf.constant([[1.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0],
                               [1.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0]]),
        rewards=tf.fill([2, 9], 1.0),
        values=tf.fill([2, 9], 3.0),
        final_value=tf.fill([2], 3.0),
        td_lambda=0.95,
        time_major=False)

    # Precomputed according to equation (16) in paper.
    ground_truth = tf.constant([[
        2.0808625, 1.13775, 0.145, -0.9, -2.0, 0.56016475, -0.16355, -1.01, -2.0
    ], [
        2.0808625, 1.13775, 0.145, -0.9, -2.0, 0.56016475, -0.16355, -1.01, -2.0
    ]])

    self.assertAllClose(advantages, ground_truth)


if __name__ == '__main__':
  tf.test.main()
