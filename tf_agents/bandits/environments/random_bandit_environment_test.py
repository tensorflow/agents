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

"""Tests for tf_agents.bandits.environments.bandit_tf_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.environments import random_bandit_environment
from tf_agents.specs import tensor_spec
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions


def get_gaussian_random_environment(
    observation_shape, action_shape, batch_size):
  """Returns a RandomBanditEnvironment with Gaussian observation and reward."""
  overall_shape = [batch_size] + observation_shape
  observation_distribution = tfd.Independent(
      tfd.Normal(loc=tf.zeros(overall_shape), scale=tf.ones(overall_shape)))
  reward_distribution = tfd.Normal(
      loc=tf.zeros(batch_size), scale=tf.ones(batch_size))
  action_spec = tensor_spec.TensorSpec(shape=action_shape, dtype=tf.float32)
  return random_bandit_environment.RandomBanditEnvironment(
      observation_distribution,
      reward_distribution,
      action_spec)


@test_util.run_all_in_graph_and_eager_modes
class RandomBanditEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(overall_observation_shape=[3, 4, 5, 6],
           batch_dims=2),
      dict(overall_observation_shape=[3, 3, 3, 3],
           batch_dims=0),
      )
  def testInvalidObservationBatchShape(
      self, overall_observation_shape, batch_dims):
    observation_distribution = tfd.Independent(
        tfd.Normal(tf.zeros(overall_observation_shape),
                   tf.ones(overall_observation_shape)),
        reinterpreted_batch_ndims=batch_dims)
    reward_distribution = tfd.Normal(tf.zeros(overall_observation_shape[0]),
                                     tf.ones(overall_observation_shape[0]))
    with self.assertRaisesRegexp(
        ValueError,
        '`observation_distribution` must have batch shape with length 1'):
      random_bandit_environment.RandomBanditEnvironment(
          observation_distribution, reward_distribution)

  @parameterized.parameters(
      dict(overall_reward_shape=[3, 4, 5, 6],
           batch_dims=2),
      dict(overall_reward_shape=[4, 5, 6],
           batch_dims=0),
      )
  def testInvalidRewardBatchShape(
      self, overall_reward_shape, batch_dims):
    observation_distribution = tfd.Normal(
        tf.zeros(overall_reward_shape[0]),
        tf.ones(overall_reward_shape[0]))
    reward_distribution = tfd.Independent(
        tfd.Normal(tf.zeros(overall_reward_shape),
                   tf.ones(overall_reward_shape)),
        reinterpreted_batch_ndims=batch_dims)
    with self.assertRaisesRegexp(
        ValueError,
        '`reward_distribution` must have batch shape with length 1'):
      random_bandit_environment.RandomBanditEnvironment(
          observation_distribution, reward_distribution)

  @parameterized.parameters(
      dict(overall_reward_shape=[3, 4, 5, 6]),
      dict(overall_reward_shape=[4, 5, 6]),
      )
  def testInvalidRewardEventShape(self, overall_reward_shape):
    observation_distribution = tfd.Normal(
        tf.zeros(overall_reward_shape[0]),
        tf.ones(overall_reward_shape[0]))
    reward_distribution = tfd.Independent(
        tfd.Normal(tf.zeros(overall_reward_shape),
                   tf.ones(overall_reward_shape)))
    with self.assertRaisesRegexp(
        ValueError, '`reward_distribution` must have event_shape ()'):
      random_bandit_environment.RandomBanditEnvironment(
          observation_distribution, reward_distribution)

  @parameterized.parameters(
      dict(overall_observation_shape=[4, 5, 6],
           overall_reward_shape=[3]),
      dict(overall_observation_shape=[3],
           overall_reward_shape=[1]),
      )
  def testMismatchedBatchShape(
      self, overall_observation_shape, overall_reward_shape):
    observation_distribution = tfd.Independent(
        tfd.Normal(tf.zeros(overall_observation_shape),
                   tf.ones(overall_observation_shape)))
    reward_distribution = tfd.Independent(
        tfd.Normal(tf.zeros(overall_reward_shape),
                   tf.ones(overall_reward_shape)))
    with self.assertRaisesRegexp(
        ValueError,
        '`reward_distribution` and `observation_distribution` must have the '
        'same batch shape'):
      random_bandit_environment.RandomBanditEnvironment(
          observation_distribution, reward_distribution)

  @parameterized.named_parameters(
      dict(testcase_name='_observation_[]_action_[]_batch_1',
           observation_shape=[],
           action_shape=[],
           batch_size=1),
      dict(testcase_name='_observation_[3, 4, 5, 6]_action_[2, 3, 4]_batch_32',
           observation_shape=[3, 4, 5, 6],
           action_shape=[2, 3, 4],
           batch_size=32),
      )
  def testObservationAndRewardShapes(
      self, observation_shape, action_shape, batch_size):
    """Exercise `reset` and `step`. Ensure correct shapes are returned."""
    env = get_gaussian_random_environment(
        observation_shape, action_shape, batch_size)
    observation = env.reset().observation
    reward = env.step(tf.zeros(batch_size)).reward

    expected_observation_shape = np.array([batch_size] + observation_shape)
    expected_reward_shape = np.array([batch_size])

    self.assertAllEqual(
        expected_observation_shape, self.evaluate(tf.shape(observation)))
    self.assertAllEqual(
        expected_reward_shape, self.evaluate(tf.shape(reward)))

  @parameterized.named_parameters(
      dict(testcase_name='_observation_[]_action_[]_batch_1',
           observation_shape=[],
           action_shape=[],
           batch_size=1,
           seed=12345),
      dict(testcase_name='_observation_[3, 4, 5, 6]_action_[2, 3, 4]_batch_32',
           observation_shape=[3, 4, 5, 6],
           action_shape=[2, 3, 4],
           batch_size=32,
           seed=98765),
      )
  def testObservationAndRewardsVary(
      self, observation_shape, action_shape, batch_size, seed):
    """Ensure that observations and rewards change in consecutive calls."""
    tf.compat.v1.set_random_seed(seed)
    env = get_gaussian_random_environment(
        observation_shape, action_shape, batch_size)

    observation0 = env.reset().observation
    reward0 = env.step(tf.zeros([batch_size] + action_shape)).reward
    observation0 = self.evaluate(observation0)
    reward0 = self.evaluate(reward0)

    observation1 = env.reset().observation
    reward1 = env.step(tf.zeros([batch_size] + action_shape)).reward
    self.evaluate(observation1)
    self.evaluate(reward1)

    self.assertNotAllClose(observation0, observation1)
    self.assertNotAllClose(reward0, reward1)


if __name__ == '__main__':
  tf.test.main()
