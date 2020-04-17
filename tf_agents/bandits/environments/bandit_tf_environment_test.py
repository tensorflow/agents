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
from tf_agents.bandits.environments import bandit_tf_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


class ZerosEnvironment(bandit_tf_environment.BanditTFEnvironment):
  """A simple environment that returns zeros for observations and rewards."""

  def __init__(self, observation_shape, batch_size=1):
    observation_spec = tensor_spec.TensorSpec(shape=observation_shape,
                                              dtype=tf.float32,
                                              name='observation')
    time_step_spec = ts.time_step_spec(observation_spec)
    super(ZerosEnvironment, self).__init__(time_step_spec=time_step_spec,
                                           batch_size=batch_size)

  def _apply_action(self, action):
    return tf.zeros(self.batch_size)

  def _observe(self):
    observation_shape = [self.batch_size] + list(self.observation_spec().shape)
    return tf.zeros(observation_shape)


class MultipleRewardsEnvironment(bandit_tf_environment.BanditTFEnvironment):
  """A simple multiple-rewards environment."""

  def __init__(self, observation_shape, batch_size=1, num_rewards=1):
    self._num_rewards = num_rewards
    reward_spec = tensor_spec.TensorSpec(shape=[self._num_rewards],
                                         dtype=tf.float32,
                                         name='reward')

    observation_spec = tensor_spec.TensorSpec(shape=observation_shape,
                                              dtype=tf.float32,
                                              name='observation')
    time_step_spec = ts.time_step_spec(observation_spec, reward_spec)
    super(MultipleRewardsEnvironment, self).__init__(
        time_step_spec=time_step_spec,
        batch_size=batch_size)

  def _apply_action(self, action):
    return tf.zeros([self.batch_size, self._num_rewards])

  def _observe(self):
    observation_shape = [self.batch_size] + list(self.observation_spec().shape)
    return tf.zeros(observation_shape)


@test_util.run_all_in_graph_and_eager_modes
class BanditTFEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_observation_[]_batch_1',
           observation_shape=[3],
           batch_size=1),
      dict(testcase_name='_observation_[7]_batch_32',
           observation_shape=[7],
           batch_size=32),
      dict(testcase_name='_observation_[3, 4, 5]_batch_11',
           observation_shape=[3, 4, 5],
           batch_size=11)
      )
  def testObservationAndRewardShapes(self, batch_size, observation_shape):
    """Exercise `reset` and `step`. Ensure correct shapes are returned."""
    env = ZerosEnvironment(batch_size=batch_size,
                           observation_shape=observation_shape)

    @common.function
    def observation_and_reward():
      observation = env.reset().observation
      reward = env.step(tf.zeros(batch_size)).reward
      return observation, reward

    observation, reward = observation_and_reward()

    expected_observation = np.zeros([batch_size] + observation_shape)
    expected_reward = np.zeros(batch_size)

    np.testing.assert_array_almost_equal(
        expected_observation, self.evaluate(observation))
    np.testing.assert_array_almost_equal(
        expected_reward, self.evaluate(reward))

  @parameterized.named_parameters(
      dict(testcase_name='',
           observation_shape=[32],
           batch_size=12),
      )
  def testTwoConsecutiveSteps(self, batch_size, observation_shape):
    """Test two consecutive calls to `step`."""
    env = ZerosEnvironment(batch_size=batch_size,
                           observation_shape=observation_shape)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(env.reset())
    self.evaluate(env.step(tf.zeros(batch_size)))
    self.evaluate(env.step(tf.zeros(batch_size)))

  @parameterized.named_parameters(
      dict(testcase_name='_observation_[3]_batch5_2rewards',
           observation_shape=[3],
           batch_size=5,
           num_rewards=2),
      dict(testcase_name='_observation_[7]_batch8_4rewards',
           observation_shape=[7],
           batch_size=8,
           num_rewards=4),
      )
  def testMultipleRewardsEnvironment(
      self, batch_size, observation_shape, num_rewards):
    """Test the multiple-rewards case. Ensure correct shapes are returned."""

    env = MultipleRewardsEnvironment(
        observation_shape=observation_shape,
        batch_size=batch_size,
        num_rewards=num_rewards)

    observation = env.reset().observation
    reward = env.step(tf.zeros(batch_size)).reward

    expected_observation = np.zeros([batch_size] + observation_shape)
    expected_reward = np.zeros([batch_size, num_rewards])

    np.testing.assert_array_almost_equal(
        expected_observation, self.evaluate(observation))
    np.testing.assert_array_almost_equal(
        expected_reward, self.evaluate(reward))

    self.assertEqual(env.reward_spec().shape, num_rewards)

if __name__ == '__main__':
  tf.test.main()
