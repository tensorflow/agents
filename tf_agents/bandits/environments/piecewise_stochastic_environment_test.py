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

"""Tests for tf_agents.bandits.environments.bandit_tf_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.environments import piecewise_stochastic_environment as pse


tfd = tfp.distributions


def get_deterministic_gaussian_non_stationary_environment(
    observation_shape, action_shape, batch_size, interval):
  """Returns a PiecewiseStochasticEnvironment with deterministic intervals."""
  overall_shape = [batch_size] + observation_shape
  observation_distribution = tfd.Normal(
      loc=tf.zeros(overall_shape), scale=tf.ones(overall_shape))
  interval_distribution = tfd.Deterministic(interval)
  observation_to_reward_shape = observation_shape + action_shape
  observation_to_reward_distribution = tfd.Normal(
      loc=tf.zeros(observation_to_reward_shape),
      scale=tf.ones(observation_to_reward_shape))
  additive_reward_distribution = tfd.Normal(
      loc=tf.zeros(action_shape),
      scale=tf.ones(action_shape))
  return pse.PiecewiseStochasticEnvironment(
      observation_distribution,
      interval_distribution,
      observation_to_reward_distribution,
      additive_reward_distribution)


class PiecewiseStochasticEnvironmentTest(tf.test.TestCase,
                                         parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_observation_[5]_action_[3]_batch_1',
           observation_shape=[5],
           action_shape=[3],
           batch_size=1,
           seed=12345),
      dict(testcase_name='_observation_[3]_action_[5]_batch_2',
           observation_shape=[3],
           action_shape=[5],
           batch_size=2,
           seed=98765),
  )
  def testObservationAndRewardsVary(
      self, observation_shape, action_shape, batch_size, seed):
    """Ensure that observations and rewards change in consecutive calls."""
    interval = 4

    env = get_deterministic_gaussian_non_stationary_environment(
        observation_shape, action_shape, batch_size, interval)
    tf.compat.v1.set_random_seed(seed)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    observation_to_reward_samples = []
    additive_reward_samples = []
    env_time = env._env_time
    observation_to_reward = (
        env._environment_dynamics._current_observation_to_reward)
    additive_reward = env._environment_dynamics._current_additive_reward

    if tf.executing_eagerly():
      for t in range(0, 10):
        ts = env.reset()
        reward = env.step(tf.zeros([batch_size])).reward

        (observation_to_reward_sample,
         additive_reward_sample,
         env_time_sample) = self.evaluate([observation_to_reward,
                                           additive_reward,
                                           env_time])
        observation_to_reward_samples.append(observation_to_reward_sample)
        additive_reward_samples.append(additive_reward_sample)
        self.assertEqual(env_time_sample, (t + 1) * batch_size)

    else:
      ts = env.reset()
      reward = env.step(tf.zeros([batch_size])).reward

      for t in range(0, 10):
        unused_ts_sample = self.evaluate(ts)
        unused_reward_sample = self.evaluate(reward)

        (observation_to_reward_sample,
         additive_reward_sample,
         env_time_sample) = self.evaluate([observation_to_reward,
                                           additive_reward,
                                           env_time])

        observation_to_reward_samples.append(observation_to_reward_sample)
        additive_reward_samples.append(additive_reward_sample)
        self.assertEqual(env_time_sample, (t + 1) * batch_size)

    for t in range(0, 10):
      self.assertAllClose(
          observation_to_reward_samples[int(t - t % (interval / batch_size))],
          observation_to_reward_samples[t])
      self.assertAllClose(
          additive_reward_samples[int(t - t % (interval / batch_size))],
          additive_reward_samples[t])

    for t in range(interval // batch_size, 10, interval // batch_size):
      self.assertNotAllClose(
          observation_to_reward_samples[int(t) - 1],
          observation_to_reward_samples[int(t)])
      self.assertNotAllClose(
          additive_reward_samples[int(t) - 1],
          additive_reward_samples[int(t)])

  @parameterized.named_parameters(
      dict(testcase_name='_observation_[5]_action_[3]_batch_1',
           observation_shape=[5],
           action_shape=[3],
           batch_size=1),
      dict(testcase_name='_observation_[7]_action_[5]_batch_32',
           observation_shape=[7],
           action_shape=[5],
           batch_size=32),
      )
  def testActionSpec(
      self, observation_shape, action_shape, batch_size):
    """Ensure that the action spec is set correctly."""
    interval = 3
    env = get_deterministic_gaussian_non_stationary_environment(
        observation_shape, action_shape, batch_size, interval)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(env.action_spec().maximum, action_shape[0] - 1)


if __name__ == '__main__':
  tf.test.main()
