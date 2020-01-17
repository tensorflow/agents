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

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.environments import non_stationary_stochastic_environment as nsse
from tf_agents.specs import tensor_spec
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions


class DummyDynamics(nsse.EnvironmentDynamics):

  @property
  def batch_size(self):
    return 2

  @property
  def observation_spec(self):
    return tensor_spec.TensorSpec(
        shape=[3],
        dtype=tf.float32,
        name='observation_spec')

  @property
  def action_spec(self):
    return tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=5,
        name='action')

  def observation(self, t):
    return (tf.constant([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]], dtype=tf.float32) +
            tf.reshape(tf.cast(t, dtype=tf.float32), [1, 1]))

  def reward(self, observation, t):
    return (tf.concat([observation, tf.zeros([2, 2])], axis=1) -
            tf.reshape(tf.cast(t, dtype=tf.float32), [1, 1]))


@test_util.run_all_in_graph_and_eager_modes
class NonStationaryStochasticEnvironmentTest(tf.test.TestCase):

  def testObservationAndRewardsVary(self):
    """Ensure that observations and rewards change in consecutive calls."""
    dynamics = DummyDynamics()
    env = nsse.NonStationaryStochasticEnvironment(dynamics)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    env_time = env._env_time
    observation_samples = []
    reward_samples = []

    if tf.executing_eagerly():
      for t in range(0, 10):
        ts = env.reset()
        observation = ts.observation
        reward = env.step(tf.zeros([2])).reward

        [observation_sample, reward_sample, env_time_sample] = self.evaluate(
            [observation, reward, env_time])
        observation_samples.append(observation_sample)
        reward_samples.append(reward_sample)
        self.assertEqual(env_time_sample, (t + 1) * dynamics.batch_size)

    else:
      ts = env.reset()
      observation = ts.observation
      reward = env.step(tf.zeros([2])).reward

      for t in range(0, 10):
        # The order of evaluations below matters. We first compute observation
        # batch, then the reward, and finally the env_time tensor. Joining the
        # evaluations in a single call does not guarantee the right order.

        observation_sample = self.evaluate(observation)
        reward_sample = self.evaluate(reward)
        env_time_sample = self.evaluate(env_time)
        observation_samples.append(observation_sample)
        reward_samples.append(reward_sample)
        self.assertEqual(env_time_sample, (t + 1) * dynamics.batch_size)

    for t in range(0, 10):
      t_b = t * dynamics.batch_size
      self.assertAllClose(observation_samples[t],
                          [[1.0 + t_b, 2.0 + t_b, 3.0 + t_b],
                           [0.0 + t_b, 4.0 + t_b, 5.0 + t_b]])
      self.assertAllClose(reward_samples[t], [1, 0])

if __name__ == '__main__':
  tf.test.main()
