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

"""Tests for tf_agents.bandits.environments.bernoulli_action_mask_tf_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.environments import bernoulli_action_mask_tf_environment as masked_tf_env
from tf_agents.bandits.environments import random_bandit_environment
from tf_agents.specs import tensor_spec

tfd = tfp.distributions


class BernoulliActionMaskTfEnvironmentTest(tf.test.TestCase,
                                           parameterized.TestCase):

  @parameterized.parameters([(7, 4), (8, 5)])
  def testMaybeAddOneAction(self, batch_size, num_actions):
    original_mask = tf.eye(batch_size, num_actions, dtype=tf.int32)
    new_mask = self.evaluate(masked_tf_env._maybe_add_one_action(original_mask))
    self.assertAllEqual(original_mask[:num_actions, :],
                        new_mask[:num_actions, :])
    ones = tf.ones([batch_size], dtype=tf.int32)
    self.assertAllEqual(tf.reduce_max(new_mask, axis=1), ones)
    self.assertAllEqual(tf.reduce_sum(new_mask, axis=1), ones)

  @parameterized.parameters([(7, 4), (10, 3)])
  def testMaskedEnvironment(self, batch_size, num_actions):
    observation_distribution = tfd.Independent(
        tfd.Normal(tf.zeros([batch_size, 2]), tf.ones([batch_size, 2])))
    reward_distribution = tfd.Normal(tf.zeros(batch_size), tf.ones(batch_size))
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), minimum=0, maximum=num_actions - 1, dtype=tf.int32)

    env = random_bandit_environment.RandomBanditEnvironment(
        observation_distribution, reward_distribution, action_spec)
    masked_env = masked_tf_env.BernoulliActionMaskTFEnvironment(
        env, lambda x, y: (x, y), 0.5)
    context, mask = self.evaluate(masked_env.reset().observation)
    self.assertAllEqual(tf.shape(context), [batch_size, 2])
    self.assertAllEqual(tf.shape(mask), [batch_size, num_actions])
    surely_allowed_actions = tf.argmax(mask, axis=-1, output_type=tf.int32)
    rewards = self.evaluate(masked_env.step(surely_allowed_actions).reward)
    self.assertAllEqual(tf.shape(rewards), [batch_size])

  @parameterized.parameters([(7, 4), (10, 3)])
  def testDisallowedAction(self, batch_size, num_actions):
    observation_distribution = tfd.Independent(
        tfd.Normal(tf.zeros([batch_size, 2]), tf.ones([batch_size, 2])))
    reward_distribution = tfd.Normal(tf.zeros(batch_size), tf.ones(batch_size))
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), minimum=0, maximum=num_actions - 1, dtype=tf.int32)

    env = random_bandit_environment.RandomBanditEnvironment(
        observation_distribution, reward_distribution, action_spec)
    masked_env = masked_tf_env.BernoulliActionMaskTFEnvironment(
        env, lambda x, y: (x, y), 0.0)
    _, mask = self.evaluate(masked_env.reset().observation)
    surely_disallowed_actions = tf.argmin(mask, axis=-1, output_type=tf.int32)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'not in allowed'):
      self.evaluate(masked_env.step(surely_disallowed_actions).reward)


if __name__ == '__main__':
  tf.test.main()
