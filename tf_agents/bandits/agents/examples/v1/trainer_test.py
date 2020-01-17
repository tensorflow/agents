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

"""Tests for tf_agents.bandits.agents.examples.v1.trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import unittest

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import exp3_agent
from tf_agents.bandits.agents.examples.v1 import trainer
from tf_agents.bandits.environments import random_bandit_environment
from tf_agents.specs import tensor_spec
from tf_agents.utils import test_utils

from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions


def get_bounded_reward_random_environment(
    observation_shape, action_shape, batch_size, num_actions):
  """Returns a RandomBanditEnvironment with U(0, 1) observation and reward."""
  overall_shape = [batch_size] + observation_shape
  observation_distribution = tfd.Independent(
      tfd.Uniform(low=tf.zeros(overall_shape), high=tf.ones(overall_shape)))
  reward_distribution = tfd.Uniform(
      low=tf.zeros(batch_size), high=tf.ones(batch_size))
  action_spec = tensor_spec.BoundedTensorSpec(
      shape=action_shape, dtype=tf.int32, minimum=0, maximum=num_actions - 1)
  return random_bandit_environment.RandomBanditEnvironment(
      observation_distribution,
      reward_distribution,
      action_spec)


class TrainerTF1Test(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_0',
           num_actions=11,
           observation_shape=[8],
           action_shape=[],
           batch_size=32,
           training_loops=10,
           steps_per_loop=10,
           learning_rate=.1)
      )
  @unittest.skipIf(tf2.enabled(), 'TF 1.x only test.')
  def testTrainerTF1ExportsCheckpoints(self,
                                       num_actions,
                                       observation_shape,
                                       action_shape,
                                       batch_size,
                                       training_loops,
                                       steps_per_loop,
                                       learning_rate):
    """Tests TF1 trainer code, checks that expected checkpoints are exported."""
    root_dir = tempfile.mkdtemp(dir=os.getenv('TEST_TMPDIR'))
    environment = get_bounded_reward_random_environment(
        observation_shape, action_shape, batch_size, num_actions)
    agent = exp3_agent.Exp3Agent(learning_rate=learning_rate,
                                 time_step_spec=environment.time_step_spec(),
                                 action_spec=environment.action_spec())

    trainer.train(root_dir, agent, environment, training_loops, steps_per_loop)
    latest_checkpoint = tf.train.latest_checkpoint(
        os.path.join(root_dir, 'train'))
    expected_checkpoint_regex = '.*ckpt-{}'.format(
        training_loops * batch_size * steps_per_loop)
    self.assertRegex(latest_checkpoint, expected_checkpoint_regex)


if __name__ == '__main__':
  tf.test.main()
