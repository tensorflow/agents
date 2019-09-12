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

"""Tests for tf_agents.bandits.agents.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.bandits.agents import utils
from tf_agents.specs import tensor_spec

tfd = tfp.distributions
tf.compat.v1.enable_v2_behavior()


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10',
          'batch_size': 1,
          'context_dim': 10,
      }, {
          'testcase_name': '_batch4_contextdim5',
          'batch_size': 4,
          'context_dim': 5,
      })


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def testNumActionsFromTensorSpecGoodSpec(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=15)
    num_actions = utils.get_num_actions_from_tensor_spec(action_spec)
    self.assertEqual(num_actions, 16)

  def testNumActionsFromTensorSpecWrongRank(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(2, 3), minimum=0, maximum=15)

    with self.assertRaisesRegexp(ValueError, r'Action spec must be a scalar'):
      utils.get_num_actions_from_tensor_spec(action_spec)

  @test_cases()
  def testBUpdate(self, batch_size, context_dim):
    b_array = np.array(range(context_dim))
    r_array = np.array(range(batch_size)).reshape((batch_size, 1))
    x_array = np.array(range(batch_size * context_dim)).reshape(
        (batch_size, context_dim))
    rx = r_array * x_array
    expected_b_updated_array = b_array + np.sum(rx, axis=0)

    b = tf.constant(b_array, dtype=tf.float32, shape=[context_dim])
    r = tf.constant(r_array, dtype=tf.float32, shape=[batch_size])
    x = tf.constant(x_array, dtype=tf.float32, shape=[batch_size, context_dim])
    b_update = utils.sum_reward_weighted_observations(r, x)
    self.assertAllClose(expected_b_updated_array, self.evaluate(b + b_update))

  @test_cases()
  def testBUpdateEmptyObservations(self, batch_size, context_dim):
    r = tf.constant([], dtype=tf.float32, shape=[0, 1])
    x = tf.constant([], dtype=tf.float32, shape=[0, context_dim])
    b_update = utils.sum_reward_weighted_observations(r, x)
    expected_b_update_array = np.zeros([context_dim], dtype=np.float32)
    self.assertAllClose(expected_b_update_array, self.evaluate(b_update))


if __name__ == '__main__':
  tf.test.main()
