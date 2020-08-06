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

"""Tests for TF Agents ppo_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.trajectories import time_step as ts


class PPOUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('OnNotAllowPartialReturnsZerosOnIncompleteSteps', False),
      ('OnAllowPartialReturnsOnesOnIncompleteStepsAndZerosBetween', True))
  def testMakeTimestepMaskWithPartialEpisode(self, allow_partial):
    first, mid, last = ts.StepType.FIRST, ts.StepType.MID, ts.StepType.LAST

    next_step_types = tf.constant([[mid, mid, last, first,
                                    mid, mid, last, first,
                                    mid, mid],
                                   [mid, mid, last, first,
                                    mid, mid, mid, mid,
                                    mid, last]])
    zeros = tf.zeros_like(next_step_types)
    next_time_step = ts.TimeStep(next_step_types, zeros, zeros, zeros)

    if not allow_partial:
      # Mask should be 0.0 for transition timesteps (3, 7) and for all timesteps
      #   belonging to the final, incomplete episode.
      expected_mask = [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    else:
      # Zeros only between episodes. Incomplete episodes are valid and not
      # zeroed out.
      expected_mask = [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    timestep_mask = ppo_utils.make_timestep_mask(
        next_time_step, allow_partial_episodes=allow_partial)

    timestep_mask_ = self.evaluate(timestep_mask)
    self.assertAllClose(expected_mask, timestep_mask_)

  def test_nested_kl_divergence(self):
    zero = tf.constant([0.0] * 3, dtype=tf.float32)
    one = tf.constant([1.0] * 3, dtype=tf.float32)
    dist_neg_one = tfp.distributions.Normal(loc=-one, scale=one)
    dist_zero = tfp.distributions.Normal(loc=zero, scale=one)
    dist_one = tfp.distributions.Normal(loc=one, scale=one)

    nested_dist1 = [dist_zero, [dist_neg_one, dist_one]]
    nested_dist2 = [dist_one, [dist_one, dist_zero]]
    kl_divergence = ppo_utils.nested_kl_divergence(
        nested_dist1, nested_dist2)
    expected_kl_divergence = 3 * 3.0  # 3 * (0.5 + (2.0 + 0.5))

    kl_divergence_ = self.evaluate(kl_divergence)
    self.assertAllClose(expected_kl_divergence, kl_divergence_)

  def test_get_distribution_params(self):
    ones = tf.ones(shape=[2], dtype=tf.float32)
    distribution = (tfp.distributions.Categorical(logits=ones),
                    tfp.distributions.Normal(ones, ones))
    params = ppo_utils.get_distribution_params(distribution)
    self.assertAllEqual([set(['logits']), set(['loc', 'scale'])],
                        [set(d.keys()) for d in params])  # pytype: disable=attribute-error
    self.assertAllEqual([[[2]], [[2], [2]]],
                        [[d[k].shape.as_list() for k in d] for d in params])  # pytype: disable=attribute-error

  def test_get_learning_rate(self):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
    learning_rate = ppo_utils.get_learning_rate(optimizer)
    expected_learning_rate = 0.1
    self.assertAlmostEqual(expected_learning_rate, learning_rate)

  def test_get_learning_rate_with_fn(self):
    learning_rate_var = tf.Variable(0.1, dtype=tf.float64)
    def learning_rate_fn():
      return learning_rate_var

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_fn)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    learning_rate = ppo_utils.get_learning_rate(optimizer)
    expected_learning_rate = 0.1
    self.assertAlmostEqual(expected_learning_rate, self.evaluate(learning_rate))

if __name__ == '__main__':
  tf.test.main()
