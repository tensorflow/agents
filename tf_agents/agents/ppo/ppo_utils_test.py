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
import mock
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.environments import time_step as ts
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_step
from tf_agents.specs import tensor_spec

nest = tf.contrib.framework.nest


class PPOUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def testMakeTimestepMaskWithPartialEpisode(self):
    first, mid, last = ts.StepType.FIRST, ts.StepType.MID, ts.StepType.LAST

    next_step_types = tf.constant([[mid, mid, last, first,
                                    mid, mid, last, first,
                                    mid, mid],
                                   [mid, mid, last, first,
                                    mid, mid, mid, mid,
                                    mid, last]])
    zeros = tf.zeros_like(next_step_types)
    next_time_step = ts.TimeStep(next_step_types, zeros, zeros, zeros)

    # Mask should be 0.0 for transition timesteps (3, 7) and for all timesteps
    #   belonging to the final, incomplete episode.
    expected_mask = [[1.0, 1.0, 1.0, 0.0,
                      1.0, 1.0, 1.0, 0.0,
                      0.0, 0.0],
                     [1.0, 1.0, 1.0, 0.0,
                      1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0]]
    timestep_mask = ppo_utils.make_timestep_mask(next_time_step)

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

  def test_get_distribution_class_spec(self):
    ones = tf.ones(shape=[2], dtype=tf.float32)
    obs_spec = tensor_spec.TensorSpec(shape=[5], dtype=tf.float32)
    time_step_spec = ts.time_step_spec(obs_spec)
    mock_policy = mock.create_autospec(actor_policy.ActorPolicy)
    mock_policy.distribution.return_value = policy_step.PolicyStep(
        (tfp.distributions.Categorical(logits=ones),
         tfp.distributions.Normal(ones, ones)), None)

    class_spec = ppo_utils.get_distribution_class_spec(mock_policy,
                                                       time_step_spec)
    self.assertAllEqual(
        (tfp.distributions.Categorical, tfp.distributions.Normal), class_spec)

  def test_get_distribution_params_spec(self):
    ones = tf.ones(shape=[1, 2], dtype=tf.float32)
    obs_spec = tensor_spec.TensorSpec(shape=[5], dtype=tf.float32)
    time_step_spec = ts.time_step_spec(obs_spec)
    mock_policy = mock.create_autospec(actor_policy.ActorPolicy)
    mock_policy._distribution.return_value = policy_step.PolicyStep(
        (tfp.distributions.Categorical(logits=ones),
         tfp.distributions.Normal(ones, ones)))

    params_spec = ppo_utils.get_distribution_params_spec(mock_policy,
                                                         time_step_spec)
    self.assertAllEqual([set(['logits']), set(['loc', 'scale'])],
                        [set(d.keys()) for d in params_spec])
    self.assertAllEqual([[[2]], [[2], [2]]],
                        [[d[k].shape for k in d] for d in params_spec])

  def test_get_distribution_params(self):
    ones = tf.ones(shape=[2], dtype=tf.float32)
    distribution = (tfp.distributions.Categorical(logits=ones),
                    tfp.distributions.Normal(ones, ones))
    params = ppo_utils.get_distribution_params(distribution)
    self.assertAllEqual([set(['logits']), set(['loc', 'scale'])],
                        [set(d.keys()) for d in params])
    self.assertAllEqual([[[2]], [[2], [2]]],
                        [[d[k].shape.as_list() for k in d] for d in params])

  def test_get_distribution_from_params_and_classes(self):
    distribution_params = ({'logits': tf.constant([1, 1], dtype=tf.float32)},
                           {'loc': tf.constant([1, 1], dtype=tf.float32),
                            'scale': tf.constant([1, 1], dtype=tf.float32)})
    get_distribution_class_spec = (tfp.distributions.Categorical,
                                   tfp.distributions.Normal)
    nested_distribution = ppo_utils.get_distribution_from_params_and_classes(
        distribution_params, get_distribution_class_spec)
    self.assertAllEqual(
        [tfp.distributions.Categorical, tfp.distributions.Normal],
        [d.__class__ for d in nested_distribution])
    self.assertAllEqual([2], nested_distribution[0].logits.shape.as_list())
    self.assertAllEqual([2], nested_distribution[1].sample().shape.as_list())

if __name__ == '__main__':
  tf.test.main()
