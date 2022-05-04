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

r"""Tests for tf_agents.policies.policy_info_updater_wrapper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tf_agents.policies import policy_info_updater_wrapper
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DistributionPolicy(tf_policy.TFPolicy):
  """A policy which always returns the configured distribution."""

  def __init__(self,
               distribution,
               time_step_spec,
               action_spec,
               info_spec,
               name=None):
    self._distribution_value = distribution
    super(DistributionPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        name=name)

  def _action(self, time_step, policy_state, seed):
    return policy_step.PolicyStep(tf.constant(1., shape=(1,)), policy_state,
                                  {'test_info': tf.constant(2, shape=(1,))})

  def _distribution(self, time_step, policy_state):
    return policy_step.PolicyStep(self._distribution_value, policy_state,
                                  {'test_info': tf.constant(2, shape=(1,))})

  def _variables(self):
    return []


class ModelIdUpdater(object):

  def __call__(self, step):
    del step
    return {'model_id': tf.expand_dims(2, axis=0)}


class PolicyInfoUpdaterWrapperTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PolicyInfoUpdaterWrapperTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)

  def test_model_id_updater(self):
    loc = 0.0
    scale = 0.5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, tf.float32.min,
                                                tf.float32.max)
    wrapped_policy = DistributionPolicy(
        distribution=tfp.distributions.Normal([loc], [scale]),
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        info_spec={
            'test_info':
                tf.TensorSpec(shape=(1,), dtype=tf.int32, name='test_info')
        })
    updater_info_spec = {
        'model_id': tf.TensorSpec(shape=(1,), dtype=tf.int32, name='model_id')
    }
    updater_info_spec.update(wrapped_policy.info_spec)
    policy = policy_info_updater_wrapper.PolicyInfoUpdaterWrapper(
        policy=wrapped_policy,
        info_spec=updater_info_spec,
        updater_fn=ModelIdUpdater(),
        name='model_id_updater')

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    distribution_step = policy.distribution(time_step)

    tf.nest.assert_same_structure(action_spec, action_step.action)
    tf.nest.assert_same_structure(action_spec, distribution_step.action)

    self.assertListEqual(list(self.evaluate(action_step.info['model_id'])), [2])
    self.assertListEqual(
        list(self.evaluate(distribution_step.info['model_id'])), [2])


if __name__ == '__main__':
  tf.test.main()
