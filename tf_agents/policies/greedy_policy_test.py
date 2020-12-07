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

"""Test for tf_agents.policies.greedy_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DistributionPolicy(tf_policy.TFPolicy):
  """A policy which always returns the configured distribution."""

  def __init__(self, distribution, time_step_spec, action_spec, name=None):
    self._distribution_value = distribution
    super(DistributionPolicy, self).__init__(
        time_step_spec, action_spec, name=name)

  def _action(self, time_step, policy_state, seed):
    raise NotImplementedError('Not implemented.')

  def _distribution(self, time_step, policy_state):
    return policy_step.PolicyStep(self._distribution_value, policy_state)

  def _variables(self):
    return []


class GreedyPolicyTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(GreedyPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)

  @parameterized.parameters(
      {'action_probs': [0.5, 0.2, 0.3]},
      {'action_probs': [0.1, 0.1, 0.6, 0.2]}
  )
  def testCategoricalActions(self, action_probs):
    action_spec = [
        tensor_spec.BoundedTensorSpec((1,), tf.int32, 0, len(action_probs)-1),
        tensor_spec.BoundedTensorSpec((), tf.int32, 0, len(action_probs)-1)]
    wrapped_policy = DistributionPolicy([
        tfp.distributions.Categorical(probs=[action_probs]),
        tfp.distributions.Categorical(probs=action_probs)
    ], self._time_step_spec, action_spec)
    policy = greedy_policy.GreedyPolicy(wrapped_policy)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, action_spec)

    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=1)
    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    action_ = self.evaluate(action_step.action)
    self.assertEqual(action_[0][0], np.argmax(action_probs))
    self.assertEqual(action_[1], np.argmax(action_probs))

  @parameterized.parameters(
      {'loc': 1.0, 'scale': 0.2},
      {'loc': -2.0, 'scale': 1.0},
      {'loc': 0.0, 'scale': 0.5}
  )
  def testNormalActions(self, loc, scale):
    action_spec = tensor_spec.BoundedTensorSpec(
        [1], tf.float32, tf.float32.min, tf.float32.max)
    wrapped_policy = DistributionPolicy(
        tfp.distributions.Normal([loc], [scale]), self._time_step_spec,
        action_spec)
    policy = greedy_policy.GreedyPolicy(wrapped_policy)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    action_ = self.evaluate(action_step.action)
    self.assertAlmostEqual(action_[0], loc)


if __name__ == '__main__':
  tf.test.main()
