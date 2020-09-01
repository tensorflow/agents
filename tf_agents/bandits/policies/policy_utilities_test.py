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

"""Tests for tf_agents.bandits.policies.policy_utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.policies import policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.utils import test_utils


_GREEDY = policy_utilities.BanditPolicyType.GREEDY
_UNIFORM = policy_utilities.BanditPolicyType.UNIFORM


class PolicyUtilitiesTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(
          input_tensor=[[4, 8, 2, -3], [0, 5, -234, 64]],
          mask=[[1, 0, 0, 1], [0, 1, 1, 1]],
          expected=[0, 3]),
      dict(
          input_tensor=[[3, 0.2, -3.3], [987, -2.5, 64], [0, 0, 0], [4, 3, 8]],
          mask=[[1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
          expected=[0, 0, 0, 2]),
      dict(input_tensor=[[1, 2]], mask=[[1, 0]], expected=[0]))
  def testMaskedArgmax(self, input_tensor, mask, expected):
    actual = policy_utilities.masked_argmax(
        tf.constant(input_tensor, dtype=tf.float32), tf.constant(mask))
    self.assertAllEqual(actual, expected)

  def testBadMask(self):
    input_tensor = tf.reshape(tf.range(12, dtype=tf.float32), shape=[3, 4])
    mask = [[1, 0, 0, 1], [0, 0, 0, 0], [1, 0, 1, 1]]
    expected = [3, -1, 3]
    actual = self.evaluate(
        policy_utilities.masked_argmax(input_tensor, tf.constant(mask)))
    self.assertAllEqual(actual, expected)

  def testNumActionsFromTensorSpecGoodSpec(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=15)
    num_actions = policy_utilities.get_num_actions_from_tensor_spec(action_spec)
    self.assertEqual(num_actions, 16)

  def testNumActionsFromTensorSpecWrongRank(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(2, 3), minimum=0, maximum=15)

    with self.assertRaisesRegex(ValueError, r'Action spec must be a scalar'):
      policy_utilities.get_num_actions_from_tensor_spec(action_spec)

  def testSetBanditPolicyType(self):
    dims = (10, 1)
    bandit_policy_spec = (
        policy_utilities.create_bandit_policy_type_tensor_spec(dims))
    info = policy_utilities.set_bandit_policy_type(None, bandit_policy_spec)
    self.assertIsInstance(info, policy_utilities.PolicyInfo)
    self.assertIsInstance(info.bandit_policy_type,
                          tensor_spec.BoundedTensorSpec)
    self.assertEqual(info.bandit_policy_type.shape, dims)
    self.assertEqual(info.bandit_policy_type.dtype, tf.int32)
    # Set to tensor.
    input_tensor = tf.fill(dims, value=_GREEDY)
    info = policy_utilities.set_bandit_policy_type(info, input_tensor)
    self.assertIsInstance(info.bandit_policy_type, tf.Tensor)
    self.assertEqual(info.bandit_policy_type.shape, input_tensor.shape)
    expected = [[_GREEDY] for _ in range(dims[0])]
    self.assertAllEqual(info.bandit_policy_type, expected)

  def testWrongPolicyInfoType(self):
    dims = (10, 1)
    log_probability = tf.fill(dims, value=-0.5)
    info = policy_step.PolicyInfo(log_probability=log_probability)
    input_tensor = tf.fill(dims, value=_GREEDY)
    result = policy_utilities.set_bandit_policy_type(info, input_tensor)
    self.assertNotIsInstance(result, policy_utilities.PolicyInfo)
    self.assertAllEqual(info.log_probability, result.log_probability)

  def testBanditPolicyUniformMask(self):
    dims = (10, 1)
    input_tensor = tf.fill(dims, value=_GREEDY)
    # Overwrite some values with UNIFORM.
    mask_idx = (range(dims[0])[1:dims[0]:2])
    mask = [[True if idx in mask_idx else False] for idx in range(dims[0])]
    expected = [[_UNIFORM if mask_value[0]  else _GREEDY]
                for mask_value in mask]
    result = policy_utilities.bandit_policy_uniform_mask(input_tensor, mask)
    self.assertAllEqual(result, expected)


if __name__ == '__main__':
  tf.test.main()
