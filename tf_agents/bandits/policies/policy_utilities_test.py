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
import tensorflow as tf

from tf_agents.bandits.policies import policy_utilities
from tf_agents.utils import test_utils
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


@test_util.run_all_in_graph_and_eager_modes
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
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          policy_utilities.masked_argmax(input_tensor, tf.constant(mask)))


if __name__ == '__main__':
  tf.test.main()
