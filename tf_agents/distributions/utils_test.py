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

"""Tests for tf_agents.distributions.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.distributions import utils
from tf_agents.specs import tensor_spec


class UtilsTest(tf.test.TestCase):

  def testScaleDistribution(self):
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -2, 4)
    distribution = tfp.distributions.Normal(0, 4)
    scaled_distribution = utils.scale_distribution_to_spec(distribution,
                                                           action_spec)
    if tf.executing_eagerly():
      sample = scaled_distribution.sample
    else:
      sample = scaled_distribution.sample()

    for _ in range(1000):
      sample_np = self.evaluate(sample)

      self.assertGreater(sample_np, -2.00001)
      self.assertLess(sample_np, 4.00001)

  def testSquashToSpecNormalModeMethod(self):
    input_dist = tfp.distributions.Normal(loc=1.0, scale=3.0)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -2.0, 4.0)
    squash_to_spec_normal = utils.SquashToSpecNormal(input_dist, action_spec)
    self.assertAlmostEqual(
        self.evaluate(squash_to_spec_normal.mode()), 3.28478247, places=5)

  def testSquashToSpecNormalStdMethod(self):
    input_dist = tfp.distributions.Normal(loc=1.0, scale=3.0)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -2.0, 4.0)
    squash_to_spec_normal = utils.SquashToSpecNormal(input_dist, action_spec)
    self.assertAlmostEqual(
        self.evaluate(squash_to_spec_normal.stddev()), 2.98516426, places=5)


if __name__ == '__main__':
  tf.test.main()
