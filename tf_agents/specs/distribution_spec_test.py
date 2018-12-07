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

"""Tests for tf_agents.specs.distribution_spec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec

tfd = tfp.distributions


class DistributionSpecTest(tf.test.TestCase):

  def testBuildsDistribution(self):
    expected_distribution = tfd.Categorical([0.2, 0.3, 0.5], validate_args=True)
    input_param_spec = tensor_spec.TensorSpec((3,), dtype=tf.float32)
    sample_spec = tensor_spec.TensorSpec((1,), dtype=tf.int32)

    spec = distribution_spec.DistributionSpec(
        tfd.Categorical,
        input_param_spec,
        sample_spec=sample_spec,
        **expected_distribution.parameters)

    self.assertEqual(expected_distribution.parameters['logits'],
                     spec.distribution_parameters['logits'])

    distribution = spec.build_distribution(logits=[0.1, 0.4, 0.5])

    self.assertTrue(isinstance(distribution, tfd.Categorical))
    self.assertTrue(distribution.parameters['validate_args'])
    self.assertEqual([0.1, 0.4, 0.5], distribution.parameters['logits'])


if __name__ == '__main__':
  tf.test.main()
