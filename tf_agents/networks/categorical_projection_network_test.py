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

"""Tests for tf_agents.networks.categorical_projection_network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.networks import categorical_projection_network
from tf_agents.specs import tensor_spec


def _get_inputs(batch_size, num_input_dims):
  return tf.random.uniform([batch_size, num_input_dims])


class CategoricalProjectionNetworkTest(tf.test.TestCase):

  def testBuild(self):
    output_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.int32, 0, 1)
    network = categorical_projection_network.CategoricalProjectionNetwork(
        output_spec)

    inputs = _get_inputs(batch_size=3, num_input_dims=5)

    distribution, _ = network(inputs, outer_rank=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    sample = self.evaluate(distribution.sample())

    self.assertEqual(tfp.distributions.Categorical, type(distribution))
    # Batch = 3; 2x3 action choices, 2x actions per choise.
    self.assertEqual((3, 2, 3, 2), distribution.logits.shape)
    self.assertAllEqual((3, 2, 3), sample.shape)

  def testTrainableVariables(self):
    output_spec = tensor_spec.BoundedTensorSpec([2], tf.int32, 0, 1)
    network = categorical_projection_network.CategoricalProjectionNetwork(
        output_spec)

    inputs = _get_inputs(batch_size=3, num_input_dims=5)

    network(inputs, outer_rank=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Dense kernel, dense bias.
    self.assertEqual(2, len(network.trainable_variables))
    self.assertEqual((5, 4), network.trainable_variables[0].shape)
    self.assertEqual((4,), network.trainable_variables[1].shape)


if __name__ == '__main__':
  tf.test.main()
