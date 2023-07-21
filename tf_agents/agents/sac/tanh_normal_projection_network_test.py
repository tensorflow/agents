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

"""Tests for tf_agents.networks.normal_projection_network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec


def _get_inputs(batch_size, num_input_dims):
  return tf.random.uniform([batch_size, num_input_dims])


class TanhNormalProjectionNetworkTest(tf.test.TestCase):

  def testBuild(self):
    output_spec = tensor_spec.BoundedTensorSpec([2], tf.float32, 0, 1)
    network = tanh_normal_projection_network.TanhNormalProjectionNetwork(
        output_spec)

    inputs = _get_inputs(batch_size=3, num_input_dims=5)

    distribution, _ = network(inputs, outer_rank=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(tfp.distributions.MultivariateNormalDiag,
                     type(distribution.input_distribution))

    means = distribution.input_distribution.loc
    stds = distribution.input_distribution.scale

    self.assertAllEqual(means.shape.as_list(),
                        [3] + output_spec.shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3] + output_spec.shape.as_list()*2)

  def testTrainableVariables(self):
    output_spec = tensor_spec.BoundedTensorSpec([2], tf.float32, 0, 1)
    network = tanh_normal_projection_network.TanhNormalProjectionNetwork(
        output_spec)

    inputs = _get_inputs(batch_size=3, num_input_dims=5)

    network(inputs, outer_rank=1)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Dense kernel and bias.
    self.assertEqual(2, len(network.trainable_variables))
    self.assertEqual((5, 4), network.trainable_variables[0].shape)
    self.assertEqual((4,), network.trainable_variables[1].shape)

  def testSequentialNetwork(self):
    output_spec = tensor_spec.BoundedTensorSpec([2], tf.float32, 0, 1)
    network = tanh_normal_projection_network.TanhNormalProjectionNetwork(
        output_spec)

    inputs = tf.random.stateless_uniform(shape=[3, 5], seed=[0, 0])
    output, _ = network(inputs, outer_rank=1)

    # Create a squashed distribution.
    def create_dist(loc_and_scale):
      ndims = output_spec.shape.num_elements()
      loc = loc_and_scale[..., :ndims]
      scale = tf.exp(loc_and_scale[..., ndims:])

      distribution = tfp.distributions.MultivariateNormalDiag(
          loc=loc,
          scale_diag=scale,
          validate_args=True,
      )
      return distribution_utils.scale_distribution_to_spec(
          distribution, output_spec)

    # Create a sequential network.
    sequential_network = sequential.Sequential(
        [network._projection_layer] + [tf.keras.layers.Lambda(create_dist)])
    sequential_output, _ = sequential_network(inputs)

    # Check that mode and standard deviation are the same.
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(output.mode()), self.evaluate(sequential_output.mode()))
    self.assertAllClose(
        self.evaluate(output.stddev()), self.evaluate(output.stddev()))


if __name__ == '__main__':
  tf.test.main()
