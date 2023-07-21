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

"""Tests for tf_agents.network.value_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import value_network
from tf_agents.specs import tensor_spec

from tensorflow.python.framework import test_util  # TF internal


class ValueNetworkTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuilds(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    observation = tensor_spec.sample_spec_nest(
        observation_spec, outer_dims=(1,))

    net = value_network.ValueNetwork(
        observation_spec, conv_layer_params=[(4, 2, 2)], fc_layer_params=(5,))

    value, _ = net(observation)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertEqual([1], value.shape.as_list())

    self.assertEqual(6, len(net.variables))
    # Conv Net Kernel
    self.assertEqual((2, 2, 3, 4), net.variables[0].shape)
    # Conv Net bias
    self.assertEqual((4,), net.variables[1].shape)
    # Fc Kernel
    self.assertEqual((64, 5), net.variables[2].shape)
    # Fc Bias
    self.assertEqual((5,), net.variables[3].shape)
    # Value Shrink Kernel
    self.assertEqual((5, 1), net.variables[4].shape)
    # Value Shrink bias
    self.assertEqual((1,), net.variables[5].shape)

  @test_util.run_in_graph_and_eager_modes()
  def testHandlesExtraOuterDims(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    observation = tensor_spec.sample_spec_nest(
        observation_spec, outer_dims=(3, 3, 2))

    net = value_network.ValueNetwork(
        observation_spec, conv_layer_params=[(4, 2, 2)], fc_layer_params=(5,))

    value, _ = net(observation)
    self.assertEqual([3, 3, 2], value.shape.as_list())

  @test_util.run_in_graph_and_eager_modes()
  def testHandlePreprocessingLayers(self):
    observation_spec = (tensor_spec.TensorSpec([1], tf.float32),
                        tensor_spec.TensorSpec([], tf.float32))
    observation = tensor_spec.sample_spec_nest(
        observation_spec, outer_dims=(3,))

    preprocessing_layers = (tf.keras.layers.Dense(4),
                            tf.keras.Sequential([
                                tf.keras.layers.Reshape((1,)),
                                tf.keras.layers.Dense(4)
                            ]))

    net = value_network.ValueNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add())

    value, _ = net(observation)
    self.assertEqual([3], value.shape.as_list())
    self.assertGreater(len(net.trainable_variables), 4)


if __name__ == '__main__':
  tf.test.main()
