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

"""Tests for tf_agents.network.q_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.networks import q_network
from tf_agents.specs import tensor_spec

from tensorflow.python.framework import test_util  # TF internal


class SingleObservationSingleActionTest(tf.test.TestCase):

  def setUp(self):
    super(SingleObservationSingleActionTest, self).setUp()
    gin.clear_config()

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_weights), 6)

  @test_util.run_in_graph_and_eager_modes()
  def testChangeHiddenLayers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        fc_layer_params=(40,))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddConvLayers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, 5, 5, num_state_dims])
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([5, 5, num_state_dims],
                                                 tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        conv_layer_params=((16, 3, 2),))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 8)

  @test_util.run_in_graph_and_eager_modes()
  def testAddPreprocessingLayers(self):
    batch_size = 3
    num_actions = 2
    states = (tf.random.uniform([batch_size, 1]),
              tf.random.uniform([batch_size]))
    preprocessing_layers = (
        tf.keras.layers.Dense(4),
        tf.keras.Sequential([
            tf.keras.layers.Reshape((1,)),
            tf.keras.layers.Dense(4)]))
    network = q_network.QNetwork(
        input_tensor_spec=(
            tensor_spec.TensorSpec([1], tf.float32),
            tensor_spec.TensorSpec([], tf.float32)),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add(),
        action_spec=tensor_spec.BoundedTensorSpec(
            [1], tf.int32, 0, num_actions - 1))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    # At least 2 variables each for the preprocessing layers.
    self.assertGreater(len(network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testCorrectOutputShape(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])

  @test_util.run_in_graph_and_eager_modes()
  def testNetworkVariablesAreReused(self):
    batch_size = 3
    num_state_dims = 5
    states = tf.ones([batch_size, num_state_dims])
    next_states = tf.ones([batch_size, num_state_dims])
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    next_q_values, _ = network(next_states)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(q_values, next_q_values)

  def testVariablesBuild(self):
    num_state_dims = 5
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    self.assertFalse(network.built)
    variables = network.variables
    self.assertTrue(network.built)
    self.assertGreater(len(variables), 0)

  def testPreprocessingLayersSingleObsevations(self):
    """Tests using preprocessing_layers without preprocessing_combiner."""
    num_state_dims = 5
    network = q_network.QNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        preprocessing_layers=tf.keras.layers.Lambda(lambda x: x),
        preprocessing_combiner=None)
    q_logits, _ = network(tf.ones((3, num_state_dims)))
    self.assertAllEqual(q_logits.shape.as_list(), [3, 2])


if __name__ == '__main__':
  tf.test.main()
