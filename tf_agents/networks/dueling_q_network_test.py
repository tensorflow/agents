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

"""Tests for tf_agents.networks.dueling_q_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util  # TF internal
import gin.tf
import numpy as np

from tf_agents.networks import dueling_q_network
from tf_agents.specs import tensor_spec


class SingleObservationSingleActionTest(tf.test.TestCase):

  def setUp(self):
    super(SingleObservationSingleActionTest, self).setUp()
    gin.clear_config()

  @test_util.run_in_graph_and_eager_modes()
  def test_network_builds(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_weights), 8)

  @test_util.run_in_graph_and_eager_modes()
  def test_change_hidden_layers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        fc_layer_params=(40,))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 6)

  @test_util.run_in_graph_and_eager_modes()
  def test_add_conv_layers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, 5, 5, num_state_dims])
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([5, 5, num_state_dims],
                                                 tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        conv_layer_params=((16, 3, 2),))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 10)

  @test_util.run_in_graph_and_eager_modes()
  def test_add_preprocessing_layers(self):
    batch_size = 3
    num_actions = 2
    states = (tf.random.uniform([batch_size, 1]),
              tf.random.uniform([batch_size]))
    preprocessing_layers = (
        tf.keras.layers.Dense(4),
        tf.keras.Sequential([
            tf.keras.layers.Reshape((1,)),
            tf.keras.layers.Dense(4)]))
    network = dueling_q_network.DuelingQNetwork(
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
  def test_correct_output_shape(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])

  @test_util.run_in_graph_and_eager_modes()
  def test_network_variables_are_reused(self):
    batch_size = 3
    num_state_dims = 5
    states = tf.ones([batch_size, num_state_dims])
    next_states = tf.ones([batch_size, num_state_dims])
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    next_q_values, _ = network(next_states)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(q_values, next_q_values)

  def test_variables_build(self):
    num_state_dims = 5
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    self.assertFalse(network.built)
    variables = network.variables
    self.assertTrue(network.built)
    self.assertGreater(len(variables), 0)

  @test_util.run_in_graph_and_eager_modes()
  def test_network_outputs_correct_values(self):
    tf.random.set_random_seed(123)
    batch_size = 1
    num_state_dims = 5
    states = tf.constant(np.random.uniform(
        -1,
        1,
        (batch_size, num_state_dims)).astype(np.float32))
    #define the network
    network = dueling_q_network.DuelingQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        fc_layer_params=(10,))
    #get the q_values from network
    q_values, _ = network(states)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    #get the values of the weights and biases
    weights_fc = tf.identity(network.layers[0].get_weights()[0])
    bias_fc = tf.identity(network.layers[0].get_weights()[1])
    weights_values = tf.identity(network.layers[1].get_weights()[0])
    bias_values = tf.identity(network.layers[1].get_weights()[1])
    weights_adv = tf.identity(network.layers[2].get_weights()[0])
    bias_adv = tf.identity(network.layers[2].get_weights()[1])

    #compute the expected q_values manually
    fc = tf.add(tf.matmul(states, weights_fc), bias_fc)
    fc = tf.nn.relu(fc)
    advantage = tf.add(tf.matmul(fc, weights_adv), bias_adv)
    value = tf.add(tf.matmul(fc, weights_values), bias_values)
    expected_q_values = tf.add(value, tf.subtract(
        advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True)))

    #compare the q_values from network with the expected q_values
    self.assertAllEqual(q_values, expected_q_values)

if __name__ == '__main__':
  tf.test.main()
