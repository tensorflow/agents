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

import tensorflow as tf

from tf_agents.agents.dqn import q_network
from tf_agents.specs import tensor_spec

import gin.tf
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
    states = tf.random_uniform([batch_size, num_state_dims])
    network = q_network.QNetwork(
        observation_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_weights), 6)

  @test_util.run_in_graph_and_eager_modes()
  def testChangeHiddenLayers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random_uniform([batch_size, num_state_dims])
    network = q_network.QNetwork(
        observation_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
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
    states = tf.random_uniform([batch_size, 5, 5, num_state_dims])
    network = q_network.QNetwork(
        observation_spec=tensor_spec.TensorSpec([5, 5, num_state_dims],
                                                tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        conv_layer_params=((16, 3, 2),))
    q_values, _ = network(states)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 8)

  @test_util.run_in_graph_and_eager_modes()
  def testCorrectOutputShape(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random_uniform([batch_size, num_state_dims])
    network = q_network.QNetwork(
        observation_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
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
        observation_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    q_values, _ = network(states)
    next_q_values, _ = network(next_states)
    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(q_values, next_q_values)

if __name__ == '__main__':
  tf.test.main()
