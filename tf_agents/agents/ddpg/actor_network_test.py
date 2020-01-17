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

"""Tests for tf_agents.agents.ddpg.actor_network."""

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import actor_network
from tf_agents.specs import tensor_spec

from tensorflow.python.framework import test_util  # TF internal


class ActorNetworkTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 3
    num_obs_dims = 5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testAddConvLayers(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([3, 3, num_obs_dims], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)

    actor_net = actor_network.ActorNetwork(
        obs_spec, action_spec, conv_layer_params=[(16, 3, 2)])

    obs = tf.random.uniform([batch_size, 3, 3, num_obs_dims])
    actions, _ = actor_net(obs)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)

    actor_net = actor_network.ActorNetwork(
        obs_spec, action_spec, fc_layer_params=[100])

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testScalarAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec([], tf.float32, 2., 3.)

    actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def test2DAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
    actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testActionsWithinRange(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
    actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(actions)
    self.assertTrue(np.all(actions_ >= action_spec.minimum))
    self.assertTrue(np.all(actions_ <= action_spec.maximum))

  @test_util.run_in_graph_and_eager_modes()
  def testListOfSingleAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)]

    actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)

    self.assertAllEqual(actions[0].shape.as_list(),
                        [batch_size] + action_spec[0].shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testDictOfSingleAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = {
        'motor': tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    }
    actor_net = actor_network.ActorNetwork(obs_spec, action_spec)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions, _ = actor_net(obs)
    self.assertAllEqual(actions['motor'].shape.as_list(),
                        [batch_size] + action_spec['motor'].shape.as_list())
    self.assertEqual(len(actor_net.trainable_variables), 2)

if __name__ == '__main__':
  tf.test.main()
