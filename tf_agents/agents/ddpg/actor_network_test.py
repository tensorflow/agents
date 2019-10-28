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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.agents.ddpg import actor_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import sequential_layer

from tensorflow.python.framework import test_util  # TF internal


class ActorNetworkTest(tf.test.TestCase, parameterized.TestCase):

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

  @test_util.run_in_graph_and_eager_modes()
  def testHandlePreprocessingLayers(self):
    observation_spec = (tensor_spec.TensorSpec([1], tf.float32),
                        tensor_spec.TensorSpec([], tf.float32))
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(3,))

    action_spec = tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3)

    preprocessing_layers = (tf.keras.layers.Dense(4),
                            sequential_layer.SequentialLayer([
                                tf.keras.layers.Reshape((1,)),
                                tf.keras.layers.Dense(4)
                            ]))

    net = actor_network.ActorNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add())

    action, _ = net(time_step.observation, time_step.step_type,
                                  ())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([3, 2], action.shape.as_list())
    self.assertGreater(len(net.trainable_variables), 4)

  @parameterized.named_parameters(
      ('TrainingTrue', True,),
      ('TrainingFalse', False))
  def testDropoutFCLayersWithConv(self, training):
    tf.compat.v1.set_random_seed(0)
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(8, 4))
    action_spec = tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3)

    net = actor_network.ActorNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(4, 2, 2)],
        fc_layer_params=[5],
        dropout_layer_params=[0.5])

    action1, _ = net(
        time_step.observation, time_step.step_type, (), training=training)
    action2, _ = net(
        time_step.observation, time_step.step_type, (), training=training)

    self.assertEqual([8, 4, 2], action1.shape.as_list())
    if training:
      self.assertGreater(np.linalg.norm(action1 - action2), 0)
    else:
      self.assertAllEqual(action1, action2)

if __name__ == '__main__':
  tf.test.main()
