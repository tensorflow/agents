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

"""Tests for tf_agents.agents.ddpg.networks."""

import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import networks
from tf_agents.environments import time_step as ts
from tf_agents.specs import tensor_spec

from tensorflow.python.framework import test_util  # TF internal

# TODO(kbanoop): Delete after Keras networks have been completely integrated.


class CriticTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    actions = tf.random_uniform([batch_size, num_actions_dims])
    critic_network = tf.make_template('critic_network',
                                      networks.critic_network)
    q_values = critic_network(time_steps, actions)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_network.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testAddObsConvLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2
    obs = tf.random_uniform([batch_size, 3, 3, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    actions = tf.random_uniform([batch_size, num_actions_dims])
    critic_network = tf.make_template('critic_network',
                                      networks.critic_network,
                                      observation_conv_layers=[(16, 3, 2)])
    q_values = critic_network(time_steps, actions)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddObsFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    actions = tf.random_uniform([batch_size, num_actions_dims])
    critic_network = tf.make_template('critic_network',
                                      networks.critic_network,
                                      observation_fc_layers=[20, 10])
    q_values = critic_network(time_steps, actions)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_network.trainable_variables), 6)

  @test_util.run_in_graph_and_eager_modes()
  def testAddActionFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    actions = tf.random_uniform([batch_size, num_actions_dims])
    critic_network = tf.make_template('critic_network',
                                      networks.critic_network,
                                      action_fc_layers=[20])
    q_values = critic_network(time_steps, actions)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddJointFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    actions = tf.random_uniform([batch_size, num_actions_dims])
    critic_network = tf.make_template('critic_network',
                                      networks.critic_network,
                                      joint_fc_layers=[20])
    q_values = critic_network(time_steps, actions)
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_network.trainable_variables), 4)


class ActorTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testAddConvLayers(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, 3, 3, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network,
                                     conv_layers=[(16, 3, 2)])
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network,
                                     fc_layers=[100])
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testScalarAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def test2DAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testActionsWithinRange(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    actions = actor_network(time_steps, action_spec)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(actions)
    self.assertTrue(np.all(actions_ >= action_spec.minimum))
    self.assertTrue(np.all(actions_ <= action_spec.maximum))

  @test_util.run_in_graph_and_eager_modes()
  def testListOfSingleAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)]
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions[0].shape.as_list(),
                        [batch_size] + action_spec[0].shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testDictOfSingleAction(self):
    batch_size = 3
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    time_steps = ts.restart(obs, batch_size)
    action_spec = {
        'motor': tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    }
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    actions = actor_network(time_steps, action_spec)
    self.assertAllEqual(actions['motor'].shape.as_list(),
                        [batch_size] + action_spec['motor'].shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 2)

if __name__ == '__main__':
  tf.test.main()
