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

"""Tests for tf_agents.agents.ddpg.critic_network."""

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.specs import tensor_spec

from tensorflow.python.framework import test_util  # TF internal


class CriticNetworkTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2
    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    critic_net = critic_network.CriticNetwork((obs_spec, action_spec))

    q_values, _ = critic_net((obs, actions))
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_net.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testAddObsConvLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2

    obs_spec = tensor_spec.TensorSpec([3, 3, num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec), observation_conv_layer_params=[(16, 3, 2)])

    obs = tf.random.uniform([batch_size, 3, 3, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    q_values, _ = critic_net((obs, actions))
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_net.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddObsFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2

    obs_spec = tensor_spec.TensorSpec([3, 3, num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec), observation_fc_layer_params=[20, 10])

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    q_values, _ = critic_net((obs, actions))

    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_net.trainable_variables), 6)

  @test_util.run_in_graph_and_eager_modes()
  def testAddActionFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2

    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec), action_fc_layer_params=[20])

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    q_values, _ = critic_net((obs, actions))
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_net.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddJointFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2

    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec), joint_fc_layer_params=[20])

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    q_values, _ = critic_net((obs, actions))
    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertEqual(len(critic_net.trainable_variables), 4)

if __name__ == '__main__':
  tf.test.main()
