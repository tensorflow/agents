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

"""Tests for tf_agents.agents.ddpg.critic_network."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import critic_network
from tf_agents.specs import tensor_spec


class CriticNetworkTest(tf.test.TestCase, parameterized.TestCase):

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
    self.assertLen(critic_net.trainable_variables, 2)

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
    self.assertLen(critic_net.trainable_variables, 4)

  def testAddObsFCLayers(self):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2

    obs_spec = tensor_spec.TensorSpec([3, num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec), observation_fc_layer_params=[20, 10])

    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    q_values, _ = critic_net((obs, actions))

    self.assertAllEqual(q_values.shape.as_list(), [batch_size])
    self.assertLen(critic_net.trainable_variables, 6)

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
    self.assertLen(critic_net.trainable_variables, 4)

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
    self.assertLen(critic_net.trainable_variables, 4)

  @parameterized.named_parameters(
      ('TrainingTrue', True,),
      ('TrainingFalse', False))
  def testDropoutJointFCLayers(self, training):
    batch_size = 3
    num_obs_dims = 5
    num_actions_dims = 2

    obs_spec = tensor_spec.TensorSpec([num_obs_dims], tf.float32)
    action_spec = tensor_spec.TensorSpec([num_actions_dims], tf.float32)
    critic_net = critic_network.CriticNetwork(
        (obs_spec, action_spec),
        joint_fc_layer_params=[20],
        joint_dropout_layer_params=[0.5])
    obs = tf.random.uniform([batch_size, num_obs_dims])
    actions = tf.random.uniform([batch_size, num_actions_dims])
    q_values1, _ = critic_net((obs, actions), training=training)
    q_values2, _ = critic_net((obs, actions), training=training)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    q_values1, q_values2 = self.evaluate([q_values1, q_values2])
    if training:
      self.assertGreater(np.linalg.norm(q_values1 - q_values2), 0)
    else:
      self.assertAllEqual(q_values1, q_values2)


if __name__ == '__main__':
  tf.test.main()
