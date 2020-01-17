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

"""Tests for tf_agents.networks.categorical_q_network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import categorical_q_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class CategoricalQNetworkTest(test_utils.TestCase):

  def tearDown(self):
    gin.clear_config()
    super(CategoricalQNetworkTest, self).tearDown()

  def testBuild(self):
    batch_size = 3
    num_state_dims = 5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self.assertEqual(num_actions, 2)

    observations_spec = tensor_spec.TensorSpec([num_state_dims], tf.float32)
    observations = tf.random.uniform([batch_size, num_state_dims])
    time_steps = ts.restart(observations, batch_size)

    q_network = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=observations_spec,
        action_spec=action_spec,
        fc_layer_params=[3])

    logits, _ = q_network(time_steps.observation)
    self.assertAllEqual(logits.shape.as_list(),
                        [batch_size, num_actions, q_network._num_atoms])

    # There are two trainable layers here: the specified fc_layer and the final
    # logits layer. Each layer has two trainable_variables (kernel and bias),
    # for a total of 4.
    self.assertLen(q_network.trainable_variables, 4)

  def testChangeHiddenLayers(self):
    batch_size = 3
    num_state_dims = 5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self.assertEqual(num_actions, 2)

    observations_spec = tensor_spec.TensorSpec([num_state_dims], tf.float32)
    observations = tf.random.uniform([batch_size, num_state_dims])
    time_steps = ts.restart(observations, batch_size)

    q_network = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=observations_spec,
        action_spec=action_spec,
        fc_layer_params=[3, 3])

    logits, _ = q_network(time_steps.observation)
    self.assertAllEqual(logits.shape.as_list(),
                        [batch_size, num_actions, q_network._num_atoms])

    # This time there is an extra fc layer, for a total of 6
    # trainable_variables.
    self.assertLen(q_network.trainable_variables, 6)

  def testAddConvLayers(self):
    batch_size = 3
    num_state_dims = 5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self.assertEqual(num_actions, 2)

    observations_spec = tensor_spec.TensorSpec(
        [3, 3, num_state_dims], tf.float32)
    observations = tf.random.uniform([batch_size, 3, 3, num_state_dims])
    time_steps = ts.restart(observations, batch_size)

    q_network = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=observations_spec,
        action_spec=action_spec,
        conv_layer_params=[(16, 2, 1), (15, 2, 1)])

    logits, _ = q_network(time_steps.observation)
    self.assertAllEqual(logits.shape.as_list(),
                        [batch_size, num_actions, q_network._num_atoms])

    # This time there are two conv layers and one final logits layer, for a
    # total of 6 trainable_variables.
    self.assertLen(q_network.trainable_variables, 6)

  def testCorrectOutputShape(self):
    batch_size = 3
    num_state_dims = 5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self.assertEqual(num_actions, 2)

    observations_spec = tensor_spec.TensorSpec([num_state_dims], tf.float32)
    observations = tf.random.uniform([batch_size, num_state_dims])
    time_steps = ts.restart(observations, batch_size)

    q_network = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=observations_spec,
        action_spec=action_spec,
        fc_layer_params=[3])

    logits, _ = q_network(time_steps.observation)
    self.assertAllEqual(logits.shape.as_list(),
                        [batch_size, num_actions, q_network._num_atoms])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    eval_logits = self.evaluate(logits)
    self.assertAllEqual(
        eval_logits.shape, [batch_size, num_actions, q_network._num_atoms])

  def testGinConfig(self):
    batch_size = 3
    num_state_dims = 5
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self.assertEqual(num_actions, 2)

    observations_spec = tensor_spec.TensorSpec(
        [3, 3, num_state_dims], tf.float32)
    observations = tf.random.uniform([batch_size, 3, 3, num_state_dims])
    next_observations = tf.random.uniform([batch_size, 3, 3, num_state_dims])
    time_steps = ts.restart(observations, batch_size)
    next_time_steps = ts.restart(next_observations, batch_size)

    # Note: this is cleared in tearDown().
    gin.parse_config("""
        CategoricalQNetwork.conv_layer_params = [(16, 2, 1), (15, 2, 1)]
        CategoricalQNetwork.fc_layer_params = [4, 3, 5]
    """)

    q_network = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=observations_spec,
        action_spec=action_spec)

    logits, _ = q_network(time_steps.observation)
    next_logits, _ = q_network(next_time_steps.observation)
    self.assertAllEqual(logits.shape.as_list(),
                        [batch_size, num_actions, q_network.num_atoms])
    self.assertAllEqual(next_logits.shape.as_list(),
                        [batch_size, num_actions, q_network.num_atoms])

    # This time there are six layers: two conv layers, three fc layers, and one
    # final logits layer, for 12 trainable_variables in total.
    self.assertLen(q_network.trainable_variables, 12)


if __name__ == '__main__':
  tf.test.main()
