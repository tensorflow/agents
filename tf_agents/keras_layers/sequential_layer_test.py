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

"""Tests for tf_agents.keras_layers.sequential_layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import sequential_layer
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class ActorNetwork(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(ActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='TestActorNetwork')
    num_actions = output_tensor_spec.shape.num_elements()
    self._sequential_layer = sequential_layer.SequentialLayer([
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(num_actions)
    ])

  def call(self, observations, step_type=(), network_state=(), training=False):
    return self._sequential_layer(observations), network_state


class SequentialLayerTest(test_utils.TestCase):

  def testBuild(self):
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(4, use_bias=False),
         tf.keras.layers.ReLU()])
    inputs = np.ones((2, 3))
    out = sequential(inputs)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    out = self.evaluate(out)
    weights = self.evaluate(sequential.layers[0].weights[0])
    expected = np.dot(inputs, weights)
    expected[expected < 0] = 0
    self.assertAllClose(expected, out)

  def testTrainableVariables(self):
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(3),
         tf.keras.layers.Dense(4)])
    sequential.build((3, 2))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    variables = self.evaluate(sequential.trainable_variables)
    self.assertLen(variables, 4)
    self.assertLen(sequential.variables, 4)
    self.assertTrue(sequential.trainable)
    sequential.trainable = False
    self.assertFalse(sequential.trainable)
    self.assertEmpty(sequential.trainable_variables)
    self.assertLen(sequential.variables, 4)

  def testTrainableVariablesNestedNetwork(self):
    sequential_inner = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(3),
         tf.keras.layers.Dense(4)])
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(3),
         sequential_inner])
    sequential.build((3, 2))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    variables = self.evaluate(sequential.trainable_variables)

    self.assertLen(variables, 6)
    self.assertLen(sequential.variables, 6)
    self.assertLen(sequential_inner.variables, 4)
    self.assertTrue(sequential.trainable)
    sequential.trainable = False
    self.assertFalse(sequential.trainable)
    self.assertEmpty(sequential.trainable_variables)
    self.assertLen(sequential.variables, 6)

  def testCopy(self):
    sequential = sequential_layer.SequentialLayer(
        [tf.keras.layers.Dense(3),
         tf.keras.layers.Dense(4, use_bias=False)])
    clone = type(sequential).from_config(sequential.get_config())
    self.assertLen(clone.layers, 2)
    for l1, l2 in zip(sequential.layers, clone.layers):
      self.assertEqual(l1.dtype, l2.dtype)
      self.assertEqual(l1.units, l2.units)
      self.assertEqual(l1.use_bias, l2.use_bias)

  def testPolicySaverCompatibility(self):
    observation_spec = tensor_spec.TensorSpec(shape=(100,), dtype=tf.float32)
    action_spec = tensor_spec.TensorSpec(shape=(5,), dtype=tf.float32)
    time_step_tensor_spec = ts.time_step_spec(observation_spec)
    net = ActorNetwork(observation_spec, action_spec)
    net.create_variables()
    policy = actor_policy.ActorPolicy(time_step_tensor_spec, action_spec, net)

    sample = tensor_spec.sample_spec_nest(
        time_step_tensor_spec, outer_dims=(5,))

    policy.action(sample)

    train_step = common.create_variable('train_step')
    saver = policy_saver.PolicySaver(policy, train_step=train_step)
    self.initialize_v1_variables()

    with self.cached_session():
      saver.save(os.path.join(FLAGS.test_tmpdir, 'sequential_layer_model'))


if __name__ == '__main__':
  test_utils.main()
