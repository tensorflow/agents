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

"""Tests for tf_agents.networks.sequential."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import os

from absl import flags

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import network
from tf_agents.networks import sequential as sequential_lib
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class ActorNetwork(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    num_actions = output_tensor_spec.shape.num_elements()
    self._sequential = sequential_lib.Sequential(
        [
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(num_actions)
        ],
        input_spec=input_tensor_spec)
    super(ActorNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=self._sequential.state_spec,
        name='TestActorNetwork')

  def call(self, observations, step_type=(), network_state=(), training=False):
    return self._sequential(observations, network_state)


class SequentialTest(test_utils.TestCase):

  def setUp(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x.')
    super(SequentialTest, self).setUp()

  def testCall(self):
    sequential = sequential_lib.Sequential(
        [tf.keras.layers.Dense(4, use_bias=False),
         tf.keras.layers.ReLU()],
        input_spec=tf.TensorSpec((3,), tf.float32))
    inputs = np.ones((2, 3))
    out, state = sequential(inputs)
    self.assertEqual(state, ())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    out = self.evaluate(out)
    weights = self.evaluate(sequential.layers[0].weights[0])
    expected = np.dot(inputs, weights)
    expected[expected < 0] = 0
    self.assertAllClose(expected, out)

  def testMixOfNonRecurrentAndRecurrent(self):
    sequential = sequential_lib.Sequential(
        [
            tf.keras.layers.Dense(2),
            tf.keras.layers.LSTM(2, return_state=True, return_sequences=True),
            tf.keras.layers.RNN(
                tf.keras.layers.StackedRNNCells([
                    tf.keras.layers.LSTMCell(1),
                    tf.keras.layers.LSTMCell(32),
                ],),
                return_state=True,
                return_sequences=True,
            ),
            # Convert inner dimension to [4, 4, 2] for convolution.
            inner_reshape.InnerReshape([32], [4, 4, 2]),
            tf.keras.layers.Conv2D(2, 3),
            # Convert 3 inner dimensions to [?] for RNN.
            inner_reshape.InnerReshape([None] * 3, [-1]),
            tf.keras.layers.GRU(2, return_state=True, return_sequences=True),
            dynamic_unroll_layer.DynamicUnroll(tf.keras.layers.LSTMCell(2)),
        ],
        input_spec=tf.TensorSpec((3,), tf.float32))
    self.assertEqual(
        sequential.input_tensor_spec, tf.TensorSpec((3,), tf.float32))

    output_spec = sequential.create_variables()
    self.assertEqual(output_spec, tf.TensorSpec((2,), dtype=tf.float32))

    tf.nest.map_structure(
        self.assertEqual,
        sequential.state_spec,
        (
            [  # LSTM
                tf.TensorSpec((2,), tf.float32),
                tf.TensorSpec((2,), tf.float32),
            ],
            (  # RNN(StackedRNNCells)
                [
                    tf.TensorSpec((1,), tf.float32),
                    tf.TensorSpec((1,), tf.float32),
                ],
                [
                    tf.TensorSpec((32,), tf.float32),
                    tf.TensorSpec((32,), tf.float32),
                ],
            ),
            # GRU
            tf.TensorSpec((2,), tf.float32),
            [  # DynamicUnroll
                tf.TensorSpec((2,), tf.float32),
                tf.TensorSpec((2,), tf.float32),
            ]))

    inputs = tf.ones((8, 10, 3), dtype=tf.float32)
    outputs, _ = sequential(inputs)
    self.assertEqual(outputs.shape, tf.TensorShape([8, 10, 2]))

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
      saver.save(os.path.join(FLAGS.test_tmpdir, 'sequential_model'))


if __name__ == '__main__':
  test_utils.main()
