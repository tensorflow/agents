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

"""Tests for tf_agents.network.value_rnn_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import value_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


class ValueRnnNetworkTest(tf.test.TestCase):

  def testBuilds(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1, 3))

    net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        conv_layer_params=[(4, 2, 2)],
        input_fc_layer_params=(5,),
        lstm_size=(7,),
        output_fc_layer_params=(3,))

    value, state = net(time_step.observation, step_type=time_step.step_type,
                       network_state=net.get_initial_state(batch_size=1))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertEqual((1, 3), value.shape)

    self.assertEqual(11, len(net.variables))
    # Conv Net Kernel
    self.assertEqual((2, 2, 3, 4), net.variables[0].shape)
    # Conv Net bias
    self.assertEqual((4,), net.variables[1].shape)
    # Fc Kernel
    self.assertEqual((64, 5), net.variables[2].shape)
    # Fc Bias
    self.assertEqual((5,), net.variables[3].shape)
    # LSTM Cell Kernel
    self.assertEqual((5, 28), net.variables[4].shape)
    # LSTM Cell Recurrent Kernel
    self.assertEqual((7, 28), net.variables[5].shape)
    # LSTM Cell Bias
    self.assertEqual((28,), net.variables[6].shape)
    # Fc Kernel
    self.assertEqual((7, 3), net.variables[7].shape)
    # Fc Bias
    self.assertEqual((3,), net.variables[8].shape)
    # Value Shrink Kernel
    self.assertEqual((3, 1), net.variables[9].shape)
    # Value Shrink bias
    self.assertEqual((1,), net.variables[10].shape)

    # Assert LSTM cell is created.
    self.assertEqual((1, 7), state[0].shape)
    self.assertEqual((1, 7), state[1].shape)

  def testBuildsStackedLstm(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1, 3))

    net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        conv_layer_params=[(4, 2, 2)],
        input_fc_layer_params=(5,),
        lstm_size=(7, 5),
        output_fc_layer_params=(3,))

    _, state = net(time_step.observation,
                   step_type=time_step.step_type,
                   network_state=net.get_initial_state(batch_size=1))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Assert LSTM cell is created.
    self.assertEqual((1, 7), state[0][0].shape)
    self.assertEqual((1, 7), state[0][1].shape)

    # Assert LSTM cell is created.
    self.assertEqual((1, 5), state[1][0].shape)
    self.assertEqual((1, 5), state[1][1].shape)

  def testHandleBatchOnlyObservation(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(3,))

    net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        conv_layer_params=[(4, 2, 2)],
        input_fc_layer_params=(5,),
        lstm_size=(7, 5),
        output_fc_layer_params=(3,))

    value, _ = net(time_step.observation,
                   step_type=time_step.step_type,
                   network_state=net.get_initial_state(batch_size=3))
    self.assertEqual([3], value.shape.as_list())

  def testHandlePreprocessingLayers(self):
    observation_spec = (tensor_spec.TensorSpec([1], tf.float32),
                        tensor_spec.TensorSpec([], tf.float32))
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(2, 3))

    preprocessing_layers = (tf.keras.layers.Dense(4),
                            tf.keras.Sequential([
                                tf.keras.layers.Reshape((1,)),
                                tf.keras.layers.Dense(4)
                            ]))

    net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add())

    value, _ = net(time_step.observation,
                   step_type=time_step.step_type,
                   network_state=net.get_initial_state(batch_size=2))
    self.assertEqual([2, 3], value.shape.as_list())
    self.assertGreater(len(net.trainable_variables), 4)


if __name__ == '__main__':
  tf.test.main()
