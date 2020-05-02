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

"""Tests for tf_agents.agents.ddpg.critic_rnn_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


def lstm_keras_fn(lstm_size):
  return tf.keras.layers.LSTM(lstm_size, return_state=True,
                              return_sequences=True)


def rnn_keras_fn(lstm_size):
  cell = tf.keras.layers.SimpleRNNCell(lstm_size)
  return tf.keras.layers.RNN(cell, return_state=True,
                             return_sequences=True)


class CriticRnnNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('RNNKerasUnroll', None, rnn_keras_fn),
  )
  def testBuildsRnn(self, lstm_size, rnn_construction_fn):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))

    action_spec = tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3)
    network_input_spec = (observation_spec, action_spec)
    action = tensor_spec.sample_spec_nest(action_spec, outer_dims=(1,))
    net = critic_rnn_network.CriticRnnNetwork(
        network_input_spec,
        observation_conv_layer_params=[(4, 2, 2)],
        observation_fc_layer_params=(4,),
        action_fc_layer_params=(5,),
        joint_fc_layer_params=(5,),
        lstm_size=lstm_size,
        output_fc_layer_params=(5,),
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs={'lstm_size': 3})

    network_input = (time_step.observation, action)
    q_values, network_state = net(network_input, time_step.step_type,
                                  net.get_initial_state(batch_size=1))

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([1], q_values.shape.as_list())

    self.assertLen(net.variables, 15)
    # Obs Conv Net Kernel
    self.assertEqual((2, 2, 3, 4), net.variables[0].shape)
    # Obs Conv Net bias
    self.assertEqual((4,), net.variables[1].shape)
    # Obs Fc Kernel
    self.assertEqual((64, 4), net.variables[2].shape)
    # Obs Fc Bias
    self.assertEqual((4,), net.variables[3].shape)
    # Action Fc Kernel
    self.assertEqual((2, 5), net.variables[4].shape)
    # Action Fc Bias
    self.assertEqual((5,), net.variables[5].shape)
    # Joint Fc Kernel
    self.assertEqual((9, 5), net.variables[6].shape)
    # Joint Fc Bias
    self.assertEqual((5,), net.variables[7].shape)
    # RNN Cell Kernel
    self.assertEqual((5, 3), net.variables[8].shape)
    # RNN Cell Recurrent Kernel
    self.assertEqual((3, 3), net.variables[9].shape)
    # RNN Cell Bias
    self.assertEqual((3,), net.variables[10].shape)
    # Output Fc Kernel
    self.assertEqual((3, 5), net.variables[11].shape)
    # Output Fc Bias
    self.assertEqual((5,), net.variables[12].shape)
    # Q Value Kernel
    self.assertEqual((5, 1), net.variables[13].shape)
    # Q Value Bias
    self.assertEqual((1,), net.variables[14].shape)

    # Assert LSTM cell is created.
    self.assertEqual((3,), network_state[0].shape)

  @parameterized.named_parameters(
      ('DynamicUnroll', (3,), None),
      ('LSTMKerasUnroll', None, lstm_keras_fn),
  )
  def testBuilds(self, lstm_size, rnn_construction_fn):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))

    action_spec = tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3)
    network_input_spec = (observation_spec, action_spec)
    action = tensor_spec.sample_spec_nest(action_spec, outer_dims=(1,))
    net = critic_rnn_network.CriticRnnNetwork(
        network_input_spec,
        observation_conv_layer_params=[(4, 2, 2)],
        observation_fc_layer_params=(4,),
        action_fc_layer_params=(5,),
        joint_fc_layer_params=(5,),
        lstm_size=lstm_size,
        output_fc_layer_params=(5,),
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs={'lstm_size': 3})

    network_input = (time_step.observation, action)
    q_values, network_state = net(network_input, time_step.step_type,
                                  net.get_initial_state(batch_size=1))

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([1], q_values.shape.as_list())

    self.assertLen(net.variables, 15)
    print('variables %s', net.variables)
    # Obs Conv Net Kernel
    self.assertEqual((2, 2, 3, 4), net.variables[0].shape)
    # Obs Conv Net bias
    self.assertEqual((4,), net.variables[1].shape)
    # Obs Fc Kernel
    self.assertEqual((64, 4), net.variables[2].shape)
    # Obs Fc Bias
    self.assertEqual((4,), net.variables[3].shape)
    # Action Fc Kernel
    self.assertEqual((2, 5), net.variables[4].shape)
    # Action Fc Bias
    self.assertEqual((5,), net.variables[5].shape)
    # Joint Fc Kernel
    self.assertEqual((9, 5), net.variables[6].shape)
    # Joint Fc Bias
    self.assertEqual((5,), net.variables[7].shape)
    # LSTM Cell Kernel
    self.assertEqual((5, 12), net.variables[8].shape)
    # LSTM Cell Recurrent Kernel
    self.assertEqual((3, 12), net.variables[9].shape)
    # LSTM Cell Bias
    self.assertEqual((12,), net.variables[10].shape)
    # Output Fc Kernel
    self.assertEqual((3, 5), net.variables[11].shape)
    # Output Fc Bias
    self.assertEqual((5,), net.variables[12].shape)
    # Q Value Kernel
    self.assertEqual((5, 1), net.variables[13].shape)
    # Q Value Bias
    self.assertEqual((1,), net.variables[14].shape)

    # Assert LSTM cell is created.
    self.assertEqual((1, 3), network_state[0].shape)
    self.assertEqual((1, 3), network_state[1].shape)

  @parameterized.named_parameters(
      ('DynamicUnroll', (3,), None),
      ('LSTMKerasUnroll', None, lstm_keras_fn),
      ('RNNKerasUnroll', None, rnn_keras_fn),
  )
  def testInit(self, lstm_size, rnn_construction_fn):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    action_spec = tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3)
    network_input_spec = (observation_spec, action_spec)
    critic_rnn_network.CriticRnnNetwork(
        network_input_spec,
        lstm_size=lstm_size,
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs={'lstm_size': 3})


if __name__ == '__main__':
  tf.test.main()
