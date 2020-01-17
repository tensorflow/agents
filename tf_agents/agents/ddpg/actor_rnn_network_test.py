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

"""Tests for tf_agents.agents.ddpg.actor_rnn_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tensorflow.python.framework import test_util  # TF internal


class ActorRnnNetworkTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuilds(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))

    action_spec = [
        tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3),
        tensor_spec.BoundedTensorSpec((3,), tf.float32, 0, 3)
    ]
    net = actor_rnn_network.ActorRnnNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(4, 2, 2)],
        input_fc_layer_params=(5,),
        lstm_size=(3,),
        output_fc_layer_params=(5,))

    actions, network_state = net(time_step.observation, time_step.step_type,
                                 net.get_initial_state(batch_size=1))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([1, 2], actions[0].shape.as_list())
    self.assertEqual([1, 3], actions[1].shape.as_list())

    self.assertEqual(13, len(net.variables))
    # Conv Net Kernel
    self.assertEqual((2, 2, 3, 4), net.variables[0].shape)
    # Conv Net bias
    self.assertEqual((4,), net.variables[1].shape)
    # Fc Kernel
    self.assertEqual((64, 5), net.variables[2].shape)
    # Fc Bias
    self.assertEqual((5,), net.variables[3].shape)
    # LSTM Cell Kernel
    self.assertEqual((5, 12), net.variables[4].shape)
    # LSTM Cell Recurrent Kernel
    self.assertEqual((3, 12), net.variables[5].shape)
    # LSTM Cell Bias
    self.assertEqual((12,), net.variables[6].shape)
    # Fc Kernel
    self.assertEqual((3, 5), net.variables[7].shape)
    # Fc Bias
    self.assertEqual((5,), net.variables[8].shape)
    # Action 1 Kernel
    self.assertEqual((5, 2), net.variables[9].shape)
    # Action 1 Bias
    self.assertEqual((2,), net.variables[10].shape)
    # Action 2 Kernel
    self.assertEqual((5, 3), net.variables[11].shape)
    # Action 2 Bias
    self.assertEqual((3,), net.variables[12].shape)

    # Assert LSTM cell is created.
    self.assertEqual((1, 3), network_state[0].shape)
    self.assertEqual((1, 3), network_state[1].shape)

  @test_util.run_in_graph_and_eager_modes()
  def testActionsWithinRange(self):
    observation_spec = tensor_spec.BoundedTensorSpec(
        (8, 8, 3), tf.float32, 0, 1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))

    action_spec = [
        tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3),
        tensor_spec.BoundedTensorSpec((3,), tf.float32, 0, 3)
    ]
    net = actor_rnn_network.ActorRnnNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(4, 2, 2)],
        input_fc_layer_params=(5,),
        output_fc_layer_params=(5,),
        lstm_size=(3,))

    actions, _ = net(time_step.observation, time_step.step_type,
                     net.get_initial_state(batch_size=1))
    self.evaluate(tf.compat.v1.global_variables_initializer())

    for (action, spec) in zip(actions, action_spec):
      action_ = self.evaluate(action)
      self.assertTrue(np.all(action_ >= spec.minimum))
      self.assertTrue(np.all(action_ <= spec.maximum))


if __name__ == '__main__':
  tf.test.main()
