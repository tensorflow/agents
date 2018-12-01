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

"""Tests for tf_agents.networks.q_rnn_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_rnn_network

nest = tf.contrib.framework.nest


class QRnnNetworkTest(tf.test.TestCase):

  def test_network_builds(self):
    env = suite_gym.load('CartPole-v0')
    tf_env = tf_py_environment.TFPyEnvironment(env)
    rnn_network = q_rnn_network.QRnnNetwork(tf_env.observation_spec(),
                                            tf_env.action_spec())

    time_step = tf_env.current_time_step()
    q_values, state = rnn_network(time_step.observation, time_step.step_type)
    self.assertEqual((1, 2), q_values.shape)
    self.assertEqual((1, 40), state[0].shape)
    self.assertEqual((1, 40), state[1].shape)

  def test_network_builds_stacked_cells(self):
    env = suite_gym.load('CartPole-v0')
    tf_env = tf_py_environment.TFPyEnvironment(env)
    rnn_network = q_rnn_network.QRnnNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), lstm_size=(10, 5))

    time_step = tf_env.current_time_step()
    q_values, state = rnn_network(time_step.observation, time_step.step_type)
    self.assertTrue(isinstance(state, tuple))
    self.assertEqual(2, len(state))

    self.assertEqual((1, 2), q_values.shape)
    self.assertEqual((1, 10), state[0][0].shape)
    self.assertEqual((1, 10), state[0][1].shape)
    self.assertEqual((1, 5), state[1][0].shape)
    self.assertEqual((1, 5), state[1][1].shape)


if __name__ == '__main__':
  tf.test.main()
