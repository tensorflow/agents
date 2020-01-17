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

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import expand_dims_layer
from tf_agents.networks import q_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step


class QRnnNetworkTest(tf.test.TestCase):

  def test_network_builds(self):
    env = suite_gym.load('CartPole-v0')
    tf_env = tf_py_environment.TFPyEnvironment(env)
    rnn_network = q_rnn_network.QRnnNetwork(tf_env.observation_spec(),
                                            tf_env.action_spec())

    first_time_step = tf_env.current_time_step()
    q_values, state = rnn_network(
        first_time_step.observation, first_time_step.step_type,
        network_state=rnn_network.get_initial_state(batch_size=1)
    )
    self.assertEqual((1, 2), q_values.shape)
    self.assertEqual((1, 40), state[0].shape)
    self.assertEqual((1, 40), state[1].shape)

  def test_network_can_preprocess_and_combine(self):
    batch_size = 3
    frames = 5
    num_actions = 2
    lstm_size = 6
    states = (tf.random.uniform([batch_size, frames, 1]),
              tf.random.uniform([batch_size, frames]))
    preprocessing_layers = (
        tf.keras.layers.Dense(4),
        tf.keras.Sequential([
            expand_dims_layer.ExpandDims(-1),  # Convert to vec size (1,).
            tf.keras.layers.Dense(4)]))
    network = q_rnn_network.QRnnNetwork(
        input_tensor_spec=(
            tensor_spec.TensorSpec([1], tf.float32),
            tensor_spec.TensorSpec([], tf.float32)),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add(),
        lstm_size=(lstm_size,),
        action_spec=tensor_spec.BoundedTensorSpec(
            [1], tf.int32, 0, num_actions - 1))
    empty_step_type = tf.constant(
        [[time_step.StepType.FIRST] * frames] * batch_size)
    q_values, _ = network(states, empty_step_type,
                          network_state=network.get_initial_state(batch_size))
    self.assertAllEqual(
        q_values.shape.as_list(), [batch_size, frames, num_actions])
    # At least 2 variables each for the preprocessing layers.
    self.assertGreater(len(network.trainable_variables), 4)

  def test_network_can_preprocess_and_combine_no_time_dim(self):
    batch_size = 3
    num_actions = 2
    lstm_size = 5
    states = (tf.random.uniform([batch_size, 1]),
              tf.random.uniform([batch_size]))
    preprocessing_layers = (
        tf.keras.layers.Dense(4),
        tf.keras.Sequential([
            expand_dims_layer.ExpandDims(-1),  # Convert to vec size (1,).
            tf.keras.layers.Dense(4)]))
    network = q_rnn_network.QRnnNetwork(
        input_tensor_spec=(
            tensor_spec.TensorSpec([1], tf.float32),
            tensor_spec.TensorSpec([], tf.float32)),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add(),
        lstm_size=(lstm_size,),
        action_spec=tensor_spec.BoundedTensorSpec(
            [1], tf.int32, 0, num_actions - 1))
    empty_step_type = tf.constant([time_step.StepType.FIRST] * batch_size)
    q_values, _ = network(
        states, empty_step_type,
        network_state=network.get_initial_state(batch_size=batch_size))

    # Processed 1 time step and the time axis was squeezed back.
    self.assertAllEqual(
        q_values.shape.as_list(), [batch_size, num_actions])

    # At least 2 variables each for the preprocessing layers.
    self.assertGreater(len(network.trainable_variables), 4)

  def test_network_builds_stacked_cells(self):
    env = suite_gym.load('CartPole-v0')
    tf_env = tf_py_environment.TFPyEnvironment(env)
    rnn_network = q_rnn_network.QRnnNetwork(
        tf_env.observation_spec(), tf_env.action_spec(), lstm_size=(10, 5))

    first_time_step = tf_env.current_time_step()
    q_values, state = rnn_network(
        first_time_step.observation, first_time_step.step_type,
        network_state=rnn_network.get_initial_state(batch_size=1)
    )
    tf.nest.assert_same_structure(rnn_network.state_spec, state)
    self.assertEqual(2, len(state))

    self.assertEqual((1, 2), q_values.shape)
    self.assertEqual((1, 10), state[0][0].shape)
    self.assertEqual((1, 10), state[0][1].shape)
    self.assertEqual((1, 5), state[1][0].shape)
    self.assertEqual((1, 5), state[1][1].shape)


if __name__ == '__main__':
  tf.test.main()
