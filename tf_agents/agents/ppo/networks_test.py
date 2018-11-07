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

"""Tests for reinforcement_learning.agents.ppo.networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.agents.ppo import networks
from tf_agents.environments import time_step as ts
from tf_agents.specs import tensor_spec
from tensorflow.python.framework import test_util  # TF internal


class ActorTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 3)

  @test_util.run_in_graph_and_eager_modes()
  def testAddConvLayers(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, 3, 3, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [3, 3, num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network,
                                     conv_layers=[(16, 3, 2)])
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 5)

  @test_util.run_in_graph_and_eager_modes()
  def testAddFCLayers(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network,
                                     fc_layers=[100])
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 5)

  @test_util.run_in_graph_and_eager_modes()
  def testScalarAction(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network', networks.actor_network)
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 3)

  @test_util.run_in_graph_and_eager_modes()
  def test2DAction(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = tensor_spec.BoundedTensorSpec([2, 3], tf.float32, 2., 3.)
    actor_network = tf.make_template('actor_network', networks.actor_network)
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec.shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 3)

  @test_util.run_in_graph_and_eager_modes()
  def testListOfSingleAction(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)]
    actor_network = tf.make_template('actor_network', networks.actor_network)
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    distribution = distribution[0]
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec[0].shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec[0].shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 3)

  @test_util.run_in_graph_and_eager_modes()
  def testDictOfSingleAction(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = {
        'motor': tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    }
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    distribution, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    distribution = distribution['motor']
    actions, stdevs = distribution.loc, distribution.scale
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec['motor'].shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec['motor'].shape.as_list())
    self.assertEqual(len(actor_network.trainable_variables), 3)

  @test_util.run_in_graph_and_eager_modes()
  def testMultipleActions(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    action_spec = [tensor_spec.BoundedTensorSpec([7], tf.float32, 2., 3.),
                   tensor_spec.BoundedTensorSpec([9], tf.int32, 0, 1)]
    actor_network = tf.make_template('actor_network',
                                     networks.actor_network)
    distributions, unused_network_state = actor_network(
        time_steps, action_spec, network_state=(),
        time_step_spec=time_step_spec)
    actions, stdevs = distributions[0].loc, distributions[0].scale
    logits = distributions[1].logits
    self.assertAllEqual(actions.shape.as_list(),
                        [batch_size] + action_spec[0].shape.as_list())
    self.assertAllEqual(stdevs.shape.as_list(),
                        [batch_size] + action_spec[0].shape.as_list())
    self.assertAllEqual(logits.shape.as_list(),
                        [batch_size] + action_spec[1].shape.as_list() + [2])
    self.assertEqual(len(actor_network.trainable_variables), 5)


class ValueNetworkTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuild(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    value_network = tf.make_template('value_network', networks.value_network,)
    value_preds, unused_network_state = value_network(
        time_steps.observation, time_steps.step_type,
        observation_spec=time_step_spec.observation, network_state=())
    self.assertAllEqual(value_preds.shape.as_list(), [batch_size])
    self.assertEqual(len(value_network.trainable_variables), 2)

  @test_util.run_in_graph_and_eager_modes()
  def testAddConvLayers(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, 3, 3, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [3, 3, num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    value_network = tf.make_template(
        'value_network',
        networks.value_network,
        conv_layers=[(16, 3, 2)])
    value_preds, unused_network_state = value_network(
        time_steps.observation, time_steps.step_type,
        observation_spec=time_step_spec.observation, network_state=())
    self.assertAllEqual(value_preds.shape.as_list(), [batch_size])
    self.assertEqual(len(value_network.trainable_variables), 4)

  @test_util.run_in_graph_and_eager_modes()
  def testAddFCLayers(self):
    batch_size = 4
    num_obs_dims = 5
    obs = tf.random_uniform([batch_size, num_obs_dims])
    obs_spec = tensor_spec.BoundedTensorSpec(
        [num_obs_dims], tf.float32, -100, 100)
    time_step_spec = ts.time_step_spec(obs_spec)
    time_steps = ts.restart(obs, batch_size)
    value_network = tf.make_template(
        'value_network', networks.value_network, fc_layers=[20, 10])
    value_preds, unused_network_state = value_network(
        time_steps.observation, time_steps.step_type,
        observation_spec=time_step_spec.observation, network_state=())
    self.assertAllEqual(value_preds.shape.as_list(), [batch_size])
    self.assertEqual(len(value_network.trainable_variables), 6)


if __name__ == '__main__':
  tf.test.main()
