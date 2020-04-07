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

"""Test for tf_agents.policies.q_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DummyNet(network.Network):

  def __init__(self, name=None, num_actions=2):
    super(DummyNet, self).__init__(
        tensor_spec.TensorSpec([2], tf.float32), (), 'DummyNet')

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant([[1, 1.5],
                                                                   [1, 1.5]]),
            bias_initializer=tf.compat.v1.initializers.constant([[1], [1]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class DummyNetWithActionSpec(DummyNet):

  def __init__(self, action_spec, name=None, num_actions=2):
    super(DummyNetWithActionSpec, self).__init__(name, num_actions)
    self._action_spec = action_spec

  @property
  def action_spec(self):
    return self._action_spec


class QPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(QPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)

  def testBuild(self):
    policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  def testMultipleActionsRaiseError(self):
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)] * 2
    with self.assertRaisesRegexp(
        NotImplementedError,
        'action_spec can only contain a single BoundedTensorSpec'):
      q_policy.QPolicy(
          self._time_step_spec, action_spec, q_network=DummyNet())

  def testAction(self):
    policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action = self.evaluate(action_step.action)
    self.assertTrue(np.all(action >= 0) and np.all(action <= 1))

  def testActionWithinBounds(self):
    bounded_action_spec = tensor_spec.BoundedTensorSpec([1],
                                                        tf.int32,
                                                        minimum=-6,
                                                        maximum=-5)
    policy = q_policy.QPolicy(
        self._time_step_spec, bounded_action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action = self.evaluate(action_step.action)
    self.assertTrue(np.all(action <= -5) and np.all(action >= -6))

  def testActionScalarSpec(self):
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 1)
    policy = q_policy.QPolicy(
        self._time_step_spec, action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action = self.evaluate(action_step.action)
    self.assertTrue(np.all(action >= 0) and np.all(action <= 1))

  def testActionList(self):
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)]
    policy = q_policy.QPolicy(
        self._time_step_spec, action_spec, q_network=DummyNet())
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertIsInstance(action_step.action, list)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action = self.evaluate(action_step.action)
    self.assertLen(action, 1)
    # Extract contents from the outer list.
    action = action[0]
    self.assertTrue(np.all(action >= 0) and np.all(action <= 1))

  def testDistribution(self):
    policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=1)
    distribution_step = policy.distribution(time_step)
    mode = distribution_step.action.mode()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # The weights of index 0 are all 1 and the weights of index 1 are all 1.5,
    # so the Q values of index 1 will be higher.
    self.assertAllEqual([[1]], self.evaluate(mode))

  def testUpdate(self):
    policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    new_policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    self.assertEqual(len(policy.variables()), 2)
    self.assertEqual(len(new_policy.variables()), 2)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(self.evaluate(new_policy.update(policy)), None)

    distribution = policy.distribution(time_step).action.parameters
    new_distribution = new_policy.distribution(time_step).action.parameters
    self.assertAllEqual(
        self.evaluate(distribution['logits']),
        self.evaluate(new_distribution['logits']))

  def testActionSpecsCompatible(self):
    q_net = DummyNetWithActionSpec(self._action_spec)
    q_policy.QPolicy(self._time_step_spec, self._action_spec, q_net)

  def testActionSpecsIncompatible(self):
    network_action_spec = tensor_spec.BoundedTensorSpec([2], tf.int32, 0, 1)
    q_net = DummyNetWithActionSpec(network_action_spec)

    with self.assertRaisesRegexp(
        ValueError,
        'action_spec must be compatible with q_network.action_spec'):
      q_policy.QPolicy(self._time_step_spec, self._action_spec, q_net)

  def testMasking(self):
    batch_size = 1000
    num_state_dims = 5
    num_actions = 8
    observations = tf.random.uniform([batch_size, num_state_dims])
    time_step = ts.restart(observations, batch_size=batch_size)
    input_tensor_spec = tensor_spec.TensorSpec([num_state_dims], tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec(
        [1], tf.int32, 0, num_actions - 1)

    # We create a fixed mask here for testing purposes. Normally the mask would
    # be part of the observation.
    mask = [0, 1, 0, 1, 0, 0, 1, 0]
    np_mask = np.array(mask)
    tf_mask = tf.constant([mask for _ in range(batch_size)])
    q_net = q_network.QNetwork(input_tensor_spec, action_spec)
    policy = q_policy.QPolicy(
        ts.time_step_spec(input_tensor_spec), action_spec, q_net,
        observation_and_action_constraint_splitter=(
            lambda observation: (observation, tf_mask)))

    # Force creation of variables before global_variables_initializer.
    policy.variables()
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Sample from the policy 1000 times, and ensure that actions considered
    # invalid according to the mask are never chosen.
    action_step = policy.action(time_step)
    action = self.evaluate(action_step.action)
    self.assertEqual(action.shape, (batch_size, 1))
    self.assertAllEqual(np_mask[action], np.ones([batch_size, 1]))


if __name__ == '__main__':
  tf.test.main()
