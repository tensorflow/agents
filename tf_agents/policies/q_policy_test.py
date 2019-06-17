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
import tensorflow as tf
from tf_agents.networks import network
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DummyNet(network.Network):

  def __init__(self, name=None, num_actions=2):
    super(DummyNet, self).__init__(name, (), 'DummyNet')
    self._layers.append(
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant([[1, 1.5],
                                                                   [1, 1.5]]),
            bias_initializer=tf.compat.v1.initializers.constant([[1], [1]])))

  def call(self, inputs, unused_step_type=None, network_state=()):
    inputs = tf.cast(inputs, tf.float32)
    for layer in self.layers:
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
    self.assertEqual(policy.variables(), [])

  def testMultipleActionsRaiseError(self):
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)] * 2
    with self.assertRaisesRegexp(
        NotImplementedError,
        'action_spec can only contain a single BoundedTensorSpec'):
      q_policy.QPolicy(
          self._time_step_spec, action_spec, q_network=DummyNet())

  def testAction(self):
    tf.compat.v1.set_random_seed(1)
    policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [[1], [1]])

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
    tf.compat.v1.set_random_seed(1)

    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 1)
    policy = q_policy.QPolicy(
        self._time_step_spec, action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    aaction_step = policy.action(time_step, seed=1)
    self.assertEqual(aaction_step.action.shape.as_list(), [2])
    self.assertEqual(aaction_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(aaction_step.action), [1, 1])

  def testActionList(self):
    action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)]
    policy = q_policy.QPolicy(
        self._time_step_spec, action_spec, q_network=DummyNet())
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertTrue(isinstance(action_step.action, list))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action), [[[1], [1]]])

  def testDistribution(self):
    tf.compat.v1.set_random_seed(1)
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
    tf.compat.v1.set_random_seed(1)
    policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    new_policy = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)

    self.assertEqual(policy.variables(), [])
    self.assertEqual(new_policy.variables(), [])

    action_step = policy.action(time_step, seed=1)
    new_action_step = new_policy.action(time_step, seed=1)

    self.assertEqual(len(policy.variables()), 2)
    self.assertEqual(len(new_policy.variables()), 2)
    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(self.evaluate(new_policy.update(policy)), None)

    self.assertAllEqual(self.evaluate(action_step.action), [[1], [1]])
    self.assertAllEqual(self.evaluate(new_action_step.action), [[1], [1]])

  def testActionSpecsCompatible(self):
    q_network = DummyNetWithActionSpec(self._action_spec)
    q_policy.QPolicy(self._time_step_spec, self._action_spec, q_network)

  def testActionSpecsIncompatible(self):
    network_action_spec = tensor_spec.BoundedTensorSpec([2], tf.int32, 0, 1)
    q_network = DummyNetWithActionSpec(network_action_spec)

    with self.assertRaisesRegexp(
        ValueError,
        'action_spec must be compatible with q_network.action_spec'):
      q_policy.QPolicy(self._time_step_spec, self._action_spec, q_network)


if __name__ == '__main__':
  tf.test.main()
