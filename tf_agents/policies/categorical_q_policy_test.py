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

"""Tests for learning.reinforcement_learning.policies.categorical_q_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from tf_agents.networks import categorical_q_network
from tf_agents.networks import network
from tf_agents.policies import categorical_q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DummyCategoricalNet(network.Network):

  def __init__(self,
               input_tensor_spec,
               num_atoms=51,
               num_actions=2,
               name=None):
    self._num_atoms = num_atoms
    self._num_actions = num_actions
    super(DummyCategoricalNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    # In CategoricalDQN we are dealing with a distribution over Q-values, which
    # are represented as num_atoms bins, ranging from min_q_value to
    # max_q_value. In order to replicate the setup in the non-categorical
    # network (namely, [[2, 1], [1, 1]]), we use the following "logits":
    # [[0, 1, ..., num_atoms-1, num_atoms, 1, ..., 1],
    #  [1, ......................................, 1]]
    # The important bit is that the first half of the first list (which
    # corresponds to the logits for the first action) place more weight on the
    # higher q_values than on the lower ones, thereby resulting in a higher
    # value for the first action.
    weights_initializer = np.array([
        np.concatenate((np.arange(num_atoms), np.ones(num_atoms))),
        np.concatenate((np.ones(num_atoms), np.ones(num_atoms)))])
    kernel_initializer = tf.compat.v1.initializers.constant(
        weights_initializer, verify_shape=True)
    bias_initializer = tf.compat.v1.initializers.ones()

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = []
    self._dummy_layers.append(
        tf.keras.layers.Dense(
            num_actions * num_atoms,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer))

  @property
  def num_atoms(self):
    return self._num_atoms

  def call(self, inputs, unused_step_type=None, network_state=()):
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    logits = tf.reshape(inputs, [-1, self._num_actions, self._num_atoms])
    return logits, network_state


class CategoricalQPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(CategoricalQPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)
    self._min_q_value = -10.0
    self._max_q_value = 10.0
    self._q_network = DummyCategoricalNet(
        input_tensor_spec=self._obs_spec,
        num_atoms=3,
        num_actions=2)

  def testBuild(self):
    policy = categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                                     self._max_q_value,
                                                     self._q_network,
                                                     self._action_spec)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

    # There should be two variables in our network for the fc_layer we specified
    # (one kernel and one bias).
    self.assertLen(policy.variables(), 2)

  def testMultipleActionsRaiseError(self):
    with self.assertRaisesRegexp(
        TypeError, '.*action_spec must be a BoundedTensorSpec.*'):
      # Replace the action_spec for this test.
      action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)] * 2
      q_network = categorical_q_network.CategoricalQNetwork(
          input_tensor_spec=self._obs_spec,
          action_spec=action_spec,
          num_atoms=3,
          fc_layer_params=[4])
      categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                              self._max_q_value,
                                              q_network,
                                              action_spec)

  def testAction(self):
    policy = categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                                     self._max_q_value,
                                                     self._q_network,
                                                     self._action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations)
    actions, _, _ = policy.action(time_step)
    self.assertEqual(actions.shape.as_list(), [2])
    self.assertEqual(actions.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(actions)

    # actions should be a list of two elements; e.g., [0, 1]
    self.assertLen(actions, 2)

    for action in actions:
      self.assertGreaterEqual(action, self._action_spec.minimum)
      self.assertLessEqual(action, self._action_spec.maximum)

  def testSample(self):
    policy = categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                                     self._max_q_value,
                                                     self._q_network,
                                                     self._action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations)
    actions, _ = policy.step(time_step)
    self.assertEqual(actions.shape.as_list(), [2])
    self.assertEqual(actions.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(actions)

    # actions should be a list of two elements; e.g., [0, 1]
    self.assertLen(actions, 2)

    for action in actions:
      self.assertGreaterEqual(action, self._action_spec.minimum)
      self.assertLessEqual(action, self._action_spec.maximum)

  def testMultiSample(self):
    policy = categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                                     self._max_q_value,
                                                     self._q_network,
                                                     self._action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations)
    actions, _ = policy.step(time_step, num_samples=2)
    self.assertEqual(actions.shape.as_list(), [2, 2])
    self.assertEqual(actions.dtype, tf.int32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(actions)

    # actions should be a nested list of the form [[0, 1], [1, 0]]
    self.assertLen(actions, 2)

    for inner_list in actions:
      self.assertLen(inner_list, 2)

      for action in inner_list:
        self.assertGreaterEqual(action, self._action_spec.minimum)
        self.assertLessEqual(action, self._action_spec.maximum)

  def testUpdate(self):
    policy = categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                                     self._max_q_value,
                                                     self._q_network,
                                                     self._action_spec)

    new_policy = categorical_q_policy.CategoricalQPolicy(self._min_q_value,
                                                         self._max_q_value,
                                                         self._q_network,
                                                         self._action_spec)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations)

    # There should be two variables in our networks for the fc_layer we
    # specified (one kernel and one bias).
    self.assertLen(policy.variables(), 2)
    self.assertLen(new_policy.variables(), 2)

    actions, _, _ = policy.action(time_step)
    new_actions, _, _ = new_policy.action(time_step)

    self.assertEqual(actions.shape, new_actions.shape)
    self.assertEqual(actions.dtype, new_actions.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(actions)

    # actions should be a list of two elements; e.g., [0, 1]
    self.assertLen(actions, 2)

    for action in actions:
      self.assertGreaterEqual(action, self._action_spec.minimum)
      self.assertLessEqual(action, self._action_spec.maximum)

    self.assertEqual(self.evaluate(new_policy.update(policy)), None)
    new_actions = self.evaluate(new_actions)

    # new_actions should also be a list of two elements; e.g., [0, 1]
    self.assertLen(new_actions, 2)

    for action in new_actions:
      self.assertGreaterEqual(action, self._action_spec.minimum)
      self.assertLessEqual(action, self._action_spec.maximum)


if __name__ == '__main__':
  tf.test.main()
