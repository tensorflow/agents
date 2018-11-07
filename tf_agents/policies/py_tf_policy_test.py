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

"""Test for tf_agents.utils.py_tf_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.environments import time_step as ts
from tf_agents.networks import network
from tf_agents.policies import py_tf_policy
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec

nest = tf.contrib.framework.nest


class DummyNet(network.Network):

  def __init__(self, name=None, num_actions=2):
    state_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.float32)
    super(DummyNet, self).__init__(name, None, state_spec, None)
    self._layers.append(
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer([[1, 2], [3, 4]],
                                                       verify_shape=True),
            bias_initializer=tf.constant_initializer([1, 1],
                                                     verify_shape=True)))

  def call(self, inputs, unused_step_type=None, network_state=()):
    inputs = tf.cast(inputs, tf.float32)
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs, network_state


# TODO(damienv): This function should belong to nest_utils
def fast_map_structure(func, *structure):
  flat_structure = [nest.flatten(s) for s in structure]
  entries = zip(*flat_structure)

  return nest.pack_sequence_as(structure[0], [func(*x) for x in entries])


class PyTFPolicyTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PyTFPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32, 'obs')
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1,
                                                      'action')
    self._tf_policy = q_policy.QPolicy(
        self._time_step_spec,
        self._action_spec,
        q_network=DummyNet())

  def testBuild(self):
    policy = py_tf_policy.PyTFPolicy(self._tf_policy)
    expected_time_step_spec = ts.time_step_spec(
        tensor_spec.to_nest_array_spec(self._obs_spec))
    expected_action_spec = tensor_spec.to_nest_array_spec(self._action_spec)
    self.assertEqual(expected_time_step_spec, policy.time_step_spec())
    self.assertEqual(expected_action_spec, policy.action_spec())

  def testRaiseValueErrorWithoutSession(self):
    policy = py_tf_policy.PyTFPolicy(self._tf_policy)
    with self.assertRaisesRegexp(
        AttributeError,
        "No TensorFlow session-like object was set on this 'PyTFPolicy'.*"):
      policy.get_initial_state()

  @parameterized.parameters([{'batch_size': None}, {'batch_size': 5}])
  def testAssignSession(self, batch_size):
    policy = py_tf_policy.PyTFPolicy(self._tf_policy, batch_size=batch_size)
    policy.session = tf.Session()
    expected_initial_state = np.zeros([batch_size or 1, 1], dtype=np.float32)
    self.assertTrue(
        np.array_equal(
            policy.get_initial_state(batch_size), expected_initial_state))

  @parameterized.parameters([{'batch_size': None}, {'batch_size': 5}])
  def testZeroState(self, batch_size):
    policy = py_tf_policy.PyTFPolicy(self._tf_policy, batch_size=batch_size)
    expected_initial_state = np.zeros([batch_size or 1, 1], dtype=np.float32)
    with self.test_session():
      self.assertTrue(
          np.array_equal(
              policy.get_initial_state(batch_size), expected_initial_state))

  @parameterized.parameters([{'batch_size': None}, {'batch_size': 5}])
  def testAction(self, batch_size):
    single_observation = np.array([1, 2], dtype=np.float32)
    time_steps = ts.restart(single_observation)
    if batch_size is not None:
      time_steps = [time_steps] * batch_size
      time_steps = fast_map_structure(lambda *arrays: np.stack(arrays),
                                      *time_steps)
    policy = py_tf_policy.PyTFPolicy(self._tf_policy, batch_size=batch_size)

    with self.test_session():
      policy_state = policy.get_initial_state(batch_size)
      tf.global_variables_initializer().run()
      action_steps = policy.action(time_steps, policy_state)
      self.assertEqual(action_steps.action.dtype, np.int32)
      if batch_size is None:
        self.assertEqual(action_steps.action.shape, ())
        self.assertEqual(action_steps.action, 1)
        self.assertEqual(action_steps.state, np.zeros([1, 1]))
      else:
        self.assertEqual(action_steps.action.shape, (batch_size,))
        self.assertAllEqual(action_steps.action, [1] * batch_size)
        self.assertAllEqual(action_steps.state, np.zeros([5, 1]))


if __name__ == '__main__':
  tf.test.main()
