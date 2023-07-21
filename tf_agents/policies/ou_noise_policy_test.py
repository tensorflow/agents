# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for third_party.py.tf_agents.policies.ou_noise_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import network

from tf_agents.policies import actor_policy
from tf_agents.policies import ou_noise_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DummyActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(DummyActionNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='DummyActionNet')
    self._output_tensor_spec = output_tensor_spec
    single_action_spec = tf.nest.flatten(output_tensor_spec)[0]
    self._sub_layers = [
        tf.keras.layers.Dense(
            single_action_spec.shape.num_elements(),
            activation=tf.nn.tanh,
            kernel_initializer=tf.constant_initializer([2, 1]),
            bias_initializer=tf.constant_initializer([5]),
        ),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    states = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    for layer in self._sub_layers:
      states = layer(states)

    single_action_spec = tf.nest.flatten(self._output_tensor_spec)[0]
    means = tf.reshape(states, [-1] + single_action_spec.shape.as_list())
    spec_means = (single_action_spec.maximum + single_action_spec.minimum) / 2.0
    spec_ranges = (
        single_action_spec.maximum - single_action_spec.minimum) / 2.0
    action_means = spec_means + spec_ranges * means

    return (tf.nest.pack_sequence_as(self._output_tensor_spec, [action_means]),
            network_state)


class OuNoisePolicyTest(test_utils.TestCase):

  def setUp(self):
    super(OuNoisePolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2, 3)
    actor_network = DummyActionNet(self._obs_spec, self._action_spec)
    self._wrapped_policy = actor_policy.ActorPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=self._action_spec,
        actor_network=actor_network,
        clip=False)

  @property
  def _time_step(self):
    return ts.restart(tf.constant([1, 2], dtype=tf.float32))

  @property
  def _time_step_batch(self):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, dtype=tf.int32, shape=[2], name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[2], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[2], name='discount'),
        tf.constant([[1, 2], [3, 4]], dtype=tf.float32, name='observation'))

  def testBuild(self):
    policy = ou_noise_policy.OUNoisePolicy(self._wrapped_policy)
    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)
    self.assertEqual(len(policy.variables()), 2)

  def testActionIsInRange(self):
    policy = ou_noise_policy.OUNoisePolicy(self._wrapped_policy)
    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  def testActionAddsOUNoise(self):
    policy = ou_noise_policy.OUNoisePolicy(self._wrapped_policy, clip=False)
    action_step = policy.action(self._time_step_batch)
    wrapped_action_step = self._wrapped_policy.action(self._time_step_batch)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    wrapped_policy_actions_ = self.evaluate(wrapped_action_step.action)

    self.assertTrue(np.linalg.norm(actions_ - wrapped_policy_actions_) > 0)

  def testActionList(self):
    action_spec = [self._action_spec]
    actor_network = DummyActionNet(self._obs_spec, action_spec)
    self._wrapped_policy = actor_policy.ActorPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        actor_network=actor_network,
        clip=False)

    policy = ou_noise_policy.OUNoisePolicy(self._wrapped_policy)
    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action[0].shape.as_list(), [2, 1])
    self.assertEqual(action_step.action[0].dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_[0] >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_[0] <= self._action_spec.maximum))


if __name__ == '__main__':
  tf.test.main()
