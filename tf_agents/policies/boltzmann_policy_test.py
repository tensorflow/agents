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

"""Test for tf_agents.policies.boltzmann_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import network
from tf_agents.policies import boltzmann_policy
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
            kernel_initializer=tf.constant_initializer([[1, 1.5], [1, 1.5]]),
            bias_initializer=tf.constant_initializer([[1], [1]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class BoltzmannPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(BoltzmannPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)

  def testBuild(self):
    wrapped = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    policy = boltzmann_policy.BoltzmannPolicy(wrapped, temperature=0.9)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)

  def testAction(self):
    tf.compat.v1.set_random_seed(1)
    wrapped = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    policy = boltzmann_policy.BoltzmannPolicy(wrapped, temperature=0.9)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=2)
    action_step = policy.action(time_step, seed=1)
    self.assertEqual(action_step.action.shape.as_list(), [2])
    self.assertEqual(action_step.action.dtype, tf.int32)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(action_step.action)

  def testDistribution(self):
    tf.compat.v1.set_random_seed(1)
    wrapped = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    policy = boltzmann_policy.BoltzmannPolicy(wrapped, temperature=0.9)

    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=1)
    distribution_step = policy.distribution(time_step)
    mode = distribution_step.action.mode()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # The weights of index 0 are all 1 and the weights of index 1 are all 1.5,
    # so the Q values of index 1 will be higher.
    self.assertAllEqual([1], self.evaluate(mode))

  def testLogits(self):
    tf.compat.v1.set_random_seed(1)
    wrapped = q_policy.QPolicy(
        self._time_step_spec, self._action_spec, q_network=DummyNet())
    policy = boltzmann_policy.BoltzmannPolicy(wrapped, temperature=0.5)

    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=1)
    distribution_step = policy.distribution(time_step)
    logits = distribution_step.action.logits
    original_logits = wrapped.distribution(time_step).action.logits
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # The un-temperature'd logits would be 4 and 5.5, because it is (1 2) . (1
    # 1) + 1 and (1 2) . (1.5 1.5) + 1. The temperature'd logits will be double
    # that.
    self.assertAllEqual([[4., 5.5]], self.evaluate(original_logits))
    self.assertAllEqual([[8., 11.]], self.evaluate(logits))


if __name__ == '__main__':
  tf.test.main()
