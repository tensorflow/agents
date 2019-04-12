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

"""Tests for tf_agents.policies.actor_policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DummyActionNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(DummyActionNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='DummyActionNet')
    single_action_spec = tf.nest.flatten(output_tensor_spec)[0]
    self._output_tensor_spec = output_tensor_spec
    self._layers = [
        tf.keras.layers.Dense(
            single_action_spec.shape.num_elements(),
            activation=tf.nn.tanh,
            kernel_initializer=tf.compat.v1.initializers.constant([2, 1]),
            bias_initializer=tf.compat.v1.initializers.constant([5]),
        ),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    states = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    for layer in self.layers:
      states = layer(states)

    single_action_spec = tf.nest.flatten(self._output_tensor_spec)[0]
    means = tf.reshape(states, [-1] + single_action_spec.shape.as_list())
    spec_means = (single_action_spec.maximum + single_action_spec.minimum) / 2.0
    spec_ranges = (
        single_action_spec.maximum - single_action_spec.minimum) / 2.0
    action_means = spec_means + spec_ranges * means

    return (tf.nest.pack_sequence_as(self._output_tensor_spec, [action_means]),
            network_state)


class DummyActionDistributionNet(DummyActionNet):

  def call(self, observations, step_type, network_state):
    action_means, network_state = super(DummyActionDistributionNet, self).call(
        observations, step_type, network_state)

    def _action_distribution(action_mean):
      action_std = tf.ones_like(action_mean)
      return tfp.distributions.Normal(action_mean, action_std)

    return tf.nest.map_structure(_action_distribution,
                                 action_means), network_state


def test_cases():
  return parameterized.named_parameters({
      'testcase_name': 'SimpleNet',
      'network_ctor': DummyActionNet,
  }, {
      'testcase_name': 'DistributionNet',
      'network_ctor': DummyActionDistributionNet,
  })


class ActorPolicyTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(ActorPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2, 3)

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

  @test_cases()
  def testBuild(self, network_ctor):
    actor_network = network_ctor(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    self.assertEqual(policy.time_step_spec, self._time_step_spec)
    self.assertEqual(policy.action_spec, self._action_spec)
    self.assertLen(policy.variables(), 2)

  @test_cases()
  def testActionBatch(self, network_ctor):
    actor_network = network_ctor(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  def testUpdate(self):
    tf.compat.v1.set_random_seed(1)
    actor_network = DummyActionNet(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)
    self.assertLen(policy.variables(), 2)
    new_policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    self.assertLen(policy.variables(), 2)
    new_action_step = new_policy.action(self._time_step_batch)
    self.assertLen(new_policy.variables(), 2)

    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(new_policy.update(policy))
    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  def testDeterministicDistribution(self):
    actor_network = DummyActionNet(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    distribution_step = policy.distribution(self._time_step_batch)
    self.assertIsInstance(distribution_step.action,
                          tfp.distributions.Deterministic)
    distribution_mean = distribution_step.action.mean()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    distribution_mean_ = self.evaluate(distribution_mean)
    self.assertNear(actions_[0], distribution_mean_[0], 1e-6)

  def testGaussianDistribution(self):
    actor_network = DummyActionDistributionNet(self._obs_spec,
                                               self._action_spec)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    distribution_step = policy.distribution(self._time_step_batch)
    self.assertIsInstance(distribution_step.action, tfp.distributions.Normal)


if __name__ == '__main__':
  tf.test.main()
