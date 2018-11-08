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

from tf_agents.environments import time_step as ts
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.specs import tensor_spec


nest = tf.contrib.framework.nest
slim = tf.contrib.slim


class DummyActionNet(network.Network):

  def __init__(self, observation_spec, action_spec):
    super(DummyActionNet, self).__init__(
        observation_spec=observation_spec,
        action_spec=action_spec,
        state_spec=(),
        name='DummyActionNet')
    single_action_spec = nest.flatten(action_spec)[0]
    self._layers = [
        tf.keras.layers.Dense(
            single_action_spec.shape.num_elements(),
            activation=tf.nn.tanh,
            kernel_initializer=tf.constant_initializer([2, 1]),
            bias_initializer=tf.constant_initializer([5]),
        ),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    states = tf.to_float(nest.flatten(observations)[0])
    for layer in self.layers:
      states = layer(states)

    single_action_spec = nest.flatten(self._action_spec)[0]
    means = tf.reshape(states, [-1] + single_action_spec.shape.as_list())
    spec_means = (single_action_spec.maximum + single_action_spec.minimum) / 2.0
    spec_ranges = (
        single_action_spec.maximum - single_action_spec.minimum) / 2.0
    action_means = spec_means + spec_ranges * means

    return (nest.pack_sequence_as(self._action_spec, [action_means]),
            network_state)


class DummyActionDistributionNet(DummyActionNet):

  def call(self, observations, step_type, network_state):
    action_means, network_state = super(DummyActionDistributionNet, self).call(
        observations, step_type, network_state)

    def _action_distribution(action_mean):
      action_std = tf.ones_like(action_mean)
      return tfp.distributions.Normal(action_mean, action_std)

    return nest.map_structure(_action_distribution, action_means), network_state


def test_cases():
  return parameterized.named_parameters({
      'testcase_name': 'SimpleNet',
      'network_ctor': DummyActionNet,
  }, {
      'testcase_name': 'DistributionNet',
      'network_ctor': DummyActionDistributionNet,
  })


class ActorPolicyKerasTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ActorPolicyKerasTest, self).setUp()
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
    policy = actor_policy.ActorPolicyKeras(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    self.assertEqual(policy.time_step_spec(), self._time_step_spec)
    self.assertEqual(policy.action_spec(), self._action_spec)
    self.assertEqual(policy.variables(), [])

  @test_cases()
  def testActionBatch(self, network_ctor):
    actor_network = network_ctor(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicyKeras(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  def testUpdate(self):
    tf.set_random_seed(1)
    actor_network = DummyActionNet(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicyKeras(
        self._time_step_spec, self._action_spec, actor_network=actor_network)
    self.assertEqual(policy.variables(), [])
    new_policy = actor_policy.ActorPolicyKeras(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    self.assertEqual(len(policy.variables()), 2)
    new_action_step = new_policy.action(self._time_step_batch)
    self.assertEqual(len(new_policy.variables()), 2)

    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(new_policy.update(policy))
    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  def testDeterministicDistribution(self):
    actor_network = DummyActionNet(self._obs_spec, self._action_spec)
    policy = actor_policy.ActorPolicyKeras(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    distribution_step = policy.distribution(self._time_step_batch)
    self.assertIsInstance(distribution_step.action,
                          tfp.distributions.Deterministic)
    distribution_mean = distribution_step.action.mean()
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    distribution_mean_ = self.evaluate(distribution_mean)
    self.assertNear(actions_[0], distribution_mean_[0], 1e-6)

  def testGaussianDistribution(self):
    actor_network = DummyActionDistributionNet(self._obs_spec,
                                               self._action_spec)
    policy = actor_policy.ActorPolicyKeras(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    distribution_step = policy.distribution(self._time_step_batch)
    self.assertIsInstance(distribution_step.action, tfp.distributions.Normal)


def _dummy_action_net(time_steps, action_spec):
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=None):

    single_action_spec = nest.flatten(action_spec)[0]
    states = tf.cast(time_steps.observation, tf.float32)

    means = slim.fully_connected(
        states,
        single_action_spec.shape.num_elements(),
        scope='actions',
        weights_initializer=tf.constant_initializer([2, 1]),
        biases_initializer=tf.constant_initializer([5]),
        normalizer_fn=None,
        activation_fn=tf.nn.tanh)
    means = tf.reshape(means, [-1] + single_action_spec.shape.as_list())
    spec_means = (
        single_action_spec.maximum + single_action_spec.minimum) / 2.0
    spec_ranges = (
        single_action_spec.maximum - single_action_spec.minimum) / 2.0
    action_means = spec_means + spec_ranges * means

  return nest.pack_sequence_as(action_spec, [action_means])


def _dummy_action_distribution_net(time_steps, action_spec):
  action_means = _dummy_action_net(time_steps, action_spec)
  def _action_distribution(action_mean):
    action_std = tf.ones_like(action_mean)
    return tfp.distributions.Normal(action_mean, action_std)

  return nest.map_structure(_action_distribution, action_means)


def _test_cases_template(prefix=''):
  return [{
      'testcase_name': '%s0' % prefix,
      'network_fn': _dummy_action_net,
  },
          {
              'testcase_name': '%s1' % prefix,
              'network_fn': _dummy_action_distribution_net,
          }]


class ActorPolicyOldTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ActorPolicyOldTest, self).setUp()
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

  @parameterized.named_parameters(*_test_cases_template('test_build'))
  def testBuild(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    self.assertEqual(policy.time_step_spec(), self._time_step_spec)
    self.assertEqual(policy.action_spec(), self._action_spec)
    self.assertEqual(policy.variables(), [])

  @parameterized.named_parameters(*_test_cases_template('test_action'))
  def testAction(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step)
    self.assertEqual(action_step.action.shape.as_list(), [1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  @parameterized.named_parameters(*_test_cases_template('test_action_list'))
  def testActionList(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    action_spec = [self._action_spec]
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step)
    self.assertIsInstance(action_step.action, list)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= action_spec[0].minimum))
    self.assertTrue(np.all(actions_ <= action_spec[0].maximum))

  @parameterized.named_parameters(*_test_cases_template('test_action_batch'))
  def testActionBatch(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  def testUpdate(self):
    tf.set_random_seed(1)
    actor_network = tf.make_template('actor_network', _dummy_action_net)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)
    self.assertEqual(policy.variables(), [])
    new_policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step)
    self.assertEqual(len(policy.variables()), 2)
    new_action_step = new_policy.action(self._time_step)
    self.assertEqual(len(new_policy.variables()), 2)

    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(new_policy.update(policy))
    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  def testDeterministicDistribution(self):
    actor_network = tf.make_template('actor_network', _dummy_action_net)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    action_step = policy.action(self._time_step)
    distribution_step = policy.distribution(self._time_step)
    self.assertIsInstance(distribution_step.action,
                          tfp.distributions.Deterministic)
    distribution_mean = distribution_step.action.mean()
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    distribution_mean_ = self.evaluate(distribution_mean)
    self.assertNear(actions_, distribution_mean_, 1e-6)

  def testGaussianDistribution(self):
    actor_network = tf.make_template('actor_network',
                                     _dummy_action_distribution_net)
    policy = actor_policy.ActorPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network)

    distribution_step = policy.distribution(self._time_step)
    self.assertIsInstance(distribution_step.action, tfp.distributions.Normal)


if __name__ == '__main__':
  tf.test.main()
