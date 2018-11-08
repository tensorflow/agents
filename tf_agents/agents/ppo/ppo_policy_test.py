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

from tf_agents.agents.ppo import ppo_policy
from tf_agents.environments import time_step as ts
from tf_agents.specs import tensor_spec


nest = tf.contrib.framework.nest
slim = tf.contrib.slim


def _dummy_action_net(time_steps, action_spec, network_state):
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

  return nest.pack_sequence_as(action_spec, [action_means]), network_state


def _dummy_action_distribution_net(time_steps, action_spec, network_state):
  action_means, network_state = _dummy_action_net(
      time_steps, action_spec, network_state)
  def _action_distribution(action_mean):
    action_std = tf.ones_like(action_mean)
    return tfp.distributions.Normal(action_mean, action_std)

  return nest.map_structure(_action_distribution, action_means), network_state


def _dummy_value_net(observations, step_types, network_state):
  del step_types
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=None):
    states = tf.to_float(observations)
    states = nest.flatten(states)[0]
    value_pred = slim.fully_connected(
        states,
        1,
        scope='value_pred',
        weights_initializer=tf.constant_initializer([2, 1]),
        biases_initializer=tf.constant_initializer([5]),
        normalizer_fn=None,
        activation_fn=None)
  return value_pred, network_state


def _test_cases(prefix=''):
  return [{
      'testcase_name': '%s0' % prefix,
      'network_fn': _dummy_action_net
  }, {
      'testcase_name': '%s1' % prefix,
      'network_fn': _dummy_action_distribution_net
  }]


class PPOPolicyTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(PPOPolicyTest, self).setUp()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, 2, 3)

  @property
  def _time_step(self):
    return ts.TimeStep(
        step_type=tf.constant([1], dtype=tf.int32),
        reward=tf.constant([1], dtype=tf.float32),
        discount=tf.constant([1], dtype=tf.float32),
        observation=tf.constant([[1, 2]], dtype=tf.float32))

  @property
  def _time_step_batch(self):
    return ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST, dtype=tf.int32, shape=[2], name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[2], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[2], name='discount'),
        tf.constant([[1, 2], [3, 4]], dtype=tf.float32, name='observation'))

  @parameterized.named_parameters(*_test_cases('test_build'))
  def testBuild(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    self.assertEqual(policy.time_step_spec(), self._time_step_spec)
    self.assertEqual(policy.action_spec(), self._action_spec)

  @parameterized.named_parameters(*_test_cases('test_reset'))
  def testReset(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    policy_state = policy.get_initial_state(batch_size=1)

    # Dummy network has no policy_state so expect empty tuple from reset.
    self.assertEqual((), policy_state)

  @parameterized.named_parameters(*_test_cases('test_action'))
  def testAction(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    self.assertEqual(action_step.action.shape.as_list(), [1, 1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  @parameterized.named_parameters(*_test_cases('test_action_list'))
  def testActionList(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    value_network = tf.make_template('value_network', _dummy_value_net)
    action_spec = [self._action_spec]
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, action_spec, actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    self.assertIsInstance(action_step.action, list)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= action_spec[0].minimum))
    self.assertTrue(np.all(actions_ <= action_spec[0].maximum))

  @parameterized.named_parameters(*_test_cases('test_action_batch'))
  def testActionBatch(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step_batch)
    self.assertEqual(action_step.action.shape.as_list(), [2, 1])
    self.assertEqual(action_step.action.dtype, tf.float32)
    self.evaluate(tf.global_variables_initializer())
    actions_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(actions_ >= self._action_spec.minimum))
    self.assertTrue(np.all(actions_ <= self._action_spec.maximum))

  @parameterized.named_parameters(*_test_cases('test_action'))
  def testValue(self, network_fn):
    actor_network = tf.make_template('actor_network', network_fn)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    batch_size = self._time_step.step_type.shape[0].value
    policy_state = policy.get_initial_state(batch_size=batch_size)
    value_pred, unused_policy_state = policy.apply_value_network(
        self._time_step.observation, self._time_step.step_type, policy_state)
    self.assertEqual(value_pred.shape.as_list(), [1, 1])
    self.assertEqual(value_pred.dtype, tf.float32)
    self.evaluate(tf.global_variables_initializer())
    self.evaluate(value_pred)

  def testUpdate(self):
    tf.set_random_seed(1)
    actor_network = tf.make_template('actor_network', _dummy_action_net)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)
    new_policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    action_step = policy.action(self._time_step)
    new_action_step = new_policy.action(self._time_step)

    self.assertEqual(action_step.action.shape, new_action_step.action.shape)
    self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(new_policy.update(policy))
    actions_, new_actions_ = self.evaluate(
        [action_step.action, new_action_step.action])
    self.assertAllEqual(actions_, new_actions_)

  def testDeterministicDistribution(self):
    actor_network = tf.make_template('actor_network', _dummy_action_net)
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

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
    value_network = tf.make_template('value_network', _dummy_value_net)
    policy = ppo_policy.PPOPolicy(
        self._time_step_spec, self._action_spec, actor_network=actor_network,
        value_network=value_network)

    distribution_step = policy.distribution(self._time_step)
    self.assertIsInstance(distribution_step.action, tfp.distributions.Normal)


if __name__ == '__main__':
  tf.test.main()
