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

"""Tests for tf_agents.bandits.policies.mixture_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.bandits.policies import mixture_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

tfd = tfp.distributions


class ConstantPolicy(tf_policy.TFPolicy):
  """A policy that outputs a constant action, for testing purposes."""

  def __init__(self, action_spec, time_step_spec, action):
    self._constant_action = action
    super(ConstantPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec={'a': tensor_spec.TensorSpec(shape=(), dtype=tf.int32)})

  def _variables(self):
    return []

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'This policy outputs an action and not a distribution.')

  def _action(self, time_step, policy_state, seed=None):
    batch_size = tf.compat.dimension_value(tf.shape(time_step.observation)[0])
    return policy_step.PolicyStep(
        tf.fill([batch_size], self._constant_action), policy_state,
        {'a': tf.range(batch_size, dtype=tf.int32)})


class MixturePolicyTest(test_utils.TestCase):

  def testMixturePolicyInconsistentSpecs(self):
    context_dim = 11
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), dtype=tf.int32, minimum=0, maximum=9, name='action')
    sub_policies = [
        ConstantPolicy(action_spec, time_step_spec, i) for i in range(9)
    ]
    wrong_obs_spec = tensor_spec.TensorSpec([context_dim + 1], tf.float32)
    wrong_time_step_spec = ts.time_step_spec(wrong_obs_spec)
    wrong_policy = ConstantPolicy(action_spec, wrong_time_step_spec, 9)
    sub_policies.append(wrong_policy)
    weights = [0, 0, 0.2, 0, 0, -0.3, 0, 0, 0.5, 0]
    dist = tfd.Categorical(probs=weights)
    with self.assertRaisesRegexp(AssertionError,
                                 'Inconsistent time step specs'):
      mixture_policy.MixturePolicy(dist, sub_policies)

  def testMixturePolicyChoices(self):
    context_dim = 34
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), dtype=tf.int32, minimum=0, maximum=9, name='action')
    sub_policies = [
        ConstantPolicy(action_spec, time_step_spec, i) for i in range(10)
    ]
    weights = [0, 0, 0.2, 0, 0, 0.3, 0, 0, 0.5, 0]
    dist = tfd.Categorical(probs=weights)
    policy = mixture_policy.MixturePolicy(dist, sub_policies)
    batch_size = 15
    time_step = ts.TimeStep(
        tf.constant(
            ts.StepType.FIRST,
            dtype=tf.int32,
            shape=[batch_size],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
        tf.constant(
            list(range(batch_size * context_dim)),
            dtype=tf.float32,
            shape=[batch_size, context_dim],
            name='observation'))
    action_step = policy.action(time_step)
    actions, infos = self.evaluate([action_step.action, action_step.info])
    tf.nest.assert_same_structure(policy.info_spec, infos)
    self.assertAllEqual(actions.shape, [batch_size])
    self.assertAllInSet(actions, [2, 5, 8])

  def testMixturePolicyDynamicBatchSize(self):
    context_dim = 35
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), dtype=tf.int32, minimum=0, maximum=9, name='action')
    sub_policies = [
        ConstantPolicy(action_spec, time_step_spec, i) for i in range(10)
    ]
    weights = [0, 0, 0.2, 0, 0, 0.3, 0, 0, 0.5, 0]
    dist = tfd.Categorical(probs=weights)

    policy = mixture_policy.MixturePolicy(dist, sub_policies)
    batch_size = tf.random.uniform(
        shape=(), minval=10, maxval=15, dtype=tf.int32)
    time_step = ts.TimeStep(
        tf.fill(
            tf.expand_dims(batch_size, axis=0),
            ts.StepType.FIRST,
            name='step_type'),
        tf.zeros(shape=[batch_size], dtype=tf.float32, name='reward'),
        tf.ones(shape=[batch_size], dtype=tf.float32, name='discount'),
        tf.reshape(
            tf.range(
                tf.cast(batch_size * context_dim, dtype=tf.float32),
                dtype=tf.float32),
            shape=[-1, context_dim],
            name='observation'))
    action_step = policy.action(time_step)
    actions, bsize = self.evaluate([action_step.action, batch_size])
    self.assertAllEqual(actions.shape, [bsize])
    self.assertAllInSet(actions, [2, 5, 8])

    train_step = tf.compat.v1.train.get_or_create_global_step()
    saver = policy_saver.PolicySaver(policy, train_step=train_step)
    location = os.path.join(self.get_temp_dir(), 'saved_policy')
    if not tf.executing_eagerly():
      with self.cached_session():
        self.evaluate(tf.compat.v1.global_variables_initializer())
        saver.save(location)
    else:
      saver.save(location)
    loaded_policy = tf.compat.v2.saved_model.load(location)
    new_batch_size = 3
    new_time_step = ts.TimeStep(
        tf.fill(
            tf.expand_dims(new_batch_size, axis=0),
            ts.StepType.FIRST,
            name='step_type'),
        tf.zeros(shape=[new_batch_size], dtype=tf.float32, name='reward'),
        tf.ones(shape=[new_batch_size], dtype=tf.float32, name='discount'),
        tf.reshape(
            tf.range(
                tf.cast(new_batch_size * context_dim, dtype=tf.float32),
                dtype=tf.float32),
            shape=[-1, context_dim],
            name='observation'))
    new_action = self.evaluate(loaded_policy.action(new_time_step).action)
    self.assertAllEqual(new_action.shape, [new_batch_size])
    self.assertAllInSet(new_action, [2, 5, 8])


if __name__ == '__main__':
  tf.test.main()
