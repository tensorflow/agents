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

"""Test for tf_agents.policies.fixed_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.policies import fixed_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class FixedPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(FixedPolicyTest, self).setUp()
    # Creates an MDP with:
    # - dim(observation) = 2
    # - number of actions = 4
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._num_actions = 4
    self._action_spec = tensor_spec.BoundedTensorSpec(
        shape=(1,), dtype=tf.int32,
        minimum=0, maximum=self._num_actions - 1)

    # The policy always outputs the same action.
    self._fixed_action = 1
    self._policy = fixed_policy.FixedPolicy(
        np.asarray([self._fixed_action], dtype=np.int32),
        self._time_step_spec,
        self._action_spec)

  def testFixedPolicySingle(self):
    observations = tf.constant([1, 2], dtype=tf.float32)
    time_step = ts.restart(observations)
    action_step = self._policy.action(time_step)
    distribution_step = self._policy.distribution(time_step)
    mode = distribution_step.action.mode()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(self.evaluate(action_step.action),
                        [self._fixed_action])
    self.assertAllEqual(self.evaluate(mode), [self._fixed_action])

  def testFixedPolicyBatched(self):
    batch_size = 2
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=batch_size)
    action_step = self._policy.action(time_step)
    distribution_step = self._policy.distribution(time_step)
    mode = distribution_step.action.mode()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(action_step.action), [[self._fixed_action]] * batch_size)
    self.assertAllEqual(
        self.evaluate(mode), [[self._fixed_action]] * batch_size)

  def testFixedPolicyBatchedOnNestedObservations(self):
    batch_size = 2
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=batch_size)
    action_spec = (tensor_spec.TensorSpec(shape=(2,), dtype=tf.float32),
                   (tensor_spec.TensorSpec(shape=(1,), dtype=tf.int64), {
                       'dict': tensor_spec.TensorSpec(shape=(), dtype=tf.int32)
                   }))
    fixed_action = (np.array([100, 200],
                             dtype=np.float32), (np.array([300],
                                                          dtype=np.int64), {
                                                              'dict': 400
                                                          }))
    policy = fixed_policy.FixedPolicy(fixed_action, self._time_step_spec,
                                      action_spec)
    action = policy.action(time_step).action
    distribution_mode = tf.nest.map_structure(
        lambda t: t.mode(),
        policy.distribution(time_step).action)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    expected = (tf.constant([[100, 200]] * batch_size, dtype=tf.float32),
                (tf.constant([[300]] * batch_size, dtype=tf.int64), {
                    'dict': tf.constant([400] * batch_size, dtype=tf.int32)
                }))
    tf.nest.map_structure(self.assertAllEqual, action, expected)
    tf.nest.map_structure(self.assertAllEqual, distribution_mode, expected)


if __name__ == '__main__':
  tf.test.main()
