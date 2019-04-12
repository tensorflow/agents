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
import tensorflow as tf
from tf_agents.policies import fixed_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class DistributionPolicy(tf_policy.Base):
  """A policy which always returns the configured distribution."""

  def __init__(self, distribution, time_step_spec, action_spec, name=None):
    self._distribution_value = distribution
    super(DistributionPolicy, self).__init__(
        time_step_spec, action_spec, name=name)

  def _distribution(self, time_step, policy_state):
    return policy_step.PolicyStep(self._distribution_value, policy_state)


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
    self.assertAllEqual(self.evaluate(action_step.action),
                        [[self._fixed_action]] * batch_size)
    self.assertAllEqual(self.evaluate(mode),
                        [[self._fixed_action]] * batch_size)


if __name__ == '__main__':
  tf.test.main()
