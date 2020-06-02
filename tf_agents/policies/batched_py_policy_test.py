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

# Lint as: python3
"""Tests for tf_agents.policies.batched_py_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import batched_py_policy
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step as ps
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

MT_PARAMETERS = ({'multithreading': False},
                 {'multithreading': True})
NP_PARAMETERS = ({'multithreading': False, 'num_policies': 1},
                 {'multithreading': True, 'num_policies': 1},
                 {'multithreading': False, 'num_policies': 5},
                 {'multithreading': True, 'num_policies': 5},
                )


class MockPyPolicy(py_policy.PyPolicy):

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedArraySpec,
               policy_state_spec: types.NestedArraySpec = (),
               info_spec: types.NestedArraySpec = ()):
    seed = 987654321
    self._rng = np.random.RandomState(seed)
    super(MockPyPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec)

  def _action(self,
              time_step: ts.TimeStep,
              policy_state: types.NestedArray) -> ps.PolicyStep:
    random_action = array_spec.sample_spec_nest(
        self._action_spec, self._rng)

    return ps.PolicyStep(random_action, policy_state)


class BatchedPyPolicyTest(tf.test.TestCase, parameterized.TestCase):

  @property
  def time_step_spec(self):
    return ts.time_step_spec(
        observation_spec=array_spec.ArraySpec((1,), np.int32))

  @property
  def action_spec(self):
    return array_spec.BoundedArraySpec(
        [7], dtype=np.float32, minimum=-1.0, maximum=1.0)

  @property
  def policy_state_spec(self):
    return array_spec.BoundedArraySpec(
        [3], dtype=np.int16, minimum=-7.0, maximum=7.0)

  def _make_batched_py_policy(self, multithreading, num_policies=3):
    policies = []
    for _ in range(num_policies):
      policies.append(MockPyPolicy(self.time_step_spec,
                                   self.action_spec,
                                   self.policy_state_spec))
    return batched_py_policy.BatchedPyPolicy(
        policies=policies, multithreading=multithreading)

  @parameterized.parameters(*MT_PARAMETERS)
  def test_close_no_hang_after_init(self, multithreading):
    self._make_batched_py_policy(multithreading)

  @parameterized.parameters(*MT_PARAMETERS)
  def test_get_specs(self, multithreading):
    policy = self._make_batched_py_policy(multithreading)
    self.assertEqual(self.time_step_spec, policy.time_step_spec)
    self.assertEqual(self.action_spec, policy.action_spec)
    self.assertEqual(self.policy_state_spec, policy.policy_state_spec)

  @parameterized.parameters(*NP_PARAMETERS)
  def test_get_initial_state(self, multithreading, num_policies):
    policy = self._make_batched_py_policy(multithreading,
                                          num_policies=num_policies)
    policy_state = policy.get_initial_state()
    # Expect policy_state.shape[0] to be batch_size aka num_policies.
    # The remaining dimensions should match the policy_state_spec.
    correct_shape = (num_policies,) + self.policy_state_spec.shape
    self.assertEqual(correct_shape, policy_state.shape)

  @parameterized.parameters(*NP_PARAMETERS)
  def test_action(self, multithreading, num_policies):
    policy = self._make_batched_py_policy(multithreading,
                                          num_policies=num_policies)
    time_steps = np.array([
        ts.restart(observation=np.array([1]))
        for _ in range(num_policies)])

    # Call policy.action() and assert PolicySteps are batched correctly.
    policy_step = policy.action(time_steps)
    self.assertEqual(num_policies, policy_step.action.shape[0])

    # Take another step and assert that actions have the same shape.
    policy_step2 = policy.action(time_steps)
    self.assertAllEqual(policy_step.action.shape[0],
                        policy_step2.action.shape[0])


if __name__ == '__main__':
  tf.test.main()
