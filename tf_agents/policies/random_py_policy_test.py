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

"""Test for tf_agents.utils.random_py_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import tensorflow as tf
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step


class RandomPyPolicyTest(absltest.TestCase):

  def testGeneratesActions(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=None, action_spec=action_spec)

    action_step = policy.action(None)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    self.assertTrue(np.all(action_step.action[0] >= -10))
    self.assertTrue(np.all(action_step.action[0] <= 10))
    self.assertTrue(np.all(action_step.action[1] >= -10))
    self.assertTrue(np.all(action_step.action[1] <= 10))

  def testGeneratesBatchedActions(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=None, action_spec=action_spec, outer_dims=(3,))

    action_step = policy.action(None)
    tf.nest.assert_same_structure(action_spec, action_step.action)
    self.assertEqual((3, 2, 3), action_step.action[0].shape)
    self.assertEqual((3, 1, 2), action_step.action[1].shape)

    self.assertTrue(np.all(action_step.action[0] >= -10))
    self.assertTrue(np.all(action_step.action[0] <= 10))
    self.assertTrue(np.all(action_step.action[1] >= -10))
    self.assertTrue(np.all(action_step.action[1] <= 10))

  def testGeneratesBatchedActionsWithoutSpecifyingOuterDims(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]
    time_step_spec = time_step.time_step_spec(
        observation_spec=array_spec.ArraySpec((1,), np.int32))
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    action_step = policy.action(
        time_step.restart(np.array([[1], [2], [3]], dtype=np.int32)))
    tf.nest.assert_same_structure(action_spec, action_step.action)
    self.assertEqual((3, 2, 3), action_step.action[0].shape)
    self.assertEqual((3, 1, 2), action_step.action[1].shape)

    self.assertTrue(np.all(action_step.action[0] >= -10))
    self.assertTrue(np.all(action_step.action[0] <= 10))
    self.assertTrue(np.all(action_step.action[1] >= -10))
    self.assertTrue(np.all(action_step.action[1] <= 10))

  def testPolicyStateSpecIsEmpty(self):
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=None, action_spec=[])
    self.assertEqual(policy.policy_state_spec, ())


if __name__ == '__main__':
  absltest.main()
