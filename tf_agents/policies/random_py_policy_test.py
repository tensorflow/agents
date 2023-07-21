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

"""Test for tf_agents.utils.random_py_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import test_utils


class RandomPyPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(RandomPyPolicyTest, self).setUp()
    self._time_step_spec = time_step.time_step_spec(
        observation_spec=array_spec.ArraySpec((1,), np.int32))
    self._time_step = time_step.restart(observation=np.array([1]))

  def testGeneratesActions(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 3), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=self._time_step_spec, action_spec=action_spec)

    action_step = policy.action(self._time_step)
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
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        outer_dims=(3,))

    action_step = policy.action(self._time_step)
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
        time_step_spec=self._time_step_spec, action_spec=[])
    self.assertEqual(policy.policy_state_spec, ())

  def testMasking(self):
    batch_size = 1000

    time_step_spec = time_step.time_step_spec(
        observation_spec=array_spec.ArraySpec((1,), np.int32))
    action_spec = array_spec.BoundedArraySpec((), np.int64, -5, 5)

    # We create a fixed mask here for testing purposes. Normally the mask would
    # be part of the observation.
    mask = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    np_mask = np.array(mask)
    batched_mask = np.array([mask for _ in range(batch_size)])

    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        observation_and_action_constraint_splitter=(
            lambda obs: (obs, batched_mask)))

    my_time_step = time_step.restart(time_step_spec, batch_size)
    action_step = policy.action(my_time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    # Sample from the policy 1000 times, and ensure that actions considered
    # invalid according to the mask are never chosen.
    action_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(action_ >= -5))
    self.assertTrue(np.all(action_ <= 5))
    self.assertAllEqual(np_mask[action_ - action_spec.minimum],
                        np.ones([batch_size]))

    # Ensure that all valid actions occur somewhere within the batch. Because we
    # sample 1000 times, the chance of this failing for any particular action is
    # (2/3)^1000, roughly 1e-176.
    for index in range(action_spec.minimum, action_spec.maximum + 1):
      if np_mask[index - action_spec.minimum]:
        self.assertIn(index, action_)


if __name__ == '__main__':
  test_utils.main()
