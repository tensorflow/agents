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

"""Tests for tf_agents.policies.scripted_py_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tf_agents.policies import scripted_py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class ScriptedPyPolicyTest(test_utils.TestCase):

  def setUp(self):
    super(ScriptedPyPolicyTest, self).setUp()
    self._obs_spec = array_spec.ArraySpec((), np.int32, 'obs')
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._time_step = ts.restart(observation=1)  # pytype: disable=wrong-arg-types

  def testFollowsScript(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 2), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]

    action_script = [
        (1, [
            np.array([[5, 2], [1, 3]], dtype=np.int32),
            np.array([[4, 6]], dtype=np.int32)
        ]),
        (0, [
            np.array([[0, 0], [0, 0]], dtype=np.int32),
            np.array([[0, 0]], dtype=np.int32)
        ]),
        (2, [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32)
        ]),
    ]

    policy = scripted_py_policy.ScriptedPyPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        action_script=action_script)
    policy_state = policy.get_initial_state()

    action_step = policy.action(self._time_step, policy_state)
    self.assertEqual(action_script[0][1], action_step.action)
    action_step = policy.action(self._time_step, action_step.state)
    self.assertEqual(action_script[2][1], action_step.action)
    action_step = policy.action(self._time_step, action_step.state)
    self.assertEqual(action_script[2][1], action_step.action)

  def testFollowsScriptWithListInsteadOfNpArrays(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 2), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]

    action_script = [
        (1, [
            [[5, 2], [1, 3]],
            [[4, 6]],
        ]),
        (2, [[[1, 2], [3, 4]], [[5, 6]]]),
    ]

    expected = [
        [
            np.array([[5, 2], [1, 3]], dtype=np.int32),
            np.array([[4, 6]], dtype=np.int32)
        ],
        [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32)
        ],
    ]

    policy = scripted_py_policy.ScriptedPyPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        action_script=action_script)  # pytype: disable=wrong-arg-types
    policy_state = policy.get_initial_state()

    action_step = policy.action(self._time_step, policy_state)
    np.testing.assert_array_equal(expected[0][0], action_step.action[0])
    np.testing.assert_array_equal(expected[0][1], action_step.action[1])
    action_step = policy.action(self._time_step, action_step.state)
    np.testing.assert_array_equal(expected[1][0], action_step.action[0])
    np.testing.assert_array_equal(expected[1][1], action_step.action[1])
    action_step = policy.action(self._time_step, action_step.state)
    np.testing.assert_array_equal(expected[1][0], action_step.action[0])
    np.testing.assert_array_equal(expected[1][1], action_step.action[1])

  def testChecksSpecBounds(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 2), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]

    action_script = [
        (1, [
            np.array([[15, 2], [1, 3]], dtype=np.int32),
            np.array([[4, 6]], dtype=np.int32)
        ]),
        (2, [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32)
        ]),
    ]

    policy = scripted_py_policy.ScriptedPyPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        action_script=action_script)
    policy_state = policy.get_initial_state()

    with self.assertRaises(ValueError):
      policy.action(self._time_step, policy_state)

  def testChecksSpecNest(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 2), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]

    action_script = [
        (1, [np.array([[5, 2], [1, 3]], dtype=np.int32)]),
        (2, [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32)
        ]),
    ]

    policy = scripted_py_policy.ScriptedPyPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        action_script=action_script)
    policy_state = policy.get_initial_state()

    with self.assertRaises(ValueError):
      policy.action(self._time_step, policy_state)

  def testEpisodeLength(self):
    action_spec = [
        array_spec.BoundedArraySpec((2, 2), np.int32, -10, 10),
        array_spec.BoundedArraySpec((1, 2), np.int32, -10, 10)
    ]

    action_script = [
        (1, [
            np.array([[5, 2], [1, 3]], dtype=np.int32),
            np.array([[4, 6]], dtype=np.int32)
        ]),
        (2, [
            np.array([[1, 2], [3, 4]], dtype=np.int32),
            np.array([[5, 6]], dtype=np.int32)
        ]),
    ]

    policy = scripted_py_policy.ScriptedPyPolicy(
        time_step_spec=self._time_step_spec,
        action_spec=action_spec,
        action_script=action_script)
    policy_state = policy.get_initial_state()

    action_step = policy.action(self._time_step, policy_state)
    self.assertEqual(action_script[0][1], action_step.action)
    action_step = policy.action(self._time_step, action_step.state)
    self.assertEqual(action_script[1][1], action_step.action)
    action_step = policy.action(self._time_step, action_step.state)
    self.assertEqual(action_script[1][1], action_step.action)
    with self.assertRaisesRegexp(ValueError, '.*Episode is longer than.*'):
      policy.action(self._time_step, action_step.state)

  def testPolicyStateSpecIsEmpty(self):
    policy = scripted_py_policy.ScriptedPyPolicy(
        time_step_spec=self._time_step_spec, action_spec=[], action_script=[])
    self.assertEqual(policy.policy_state_spec, ())

if __name__ == '__main__':
  test_utils.main()
