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

"""Tests for tf_agents.trajectories.policy_step."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.trajectories import policy_step


class PolicyStepTest(tf.test.TestCase):

  def testCreate(self):
    action = 1
    state = 2
    info = 3
    step = policy_step.PolicyStep(action=action, state=state, info=info)
    self.assertEqual(step.action, action)
    self.assertEqual(step.state, state)
    self.assertEqual(step.info, info)

  def testCreateWithAllDefaults(self):
    action = 1
    state = ()
    info = ()
    step = policy_step.PolicyStep(action)
    self.assertEqual(step.action, action)
    self.assertEqual(step.state, state)
    self.assertEqual(step.info, info)

  def testCreateWithDefaultInfo(self):
    action = 1
    state = 2
    info = ()
    step = policy_step.PolicyStep(action, state)
    self.assertEqual(step.action, action)
    self.assertEqual(step.state, state)
    self.assertEqual(step.info, info)


if __name__ == '__main__':
  tf.test.main()
