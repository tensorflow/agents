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

"""Tests for environments.trajectory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.environments import time_step as ts
from tf_agents.environments import trajectory


class TrajectoryTest(tf.test.TestCase):

  def testFirstTensors(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = tf.constant([1.0, 1.0, 2.0])
    discount = tf.constant([1.0, 1.0, 1.0])
    traj = trajectory.first(observation, action, policy_info, reward, discount)
    self.assertTrue(tf.is_tensor(traj.step_type))
    traj_val = self.evaluate(traj)
    self.assertAllEqual(traj_val.step_type, [ts.StepType.FIRST] * 3)
    self.assertAllEqual(traj_val.next_step_type, [ts.StepType.MID] * 3)

  def testFirstArrays(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = np.array([1.0, 1.0, 2.0])
    discount = np.array([1.0, 1.0, 1.0])
    traj = trajectory.first(observation, action, policy_info, reward, discount)
    self.assertFalse(tf.is_tensor(traj.step_type))
    self.assertAllEqual(traj.step_type, [ts.StepType.FIRST] * 3)
    self.assertAllEqual(traj.next_step_type, [ts.StepType.MID] * 3)

  def testFromEpisodeTensor(self):
    observation = tf.random.uniform((4, 5))
    action = ()
    policy_info = ()
    reward = tf.random.uniform((4,))
    traj = trajectory.from_episode(
        observation, action, policy_info, reward, discount=None)
    self.assertTrue(tf.is_tensor(traj.step_type))
    traj_val, obs_val, reward_val = self.evaluate((traj, observation, reward))
    first = ts.StepType.FIRST
    mid = ts.StepType.MID
    last = ts.StepType.LAST
    self.assertAllEqual(
        traj_val.step_type, [first, mid, mid, mid])
    self.assertAllEqual(
        traj_val.next_step_type, [mid, mid, mid, last])
    self.assertAllEqual(traj_val.observation, obs_val)
    self.assertAllEqual(traj_val.reward, reward_val)
    self.assertAllEqual(traj_val.discount, [1.0, 1.0, 1.0, 1.0])

  def testFromEpisodeArray(self):
    observation = np.random.rand(4, 5)
    action = ()
    policy_info = ()
    reward = np.random.rand(4)
    traj = trajectory.from_episode(
        observation, action, policy_info, reward, discount=None)
    self.assertFalse(tf.is_tensor(traj.step_type))
    first = ts.StepType.FIRST
    mid = ts.StepType.MID
    last = ts.StepType.LAST
    self.assertAllEqual(
        traj.step_type, [first, mid, mid, mid])
    self.assertAllEqual(
        traj.next_step_type, [mid, mid, mid, last])
    self.assertAllEqual(traj.observation, observation)
    self.assertAllEqual(traj.reward, reward)
    self.assertAllEqual(traj.discount, [1.0, 1.0, 1.0, 1.0])


if __name__ == '__main__':
  tf.test.main()
