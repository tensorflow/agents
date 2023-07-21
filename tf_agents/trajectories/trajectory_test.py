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

"""Tests for trajectory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import test_utils as drivers_test_utils
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils


class TrajectoryTest(test_utils.TestCase):

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

  def testMidTensors(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = tf.constant([1.0, 1.0, 2.0])
    discount = tf.constant([1.0, 1.0, 1.0])
    traj = trajectory.mid(observation, action, policy_info, reward, discount)
    self.assertTrue(tf.is_tensor(traj.step_type))
    traj_val = self.evaluate(traj)
    self.assertAllEqual(traj_val.step_type, [ts.StepType.MID] * 3)
    self.assertAllEqual(traj_val.next_step_type, [ts.StepType.MID] * 3)

  def testMidArrays(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = np.array([1.0, 1.0, 2.0])
    discount = np.array([1.0, 1.0, 1.0])
    traj = trajectory.mid(observation, action, policy_info, reward, discount)
    self.assertFalse(tf.is_tensor(traj.step_type))
    self.assertAllEqual(traj.step_type, [ts.StepType.MID] * 3)
    self.assertAllEqual(traj.next_step_type, [ts.StepType.MID] * 3)

  def testLastTensors(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = tf.constant([1.0, 1.0, 2.0])
    discount = tf.constant([1.0, 1.0, 1.0])
    traj = trajectory.last(observation, action, policy_info, reward, discount)
    self.assertTrue(tf.is_tensor(traj.step_type))
    traj_val = self.evaluate(traj)
    self.assertAllEqual(traj_val.step_type, [ts.StepType.MID] * 3)
    self.assertAllEqual(traj_val.next_step_type, [ts.StepType.LAST] * 3)

  def testLastArrays(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = np.array([1.0, 1.0, 2.0])
    discount = np.array([1.0, 1.0, 1.0])
    traj = trajectory.last(observation, action, policy_info, reward, discount)
    self.assertFalse(tf.is_tensor(traj.step_type))
    self.assertAllEqual(traj.step_type, [ts.StepType.MID] * 3)
    self.assertAllEqual(traj.next_step_type, [ts.StepType.LAST] * 3)

  def testSingleStepTensors(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = tf.constant([1.0, 1.0, 2.0])
    discount = tf.constant([1.0, 1.0, 1.0])
    traj = trajectory.single_step(observation, action, policy_info, reward,
                                  discount)
    self.assertTrue(tf.is_tensor(traj.step_type))
    traj_val = self.evaluate(traj)
    self.assertAllEqual(traj_val.step_type, [ts.StepType.FIRST] * 3)
    self.assertAllEqual(traj_val.next_step_type, [ts.StepType.LAST] * 3)

  def testSingleStepArrays(self):
    observation = ()
    action = ()
    policy_info = ()
    reward = np.array([1.0, 1.0, 2.0])
    discount = np.array([1.0, 1.0, 1.0])
    traj = trajectory.single_step(observation, action, policy_info, reward,
                                  discount)
    self.assertFalse(tf.is_tensor(traj.step_type))
    self.assertAllEqual(traj.step_type, [ts.StepType.FIRST] * 3)
    self.assertAllEqual(traj.next_step_type, [ts.StepType.LAST] * 3)

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
    self.assertAllClose(traj_val.observation, obs_val)
    self.assertAllEqual(traj_val.reward, reward_val)
    self.assertAllEqual(traj_val.discount, [1.0, 1.0, 1.0, 1.0])

  def testFromEpisodeWithCompositeTensorOfTensors(self):
    observation = tf.SparseTensor(
        indices=tf.random.uniform((7, 2), maxval=9, dtype=tf.int64),
        values=tf.random.uniform((7,)),
        dense_shape=[4, 10])  # The 4 is important, it must match reward length.
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
    self.assertAllClose(traj_val.observation, obs_val)
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

  def testToTransition(self):
    first = ts.StepType.FIRST
    mid = ts.StepType.MID
    last = ts.StepType.LAST

    # Define a batch size 1, 3-step trajectory.
    traj = trajectory.Trajectory(
        step_type=np.array([[first, mid, last]]),
        next_step_type=np.array([[mid, last, first]]),
        observation=np.array([[10.0, 20.0, 30.0]]),
        action=np.array([[11.0, 22.0, 33.0]]),
        # reward at step 2 is an invalid dummy reward.
        reward=np.array([[0.0, 1.0, 2.0]]),
        discount=np.array([[1.0, 1.0, 0.0]]),
        policy_info=np.array([[1.0, 2.0, 3.0]]))

    transition = trajectory.to_transition(traj)
    self.assertIsInstance(transition, trajectory.Transition)
    time_steps, policy_steps, next_time_steps = transition

    self.assertAllEqual(time_steps.step_type, np.array([[first, mid]]))
    self.assertAllEqual(time_steps.observation, np.array([[10.0, 20.0]]))
    # reward and discount are filled with zero (dummy) values
    self.assertAllEqual(time_steps.reward, np.array([[0.0, 0.0]]))
    self.assertAllEqual(time_steps.discount, np.array([[0.0, 0.0]]))

    self.assertAllEqual(next_time_steps.step_type, np.array([[mid, last]]))
    self.assertAllEqual(next_time_steps.observation, np.array([[20.0, 30.0]]))
    self.assertAllEqual(next_time_steps.reward, np.array([[0.0, 1.0]]))
    self.assertAllEqual(next_time_steps.discount, np.array([[1.0, 1.0]]))

    self.assertAllEqual(policy_steps.action, np.array([[11.0, 22.0]]))
    self.assertAllEqual(policy_steps.info, np.array([[1.0, 2.0]]))

  def testToNStepTransitionForNEquals1(self):
    first = ts.StepType.FIRST
    last = ts.StepType.LAST

    # Define a batch size 1, 2-step trajectory.
    traj = trajectory.Trajectory(
        step_type=np.array([[first, last]]),
        next_step_type=np.array([[last, first]]),
        observation=np.array([[10.0, 20.0]]),
        action=np.array([[11.0, 22.0]]),
        # reward & discount values at step 1 is an invalid dummy reward.
        reward=np.array([[-1.0, 0.0]]),
        discount=np.array([[0.9, 0.0]]),
        policy_info=np.array([[10.0, 20.0]]))

    transition = trajectory.to_n_step_transition(traj, gamma=0.5)
    self.assertIsInstance(transition, trajectory.Transition)
    time_steps, policy_steps, next_time_steps = transition

    self.assertAllEqual(time_steps.step_type, np.array([first]))
    self.assertAllEqual(time_steps.observation, np.array([10.0]))
    self.assertAllEqual(time_steps.reward, np.array([np.nan]))
    self.assertAllEqual(time_steps.discount, np.array([np.nan]))

    self.assertAllEqual(next_time_steps.step_type, np.array([last]))
    self.assertAllEqual(next_time_steps.observation, np.array([20.0]))
    # r0
    self.assertAllEqual(next_time_steps.reward, np.array([-1.0]))
    # d0
    self.assertAllEqual(next_time_steps.discount, np.array([0.9]))

    self.assertAllEqual(policy_steps.action, np.array([11.0]))
    self.assertAllEqual(policy_steps.info, np.array([10.0]))

  def testToNStepTransition(self):
    first = ts.StepType.FIRST
    mid = ts.StepType.MID
    last = ts.StepType.LAST
    gamma = 0.5

    # Define a batch size 1, 4-step trajectory.
    traj = trajectory.Trajectory(
        step_type=np.array([[first, mid, mid, last]]),
        next_step_type=np.array([[mid, mid, last, first]]),
        observation=np.array([[10.0, 20.0, 30.0, 40.0]]),
        action=np.array([[11.0, 22.0, 33.0, 44.0]]),
        # reward & discount values at step 3 is an invalid dummy reward.
        reward=np.array([[-1.0, 1.0, 2.0, 0.0]]),
        discount=np.array([[0.9, 0.95, 1.0, 0.0]]),
        policy_info=np.array([[10.0, 20.0, 30.0, 40.0]]))

    transition = trajectory.to_n_step_transition(traj, gamma=gamma)
    self.assertIsInstance(transition, trajectory.Transition)
    time_steps, policy_steps, next_time_steps = transition

    self.assertAllEqual(time_steps.step_type, np.array([first]))
    self.assertAllEqual(time_steps.observation, np.array([10.0]))
    self.assertAllEqual(time_steps.reward, np.array([np.nan]))
    self.assertAllEqual(time_steps.discount, np.array([np.nan]))

    self.assertAllEqual(next_time_steps.step_type, np.array([last]))
    self.assertAllEqual(next_time_steps.observation, np.array([40.0]))
    # r0 + r1 * g * d0 + r2 * g * d0 * d1
    #   == -1.0 + 1.0*0.5*(0.9) + 2.0*(0.5**2)*(0.9*0.95)
    self.assertAllEqual(
        next_time_steps.reward,
        np.array([-1.0 + 1.0 * gamma * 0.9 + 2.0 * gamma**2 * 0.9 * 0.95]))
    # gamma**2 * (d0 * d1 * d2)
    self.assertAllEqual(
        next_time_steps.discount, np.array([gamma**2 * (0.9 * 0.95 * 1.0)]))

    self.assertAllEqual(policy_steps.action, np.array([11.0]))
    self.assertAllEqual(policy_steps.info, np.array([10.0]))

  def testToTransitionHandlesTrajectoryFromDriverCorrectly(self):
    env = tf_py_environment.TFPyEnvironment(
        drivers_test_utils.PyEnvironmentMock())
    policy = drivers_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    replay_buffer = drivers_test_utils.make_replay_buffer(policy)

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, num_episodes=3, observers=[replay_buffer.add_batch])

    run_driver = driver.run()
    rb_gather_all = replay_buffer.gather_all()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(run_driver)
    trajectories = self.evaluate(rb_gather_all)

    transitions = trajectory.to_transition(trajectories)
    self.assertIsInstance(transitions, trajectory.Transition)
    time_steps, policy_step, next_time_steps = transitions

    self.assertAllEqual(time_steps.observation,
                        trajectories.observation[:, :-1])
    self.assertAllEqual(time_steps.step_type, trajectories.step_type[:, :-1])
    self.assertAllEqual(next_time_steps.observation,
                        trajectories.observation[:, 1:])
    self.assertAllEqual(next_time_steps.step_type,
                        trajectories.step_type[:, 1:])
    self.assertAllEqual(next_time_steps.reward, trajectories.reward[:, :-1])
    self.assertAllEqual(next_time_steps.discount, trajectories.discount[:, :-1])

    self.assertAllEqual(policy_step.action, trajectories.action[:, :-1])
    self.assertAllEqual(policy_step.info, trajectories.policy_info[:, :-1])

  def testToTransitionSpec(self):
    env = tf_py_environment.TFPyEnvironment(
        drivers_test_utils.PyEnvironmentMock())
    policy = drivers_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    trajectory_spec = policy.trajectory_spec
    transition_spec = trajectory.to_transition_spec(trajectory_spec)
    self.assertIsInstance(transition_spec, trajectory.Transition)
    ts_spec, ps_spec, nts_spec = transition_spec

    self.assertAllEqual(ts_spec, env.time_step_spec())
    self.assertAllEqual(ps_spec.action, env.action_spec())
    self.assertAllEqual(nts_spec, env.time_step_spec())


if __name__ == '__main__':
  tf.test.main()
