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

"""Tests for tf_agents.drivers.dynamic_episode_driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.environments import tf_py_environment
from tf_agents.environments import trajectory
from tf_agents.policies import policy_step
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec


nest = tf.contrib.framework.nest


class NumStepsObserver(object):

  def __init__(self, variable_scope='num_steps_step_observer'):
    with tf.variable_scope(variable_scope):
      self._num_steps = tf.get_variable(
          'num_steps',
          shape=[],
          dtype=tf.int32,
          initializer=tf.zeros_initializer())

  @property
  def num_steps(self):
    return self._num_steps

  def __call__(self, traj):
    num_steps = tf.reduce_sum(tf.to_int32(~traj.is_boundary()))
    with tf.control_dependencies([self._num_steps.assign_add(num_steps)]):
      return nest.map_structure(tf.identity, traj)


class NumEpisodesObserver(object):

  def __init__(self, variable_scope='num_episodes_step_observer'):
    with tf.variable_scope(variable_scope):
      self._num_episodes = tf.get_variable(
          'num_episodes',
          shape=[],
          dtype=tf.int32,
          initializer=tf.zeros_initializer())

  @property
  def num_episodes(self):
    return self._num_episodes

  def __call__(self, traj):
    with tf.control_dependencies(
        [self._num_episodes.assign_add(tf.to_int32(traj.is_last()[0]))]):
      return nest.map_structure(
          tf.identity, traj)


def make_replay_buffer(tf_env):
  """Default replay buffer factory."""

  time_step_spec = tf_env.time_step_spec()
  action_spec = tf_env.action_spec()
  action_step_spec = policy_step.PolicyStep(
      action_spec, (), tensor_spec.TensorSpec((), tf.int32))
  trajectory_spec = trajectory.from_transition(time_step_spec, action_step_spec,
                                               time_step_spec)
  return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      trajectory_spec, batch_size=1)


class DynamicEpisodeDriverTest(tf.test.TestCase):

  def testOneStepUpdatesObservers(self):
    env = tf_py_environment.TFPyEnvironment(
        driver_test_utils.PyEnvironmentMock())
    policy = driver_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    policy_state = policy.get_initial_state(1)
    num_episodes_observer = NumEpisodesObserver()
    num_steps_observer = NumStepsObserver()

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, observers=[num_episodes_observer, num_steps_observer])
    run_driver = driver.run(policy_state=policy_state)

    self.evaluate(tf.global_variables_initializer())
    for _ in range(5):
      self.evaluate(run_driver)

    self.assertEqual(self.evaluate(num_episodes_observer.num_episodes), 5)
    # Two steps per episode.
    self.assertEqual(self.evaluate(num_steps_observer.num_steps), 10)

  def testMultiStepUpdatesObservers(self):
    env = tf_py_environment.TFPyEnvironment(
        driver_test_utils.PyEnvironmentMock())
    policy = driver_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    policy_state = policy.get_initial_state(1)
    num_episodes_observer = NumEpisodesObserver()
    num_steps_observer = NumStepsObserver()

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, num_episodes=5, observers=[num_episodes_observer,
                                                num_steps_observer])

    run_driver = driver.run(policy_state=policy_state)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(run_driver)
    self.assertEqual(self.evaluate(num_episodes_observer.num_episodes), 5)
    # Two steps per episode.
    self.assertEqual(self.evaluate(num_steps_observer.num_steps), 10)

  def testTwoStepObservers(self):
    env = tf_py_environment.TFPyEnvironment(
        driver_test_utils.PyEnvironmentMock())
    policy = driver_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    policy_state = policy.get_initial_state(1)
    num_episodes_observer0 = NumEpisodesObserver(variable_scope='observer0')
    num_episodes_observer1 = NumEpisodesObserver(variable_scope='observer1')

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, num_episodes=5, observers=[num_episodes_observer0,
                                                num_episodes_observer1])
    run_driver = driver.run(policy_state=policy_state)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(run_driver)
    self.assertEqual(self.evaluate(num_episodes_observer0.num_episodes), 5)
    self.assertEqual(self.evaluate(num_episodes_observer1.num_episodes), 5)

  def testOneStepReplayBufferObservers(self):
    env = tf_py_environment.TFPyEnvironment(
        driver_test_utils.PyEnvironmentMock())
    policy = driver_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    policy_state = policy.get_initial_state(1)
    replay_buffer = make_replay_buffer(env)

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, num_episodes=1, observers=[replay_buffer.add_batch])

    run_driver = driver.run(policy_state=policy_state)
    rb_gather_all = replay_buffer.gather_all()

    self.evaluate(tf.global_variables_initializer())

    for _ in range(3):
      self.evaluate(run_driver)

    trajectories = self.evaluate(rb_gather_all)

    self.assertAllEqual(trajectories.step_type, [[0, 1, 2, 0, 1, 2, 0, 1, 2]])
    self.assertAllEqual(trajectories.action, [[1, 2, 1, 1, 2, 1, 1, 2, 1]])
    self.assertAllEqual(trajectories.observation, [[0, 1, 3, 0, 1, 3, 0, 1, 3]])
    self.assertAllEqual(trajectories.policy_info, [[2, 4, 2, 2, 4, 2, 2, 4, 2]])
    self.assertAllEqual(trajectories.next_step_type,
                        [[1, 2, 0, 1, 2, 0, 1, 2, 0]])
    self.assertAllEqual(trajectories.reward,
                        [[1., 1., 0., 1., 1., 0., 1., 1., 0.]])
    self.assertAllEqual(trajectories.discount,
                        [[1., 0., 1, 1, 0, 1., 1., 0., 1.]])

  def testMultiStepReplayBufferObservers(self):
    env = tf_py_environment.TFPyEnvironment(
        driver_test_utils.PyEnvironmentMock())
    policy = driver_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec())
    policy_state = policy.get_initial_state(1)
    replay_buffer = make_replay_buffer(env)

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, num_episodes=3, observers=[replay_buffer.add_batch])

    run_driver = driver.run(policy_state=policy_state)
    rb_gather_all = replay_buffer.gather_all()

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(run_driver)
    trajectories = self.evaluate(rb_gather_all)

    self.assertAllEqual(trajectories.step_type, [[0, 1, 2, 0, 1, 2, 0, 1, 2]])
    self.assertAllEqual(trajectories.action, [[1, 2, 1, 1, 2, 1, 1, 2, 1]])
    self.assertAllEqual(trajectories.observation, [[0, 1, 3, 0, 1, 3, 0, 1, 3]])
    self.assertAllEqual(trajectories.policy_info, [[2, 4, 2, 2, 4, 2, 2, 4, 2]])
    self.assertAllEqual(trajectories.next_step_type,
                        [[1, 2, 0, 1, 2, 0, 1, 2, 0]])
    self.assertAllEqual(trajectories.reward,
                        [[1., 1., 0., 1., 1., 0., 1., 1., 0.]])
    self.assertAllEqual(trajectories.discount,
                        [[1., 0., 1., 1., 0., 1., 1., 0., 1.]])


if __name__ == '__main__':
  tf.test.main()
