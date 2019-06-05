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

"""Tests for reinforment_learning.environment.tf_py_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

import tensorflow as tf

from tf_agents import specs
from tf_agents.environments import batched_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts


class PYEnvironmentMock(py_environment.PyEnvironment):
  """MockPyEnvironment.

  Stores all actions taken in `actions_taken`. The returned values are:

  step: step_type, discount, reward, observation

  step: FIRST, 1., 0., [0]
  step: MID, 1., 0., [1]
  step: LAST, 0., 1. [2]
  ...repeated
  """

  def __init__(self):
    self.actions_taken = []
    self.steps = 0
    self.episodes = 0
    self.resets = 0
    self._state = 0

  def _reset(self):
    self._state = 0
    self.resets += 1
    return ts.restart([self._state])

  def _step(self, action):
    self._state = (self._state + 1) % 3
    self.steps += 1
    self.actions_taken.append(action)

    observation = [self._state]
    if self._state == 0:
      return ts.restart(observation)
    elif self._state == 2:
      self.episodes += 1
      return ts.termination(observation, reward=1.0)
    return ts.transition(observation, reward=0.0)

  def action_spec(self):
    return specs.BoundedArraySpec(
        [], np.int32, minimum=0, maximum=10, name='action')

  def observation_spec(self):
    return specs.ArraySpec([], np.int64, name='observation')


class TFPYEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  def testPyenv(self):
    py_env = PYEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    self.assertIsInstance(tf_env.pyenv,
                          batched_py_environment.BatchedPyEnvironment)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testActionSpec(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      py_env = batched_py_environment.BatchedPyEnvironment([py_env])
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    self.assertTrue(tf_env.batched)
    self.assertEqual(tf_env.batch_size, 1)
    spec = tf_env.action_spec()
    self.assertEqual(type(spec), specs.BoundedTensorSpec)
    self.assertEqual(spec.dtype, tf.int32)
    self.assertEqual(spec.shape, tf.TensorShape([]))
    self.assertEqual(spec.name, 'action')

  def testObservationSpec(self):
    py_env = PYEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    spec = tf_env.observation_spec()
    self.assertEqual(type(spec), specs.TensorSpec)
    self.assertEqual(spec.dtype, tf.int64)
    self.assertEqual(spec.shape, tf.TensorShape([]))
    self.assertEqual(spec.name, 'observation')

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testTimeStepSpec(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    spec = tf_env.time_step_spec()

    # step_type
    self.assertEqual(type(spec.step_type), specs.TensorSpec)
    self.assertEqual(spec.step_type.dtype, tf.int32)
    self.assertEqual(spec.step_type.shape, tf.TensorShape([]))

    # reward
    self.assertEqual(type(spec.reward), specs.TensorSpec)
    self.assertEqual(spec.reward.dtype, tf.float32)
    self.assertEqual(spec.reward.shape, tf.TensorShape([]))

    # discount
    self.assertEqual(type(spec.discount), specs.BoundedTensorSpec)
    self.assertEqual(spec.discount.dtype, tf.float32)
    self.assertEqual(spec.discount.shape, tf.TensorShape([]))
    self.assertEqual(spec.discount.minimum, 0.0)
    self.assertEqual(spec.discount.maximum, 1.0)

    # observation
    self.assertEqual(type(spec.observation), specs.TensorSpec)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testResetOp(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    reset = tf_env.reset()
    self.evaluate(reset)
    self.assertEqual(1, py_env.resets)
    self.assertEqual(0, py_env.steps)
    self.assertEqual(0, py_env.episodes)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testMultipleReset(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)

    self.evaluate(tf_env.reset())
    self.assertEqual(1, py_env.resets)
    self.evaluate(tf_env.reset())
    self.assertEqual(2, py_env.resets)
    self.evaluate(tf_env.reset())
    self.assertEqual(3, py_env.resets)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testFirstTimeStep(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    time_step = tf_env.current_time_step()
    time_step = self.evaluate(time_step)
    self.assertAllEqual([ts.StepType.FIRST], time_step.step_type)
    self.assertAllEqual([0.0], time_step.reward)
    self.assertAllEqual([1.0], time_step.discount)
    self.assertAllEqual([0], time_step.observation)
    self.assertAllEqual([], py_env.actions_taken)
    self.assertEqual(1, py_env.resets)
    self.assertEqual(0, py_env.steps)
    self.assertEqual(0, py_env.episodes)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testOneStep(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([1])
    time_step = self.evaluate(tf_env.step(action))

    self.assertAllEqual([ts.StepType.MID], time_step.step_type)
    self.assertAllEqual([0.], time_step.reward)
    self.assertAllEqual([1.0], time_step.discount)
    self.assertAllEqual([1], time_step.observation)
    self.assertAllEqual([1], py_env.actions_taken)
    self.assertEqual(1, py_env.resets)
    self.assertEqual(1, py_env.steps)
    self.assertEqual(0, py_env.episodes)

  def testBatchedFirstTimeStepAndOneStep(self):
    py_envs = [PYEnvironmentMock() for _ in range(3)]
    batched_py_env = batched_py_environment.BatchedPyEnvironment(py_envs)
    tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    self.assertEqual(tf_env.batch_size, 3)
    time_step_0 = tf_env.current_time_step()
    time_step_0_val = self.evaluate(time_step_0)

    self.assertAllEqual([ts.StepType.FIRST] * 3, time_step_0_val.step_type)
    self.assertAllEqual([0.0] * 3, time_step_0_val.reward)
    self.assertAllEqual([1.0] * 3, time_step_0_val.discount)
    self.assertAllEqual(np.array([0, 0, 0]), time_step_0_val.observation)
    for py_env in py_envs:
      self.assertEqual([], py_env.actions_taken)
      self.assertEqual(1, py_env.resets)
      self.assertEqual(0, py_env.steps)
      self.assertEqual(0, py_env.episodes)

    time_step_1 = tf_env.step(np.array([1, 1, 1]))

    time_step_1_val = self.evaluate(time_step_1)

    self.assertAllEqual([ts.StepType.MID] * 3, time_step_1_val.step_type)
    self.assertAllEqual([0.] * 3, time_step_1_val.reward)
    self.assertAllEqual([1.0] * 3, time_step_1_val.discount)
    self.assertAllEqual(np.array([1, 1, 1]), time_step_1_val.observation)
    for py_env in py_envs:
      self.assertEqual([1], py_env.actions_taken)
      self.assertEqual(1, py_env.resets)
      self.assertEqual(1, py_env.steps)
      self.assertEqual(0, py_env.episodes)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testTwoStepsDependenceOnTheFirst(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([1])
    time_step = tf_env.step(action)
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([2])
    time_step = self.evaluate(tf_env.step(action))

    self.assertEqual(ts.StepType.LAST, time_step.step_type)
    self.assertEqual([2], time_step.observation)
    self.assertEqual(1., time_step.reward)
    self.assertEqual(0., time_step.discount)
    self.assertEqual([1, 2], py_env.actions_taken)

  @parameterized.parameters({'batch_py_env': True}, {'batch_py_env': False})
  def testFirstObservationIsPreservedAfterTwoSteps(self, batch_py_env):
    py_env = PYEnvironmentMock()
    if batch_py_env:
      batched_py_env = batched_py_environment.BatchedPyEnvironment([py_env])
      tf_env = tf_py_environment.TFPyEnvironment(batched_py_env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([1])
    next_time_step = tf_env.step(action)
    with tf.control_dependencies([next_time_step.step_type]):
      action = tf.constant([2])
    _, observation = self.evaluate([tf_env.step(action), time_step.observation])

    self.assertEqual(np.array([0]), observation)


if __name__ == '__main__':
  tf.test.main()
