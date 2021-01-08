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

"""Tests for reinforment_learning.environment.tf_py_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
from typing import Text

from absl.testing import parameterized
from absl.testing.absltest import mock
import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.environments import batched_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

COMMON_PARAMETERS = (
    dict(batch_py_env=True, isolation=True),
    dict(batch_py_env=False, isolation=True),
    dict(batch_py_env=True, isolation=False),
    dict(batch_py_env=False, isolation=False),
)


def get(env, property_name):
  if isinstance(env, batched_py_environment.BatchedPyEnvironment):
    assert env.batch_size == 1
    return getattr(env.envs[0], property_name)
  return getattr(env, property_name)


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
    self.last_call_thread_id = threading.current_thread().ident
    self._state = 0

  def _reset(self):
    self._state = 0
    self.resets += 1
    self.last_call_thread_id = threading.current_thread().ident
    return ts.restart([self._state])  # pytype: disable=wrong-arg-types

  def _step(self, action):
    self._state = (self._state + 1) % 3
    self.steps += 1
    self.last_call_thread_id = threading.current_thread().ident
    self.actions_taken.append(action)

    observation = [self._state]
    if self._state == 0:
      return ts.restart(observation)  # pytype: disable=wrong-arg-types
    elif self._state == 2:
      self.episodes += 1
      return ts.termination(observation, reward=1.0)  # pytype: disable=wrong-arg-types
    return ts.transition(observation, reward=0.0)  # pytype: disable=wrong-arg-types

  def action_spec(self):
    return specs.BoundedArraySpec(
        [], np.int32, minimum=0, maximum=10, name='action')

  def observation_spec(self):
    return specs.ArraySpec([], np.int64, name='observation')

  def render(self, mode):
    assert isinstance(mode, (str, Text)), 'Got: {}'.format(type(mode))
    if mode == 'rgb_array':
      return np.ones((4, 4, 3), dtype=np.uint8)
    elif mode == 'human':
      # Many environments often do not return anything on human mode.
      return None
    else:
      raise ValueError('Unknown mode: {}'.format(mode))


class PYEnvironmentMockNestedRewards(py_environment.PyEnvironment):
  """Mock PyEnvironment with rewards that are nested dicts."""

  def __init__(self):
    self.last_call_thread_id = threading.current_thread().ident
    self._state = 0
    self._observation_spec = self.observation_spec()
    self._action_spec = self.action_spec()
    self._reward_spec = self.reward_spec()

  def action_spec(self):
    return specs.BoundedArraySpec(
        [], np.int32, minimum=0, maximum=10, name='action')

  def observation_spec(self):
    return specs.ArraySpec([], np.int64, name='observation')

  def reward_spec(self):
    return {
        'reward': specs.ArraySpec([], np.float32, name='reward'),
        'constraint': specs.ArraySpec([], np.float32, name='constraint')
    }

  def _reset(self):
    self._state = 0
    self.last_call_thread_id = threading.current_thread().ident
    return ts.restart(
        [self._state], batch_size=1, reward_spec=self._reward_spec)  # pytype: disable=wrong-arg-types

  def _step(self, action):
    self._state = (self._state + 1) % 3
    self.last_call_thread_id = threading.current_thread().ident

    observation = [self._state]
    reward = {
        'constraint': 2.,
        'reward': 1.,
    }
    if self._state == 0:
      return ts.restart(
          observation, batch_size=1, reward_spec=self._reward_spec)  # pytype: disable=wrong-arg-types
    elif self._state == 2:
      return ts.termination(observation, reward=reward)  # pytype: disable=wrong-arg-types
    return ts.transition(observation, reward=reward)  # pytype: disable=wrong-arg-types


class TFPYEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  def testPyenv(self):
    py_env = PYEnvironmentMock()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    self.assertIsInstance(tf_env.pyenv,
                          batched_py_environment.BatchedPyEnvironment)

  def _get_py_env(self, batch_py_env, isolation, batch_size=None):
    def _create_env():
      if batch_size is None:
        py_env = PYEnvironmentMock()
      else:
        py_env = [PYEnvironmentMock() for _ in range(batch_size)]
      if batch_py_env:
        py_env = batched_py_environment.BatchedPyEnvironment(
            py_env if isinstance(py_env, list) else [py_env])
      return py_env
    # If using isolation, we'll pass a callable
    return _create_env if isolation else _create_env()

  def testMethodPropagation(self):
    env = self._get_py_env(True, False, batch_size=1)
    env.foo = mock.Mock()
    tf_env = tf_py_environment.TFPyEnvironment(env)
    tf_env.foo()
    env.foo.assert_called_once()

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testActionSpec(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
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

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testTimeStepSpec(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
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

  @parameterized.parameters(
      *COMMON_PARAMETERS)
  def testResetOp(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
    reset = tf_env.reset()
    self.evaluate(reset)
    self.assertEqual(1, get(tf_env.pyenv, 'resets'))
    self.assertEqual(0, get(tf_env.pyenv, 'steps'))
    self.assertEqual(0, get(tf_env.pyenv, 'episodes'))

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testMultipleReset(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)

    self.evaluate(tf_env.reset())
    self.assertEqual(1, get(tf_env.pyenv, 'resets'))
    self.evaluate(tf_env.reset())
    self.assertEqual(2, get(tf_env.pyenv, 'resets'))
    self.evaluate(tf_env.reset())
    self.assertEqual(3, get(tf_env.pyenv, 'resets'))

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testFirstTimeStep(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
    time_step = tf_env.current_time_step()
    time_step = self.evaluate(time_step)
    self.assertAllEqual([ts.StepType.FIRST], time_step.step_type)
    self.assertAllEqual([0.0], time_step.reward)
    self.assertAllEqual([1.0], time_step.discount)
    self.assertAllEqual([0], time_step.observation)
    self.assertAllEqual([], get(tf_env.pyenv, 'actions_taken'))
    self.assertEqual(1, get(tf_env.pyenv, 'resets'))
    self.assertEqual(0, get(tf_env.pyenv, 'steps'))
    self.assertEqual(0, get(tf_env.pyenv, 'episodes'))

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testOneStep(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([1])
    time_step = self.evaluate(tf_env.step(action))

    self.assertAllEqual([ts.StepType.MID], time_step.step_type)
    self.assertAllEqual([0.], time_step.reward)
    self.assertAllEqual([1.0], time_step.discount)
    self.assertAllEqual([1], time_step.observation)
    self.assertAllEqual([1], get(tf_env.pyenv, 'actions_taken'))
    self.assertEqual(1, get(tf_env.pyenv, 'resets'))
    self.assertEqual(1, get(tf_env.pyenv, 'steps'))
    self.assertEqual(0, get(tf_env.pyenv, 'episodes'))

  @parameterized.parameters(dict(isolation=False), dict(isolation=True))
  def testBatchedFirstTimeStepAndOneStep(self, isolation):
    py_env = self._get_py_env(
        batch_py_env=True, isolation=isolation, batch_size=3)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
    self.assertEqual(tf_env.batch_size, 3)
    time_step_0 = tf_env.current_time_step()
    time_step_0_val = self.evaluate(time_step_0)

    self.assertAllEqual([ts.StepType.FIRST] * 3, time_step_0_val.step_type)
    self.assertAllEqual([0.0] * 3, time_step_0_val.reward)
    self.assertAllEqual([1.0] * 3, time_step_0_val.discount)
    self.assertAllEqual(np.array([0, 0, 0]), time_step_0_val.observation)
    for py_env in tf_env.pyenv.envs:
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
    for py_env in tf_env.pyenv.envs:
      self.assertEqual([1], py_env.actions_taken)
      self.assertEqual(1, py_env.resets)
      self.assertEqual(1, py_env.steps)
      self.assertEqual(0, py_env.episodes)

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testTwoStepsDependenceOnTheFirst(self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
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
    self.assertEqual([1, 2], get(tf_env.pyenv, 'actions_taken'))

  @parameterized.parameters(*COMMON_PARAMETERS)
  def testFirstObservationIsPreservedAfterTwoSteps(
      self, batch_py_env, isolation):
    py_env = self._get_py_env(batch_py_env, isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([1])
    next_time_step = tf_env.step(action)
    with tf.control_dependencies([next_time_step.step_type]):
      action = tf.constant([2])
    _, observation = self.evaluate([tf_env.step(action), time_step.observation])

    self.assertEqual(np.array([0]), observation)

  @parameterized.parameters(dict(isolation=False), dict(isolation=True))
  def testIsolation(self, isolation):
    py_env = self._get_py_env(batch_py_env=False, isolation=isolation)
    tf_env = tf_py_environment.TFPyEnvironment(py_env, isolation=isolation)
    last_env_thread = lambda: get(tf_env.pyenv, 'last_call_thread_id')
    local_thread = threading.current_thread().ident
    if isolation:
      self.assertNotEqual(local_thread, last_env_thread())
    else:
      self.assertEqual(local_thread, last_env_thread())

    # The remaining tests apply only to isolation == True
    if not isolation:
      return

    init_env_thread = last_env_thread()
    # Ensure that parallel computation does run in a thread different from the
    # one the pyenv was initialized in: that isolation forced execution on a
    # single dedicated threadpool.
    for _ in range(30):
      self.evaluate([tf_env.reset() for _ in range(16)])
      self.assertEqual(init_env_thread, last_env_thread())
      self.evaluate([tf_env.current_time_step() for _ in range(16)])
      self.assertEqual(init_env_thread, last_env_thread())
      self.evaluate([tf_env.step(tf.constant([1])) for _ in range(16)])
      self.assertEqual(init_env_thread, last_env_thread())

  def testRender(self):
    py_env = self._get_py_env(False, False, batch_size=None)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    img = self.evaluate(tf_env.render('rgb_array'))
    self.assertEqual(img.shape, (1, 4, 4, 3))
    self.assertEqual(img.dtype, np.uint8)
    img = self.evaluate(tf_env.render('human'))
    self.assertEqual(img.shape, (1, 4, 4, 3))
    self.assertEqual(img.dtype, np.uint8)
    img = self.evaluate(tf_env.render())  # defaults to rgb_array
    self.assertEqual(img.shape, (1, 4, 4, 3))
    self.assertEqual(img.dtype, np.uint8)

  def testRenderBatched(self):
    py_env = self._get_py_env(True, False, batch_size=3)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    img = self.evaluate(tf_env.render('rgb_array'))
    self.assertEqual(img.shape, (3, 4, 4, 3))
    self.assertEqual(img.dtype, np.uint8)
    img = self.evaluate(tf_env.render('human'))
    self.assertEqual(img.shape, (3, 4, 4, 3))
    self.assertEqual(img.dtype, np.uint8)
    img = self.evaluate(tf_env.render())  # defaults to rgb_array
    self.assertEqual(img.shape, (3, 4, 4, 3))
    self.assertEqual(img.dtype, np.uint8)

  def testOneStepNestedRewards(self):
    py_env = PYEnvironmentMockNestedRewards()
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant([1])
    time_step = self.evaluate(tf_env.step(action))

    self.assertAllEqual([ts.StepType.MID], time_step.step_type)
    self.assertAllEqual([1.], time_step.reward['reward'])
    self.assertAllEqual([2.], time_step.reward['constraint'])
    self.assertAllEqual([1.0], time_step.discount)
    self.assertAllEqual([1], time_step.observation)

  def testObservationsNotCached(self):
    py_env = suite_gym.load('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    time_step_1 = py_env.reset()
    time_step_2 = py_env.reset()
    # Make sure py env generates unique observations
    self.assertNotEqual(time_step_1.observation.tolist(),
                        time_step_2.observation.tolist())

    # Test tf_env also creates uniquee observations
    time_step_1 = tf_env.reset()
    time_step_2 = tf_env.reset()

    observation_1 = self.evaluate(time_step_1.observation).tolist()
    observation_2 = self.evaluate(time_step_2.observation).tolist()

    self.assertNotEqual(observation_1, observation_2)

  @parameterized.parameters(dict(autograph=True), dict(autograph=False))
  def testObservationsNotCachedWithTFFunction(self, autograph):
    py_env = suite_gym.load('CartPole-v1')
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    tf_env.reset = common.function(tf_env.reset, autograph=autograph)
    # Test tf_env also creates uniquee observations
    time_step_1 = tf_env.reset()
    time_step_2 = tf_env.reset()
    observation_1 = self.evaluate(time_step_1.observation)
    observation_2 = self.evaluate(time_step_2.observation)
    self.assertNotEqual(observation_1.tolist(), observation_2.tolist())

    # Check observation is not all 0.
    self.assertGreater(abs(sum(observation_2.flatten().tolist())), 0)


if __name__ == '__main__':
  tf.test.main()
