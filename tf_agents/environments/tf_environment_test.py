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

"""Tests for environments.tf_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST


class TFEnvironmentMock(tf_environment.TFEnvironment):
  """MockTFEnvironment.

  Stores all actions taken in `actions_taken`. The returned values are:

  step: FIRST, 1., 0., [0]
  step: MID, 1., 0., [1]
  step: LAST, 0., 1. [2]
  ...repeated
  """

  def __init__(self, initial_state=0, dtype=tf.int64, scope='TFEnviroment'):
    self._dtype = dtype
    self._scope = scope
    self._initial_state = tf.cast(initial_state, dtype=self._dtype)
    observation_spec = specs.TensorSpec([1], self._dtype, 'observation')
    action_spec = specs.BoundedTensorSpec([], tf.int32, minimum=0, maximum=10)
    time_step_spec = ts.time_step_spec(observation_spec)
    super(TFEnvironmentMock, self).__init__(time_step_spec, action_spec)
    self._state = common.create_variable('state', initial_state,
                                         dtype=self._dtype)
    self.steps = common.create_variable('steps', 0)
    self.episodes = common.create_variable('episodes', 0)
    self.resets = common.create_variable('resets', 0)

  def _current_time_step(self):
    def first():
      return (tf.constant(FIRST, dtype=tf.int32),
              tf.constant(0.0, dtype=tf.float32),
              tf.constant(1.0, dtype=tf.float32))
    def mid():
      return (tf.constant(MID, dtype=tf.int32),
              tf.constant(0.0, dtype=tf.float32),
              tf.constant(1.0, dtype=tf.float32))
    def last():
      return (tf.constant(LAST, dtype=tf.int32),
              tf.constant(1.0, dtype=tf.float32),
              tf.constant(0.0, dtype=tf.float32))
    state_value = tf.math.mod(self._state.value(), 3)
    step_type, reward, discount = tf.case(
        {tf.equal(state_value, FIRST): first,
         tf.equal(state_value, MID): mid,
         tf.equal(state_value, LAST): last},
        exclusive=True, strict=True)
    return ts.TimeStep(step_type, reward, discount, state_value)

  def _reset(self):
    increase_resets = self.resets.assign_add(1)
    with tf.control_dependencies([increase_resets]):
      reset_op = self._state.assign(self._initial_state)
    with tf.control_dependencies([reset_op]):
      time_step = self.current_time_step()
    return time_step

  def _step(self, action):
    action = tf.convert_to_tensor(value=action)
    with tf.control_dependencies(tf.nest.flatten(action)):
      state_assign = self._state.assign_add(1)
    with tf.control_dependencies([state_assign]):
      state_value = self._state.value()
      increase_steps = tf.cond(
          pred=tf.equal(tf.math.mod(state_value, 3), FIRST),
          true_fn=self.steps.value,
          false_fn=lambda: self.steps.assign_add(1))
      increase_episodes = tf.cond(
          pred=tf.equal(tf.math.mod(state_value, 3), LAST),
          true_fn=lambda: self.episodes.assign_add(1),
          false_fn=self.episodes.value)
    with tf.control_dependencies([increase_steps, increase_episodes]):
      return self.current_time_step()


class TFEnvironmentTest(tf.test.TestCase):

  def testResetOp(self):
    tf_env = TFEnvironmentMock()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf_env.reset())
    self.assertEqual(1, self.evaluate(tf_env.resets))
    self.assertEqual(0, self.evaluate(tf_env.steps))
    self.assertEqual(0, self.evaluate(tf_env.episodes))

  def testMultipleReset(self):
    tf_env = TFEnvironmentMock()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf_env.reset())
    self.assertEqual(1, self.evaluate(tf_env.resets))
    self.evaluate(tf_env.reset())
    self.assertEqual(2, self.evaluate(tf_env.resets))
    self.evaluate(tf_env.reset())
    self.assertEqual(3, self.evaluate(tf_env.resets))
    self.assertEqual(0, self.evaluate(tf_env.steps))
    self.assertEqual(0, self.evaluate(tf_env.episodes))

  def testFirstTimeStep(self):
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    time_step = self.evaluate(time_step)
    self.assertEqual(FIRST, time_step.step_type)
    self.assertEqual(0.0, time_step.reward)
    self.assertEqual(1.0, time_step.discount)
    self.assertEqual([0], time_step.observation)
    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(0, self.evaluate(tf_env.steps))
    self.assertEqual(0, self.evaluate(tf_env.episodes))

  def testFirstStepState(self):
    tf_env = TFEnvironmentMock()
    tf_env.current_time_step()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(0, self.evaluate(tf_env.steps))
    self.assertEqual(0, self.evaluate(tf_env.episodes))

  def testOneStep(self):
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant(1)
    next_time_step = tf_env.step(action)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    time_step, next_time_step = self.evaluate([time_step, next_time_step])

    self.assertEqual(FIRST, time_step.step_type)
    self.assertEqual(0., time_step.reward)
    self.assertEqual(1.0, time_step.discount)
    self.assertEqual([0], time_step.observation)

    self.assertEqual(MID, next_time_step.step_type)
    self.assertEqual(0., next_time_step.reward)
    self.assertEqual(1.0, next_time_step.discount)
    self.assertEqual([1], next_time_step.observation)

    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(1, self.evaluate(tf_env.steps))
    self.assertEqual(0, self.evaluate(tf_env.episodes))

  def testCurrentStep(self):
    if tf.executing_eagerly():
      self.skipTest('b/123881612')
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant(1)
    next_time_step = tf_env.step(action)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    time_step_np, next_time_step_np = self.evaluate([time_step, next_time_step])
    self.assertEqual(FIRST, time_step_np.step_type)
    self.assertEqual(0., time_step_np.reward)
    self.assertEqual(1.0, time_step_np.discount)
    self.assertEqual([0], time_step_np.observation)

    self.assertEqual(MID, next_time_step_np.step_type)
    self.assertEqual(0., next_time_step_np.reward)
    self.assertEqual(1.0, next_time_step_np.discount)
    self.assertEqual([1], next_time_step_np.observation)

    time_step_np, next_time_step_np = self.evaluate([time_step, next_time_step])
    self.assertEqual(MID, time_step_np.step_type)
    self.assertEqual(0., time_step_np.reward)
    self.assertEqual(1.0, time_step_np.discount)
    self.assertEqual([1], time_step_np.observation)

    self.assertEqual(LAST, next_time_step_np.step_type)
    self.assertEqual(1., next_time_step_np.reward)
    self.assertEqual(0.0, next_time_step_np.discount)
    self.assertEqual([2], next_time_step_np.observation)

    time_step_np = self.evaluate(time_step)
    self.assertEqual(LAST, time_step_np.step_type)
    self.assertEqual(1., time_step_np.reward)
    self.assertEqual(0.0, time_step_np.discount)
    self.assertEqual([2], time_step_np.observation)

    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(2, self.evaluate(tf_env.steps))
    self.assertEqual(1, self.evaluate(tf_env.episodes))

  def testTwoStepsDependenceOnTheFirst(self):
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant(1)
    time_step = tf_env.step(action)
    with tf.control_dependencies([time_step.step_type]):
      action = tf.constant(2)
    time_step = self.evaluate(tf_env.step(action))
    self.assertEqual(LAST, time_step.step_type)
    self.assertEqual(1., time_step.reward)
    self.assertEqual(0.0, time_step.discount)
    self.assertEqual([2], time_step.observation)
    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(2, self.evaluate(tf_env.steps))
    self.assertEqual(1, self.evaluate(tf_env.episodes))

  def testAutoReset(self):
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    with tf.control_dependencies([time_step.step_type]):
      time_step = tf_env.step(1)
    with tf.control_dependencies([time_step.step_type]):
      time_step = tf_env.step(2)
    with tf.control_dependencies([time_step.step_type]):
      time_step = self.evaluate(tf_env.step(3))
    self.assertEqual(FIRST, time_step.step_type)
    self.assertEqual(0.0, time_step.reward)
    self.assertEqual(1.0, time_step.discount)
    self.assertEqual([0], time_step.observation)
    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(2, self.evaluate(tf_env.steps))
    self.assertEqual(1, self.evaluate(tf_env.episodes))

  def testFirstObservationIsPreservedAfterTwoSteps(self):
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    time_step_np = self.evaluate(time_step)
    self.assertEqual([0], time_step_np.observation)
    time_step = tf_env.step(1)
    with tf.control_dependencies([time_step.step_type]):
      next_time_step = tf_env.step(2)

    observation_np, _ = self.evaluate([time_step.observation, next_time_step])

    self.assertEqual([1], observation_np)

  def testRandomAction(self):
    tf_env = TFEnvironmentMock()
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tf.random.uniform([], minval=0, maxval=10, dtype=tf.int32)
    next_time_step = tf_env.step(action)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    [time_step_np, next_time_step_np] = self.evaluate(
        [time_step, next_time_step])
    self.assertEqual([0], time_step_np.observation)
    self.assertEqual([1], next_time_step_np.observation)
    self.assertEqual(0, self.evaluate(tf_env.resets))
    self.assertEqual(1, self.evaluate(tf_env.steps))
    self.assertEqual(0, self.evaluate(tf_env.episodes))

  def testRunEpisode(self):
    tf_env = TFEnvironmentMock()
    c = lambda t: tf.logical_not(t.is_last())
    body = lambda t: [tf_env.step(t.observation)]

    @common.function
    def run_episode():
      time_step = tf_env.reset()
      return tf.while_loop(cond=c, body=body, loop_vars=[time_step])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    [final_time_step_np] = self.evaluate(run_episode())
    self.assertEqual([2], final_time_step_np.step_type)
    self.assertEqual([2], final_time_step_np.observation)
    self.assertEqual(1, self.evaluate(tf_env.resets))
    self.assertEqual(2, self.evaluate(tf_env.steps))
    self.assertEqual(1, self.evaluate(tf_env.episodes))
    # Run another episode.
    [final_time_step_np] = self.evaluate(run_episode())
    self.assertEqual([2], final_time_step_np.step_type)
    self.assertEqual([2], final_time_step_np.observation)
    self.assertEqual(2, self.evaluate(tf_env.resets))
    self.assertEqual(4, self.evaluate(tf_env.steps))
    self.assertEqual(2, self.evaluate(tf_env.episodes))


if __name__ == '__main__':
  tf.test.main()
