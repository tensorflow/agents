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

"""Tests for trajectories.time_step."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils


class TimeStepTest(tf.test.TestCase):

  def testRestart(self):
    observation = -1
    time_step = ts.restart(observation)

    self.assertEqual(ts.StepType.FIRST, time_step.step_type)
    self.assertEqual(-1, time_step.observation)
    self.assertEqual(0.0, time_step.reward)
    self.assertEqual(1.0, time_step.discount)

  def testTransition(self):
    observation = -1
    reward = 2.0
    discount = 1.0
    time_step = ts.transition(observation, reward, discount)

    self.assertEqual(ts.StepType.MID, time_step.step_type)
    self.assertEqual(-1, time_step.observation)
    self.assertEqual(2.0, time_step.reward)
    self.assertEqual(1.0, time_step.discount)

  def testTermination(self):
    observation = -1
    reward = 2.0
    time_step = ts.termination(observation, reward)

    self.assertEqual(ts.StepType.LAST, time_step.step_type)
    self.assertEqual(-1, time_step.observation)
    self.assertEqual(2.0, time_step.reward)
    self.assertEqual(0.0, time_step.discount)

  def testTruncation(self):
    observation = -1
    reward = 2.0
    discount = 1.0
    time_step = ts.truncation(observation, reward, discount)

    self.assertEqual(ts.StepType.LAST, time_step.step_type)
    self.assertEqual(-1, time_step.observation)
    self.assertEqual(2.0, time_step.reward)
    self.assertEqual(1.0, time_step.discount)

  def testRestartIsFirst(self):
    observation = -1
    time_step = ts.restart(observation)
    self.assertTrue(time_step.is_first())

  def testTransitionIsMid(self):
    observation = -1
    reward = 2.0
    time_step = ts.transition(observation, reward)
    self.assertTrue(time_step.is_mid())

  def testTerminationIsLast(self):
    observation = -1
    reward = 2.0
    time_step = ts.termination(observation, reward)
    self.assertTrue(time_step.is_last())

  def testLastNumpy(self):
    observation = -1
    reward = 2.0
    discount = 1.0
    time_step = ts.TimeStep(np.asarray(ts.StepType.LAST),
                            np.asarray(reward),
                            np.asarray(discount),
                            np.asarray(observation))
    self.assertTrue(time_step.is_last())
    self.assertEqual(ts.StepType.LAST, time_step.step_type)
    self.assertEqual(-1, time_step.observation)
    self.assertEqual(2.0, time_step.reward)
    self.assertEqual(1.0, time_step.discount)

  def testRestartBatched(self):
    observation = np.array([[-1], [-1]])
    time_step = ts.restart(observation, batch_size=2)

    self.assertItemsEqual([ts.StepType.FIRST] * 2, time_step.step_type)
    self.assertItemsEqual(observation, time_step.observation)
    self.assertItemsEqual([0.0, 0.0], time_step.reward)
    self.assertItemsEqual([1.0, 1.0], time_step.discount)

  def testTransitionBatched(self):
    observation = np.array([[-1], [-1]])
    reward = np.array([2., 2.])
    discount = np.array([1., 1.])
    time_step = ts.transition(observation, reward, discount)

    self.assertItemsEqual([ts.StepType.MID] * 2, time_step.step_type)
    self.assertItemsEqual(observation, time_step.observation)
    self.assertItemsEqual(reward, time_step.reward)
    self.assertItemsEqual(discount, time_step.discount)

  def testTerminationBatched(self):
    observation = np.array([[-1], [-1]])
    reward = np.array([2., 2.])
    time_step = ts.termination(observation, reward)

    self.assertItemsEqual([ts.StepType.LAST] * 2, time_step.step_type)
    self.assertItemsEqual(observation, time_step.observation)
    self.assertItemsEqual(reward, time_step.reward)
    self.assertItemsEqual([0., 0.], time_step.discount)

  def testTruncationBatched(self):
    observation = np.array([[-1], [-1]])
    reward = np.array([2., 2.])
    discount = np.array([1., 1.])
    time_step = ts.truncation(observation, reward, discount)

    self.assertItemsEqual([ts.StepType.LAST] * 2, time_step.step_type)
    self.assertItemsEqual(observation, time_step.observation)
    self.assertItemsEqual(reward, time_step.reward)
    self.assertItemsEqual(discount, time_step.discount)


class TimeStepSpecTest(tf.test.TestCase):

  def testObservationSpec(self):
    observation_spec = [array_spec.ArraySpec((1, 2, 3), np.int32, 'obs1'),
                        array_spec.ArraySpec((1, 2, 3), np.float32, 'obs2')]
    time_step_spec = ts.time_step_spec(observation_spec)

    self.assertEqual(time_step_spec.observation, observation_spec)
    self.assertEqual(time_step_spec.step_type,
                     array_spec.ArraySpec([], np.int32, name='step_type'))
    self.assertEqual(time_step_spec.reward,
                     array_spec.ArraySpec([], np.float32, name='reward'))
    self.assertEqual(time_step_spec.discount,
                     array_spec.BoundedArraySpec([], np.float32,
                                                 minimum=0.0, maximum=1.0,
                                                 name='discount'))


class TFTimeStepTest(tf.test.TestCase):

  def testRestart(self):
    observation = tf.constant(-1)
    time_step = ts.restart(observation)
    time_step_ = self.evaluate(time_step)
    self.assertEqual(ts.StepType.FIRST, time_step_.step_type)
    self.assertEqual(-1, time_step_.observation)
    self.assertEqual(0.0, time_step_.reward)
    self.assertEqual(1.0, time_step_.discount)

  def testBatchRestart(self):
    obs_spec = [tensor_spec.TensorSpec([2], tf.float32)]
    time_step_spec = ts.time_step_spec(obs_spec)
    observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
    time_steps = ts.restart(observations, 2)
    time_step_batched = nest_utils.is_batched_nested_tensors(
        time_steps, time_step_spec)
    self.assertTrue(time_step_batched)

  def testTransition(self):
    observation = tf.constant(-1)
    reward = tf.constant(2.0)
    discount = tf.constant(1.0)
    time_step = ts.transition(observation, reward, discount)
    time_step_ = self.evaluate(time_step)
    self.assertEqual(ts.StepType.MID, time_step_.step_type)
    self.assertEqual(-1, time_step_.observation)
    self.assertEqual(2.0, time_step_.reward)
    self.assertEqual(1.0, time_step_.discount)

  def testTermination(self):
    observation = tf.constant(-1)
    reward = tf.constant(2.0)
    time_step = ts.termination(observation, reward)
    time_step_ = self.evaluate(time_step)
    self.assertEqual(ts.StepType.LAST, time_step_.step_type)
    self.assertEqual(-1, time_step_.observation)
    self.assertEqual(2.0, time_step_.reward)
    self.assertEqual(0.0, time_step_.discount)

  def testTruncation(self):
    observation = tf.constant(-1)
    reward = tf.constant(2.0)
    discount = tf.constant(1.0)
    time_step = ts.truncation(observation, reward, discount)
    time_step_ = self.evaluate(time_step)
    self.assertEqual(ts.StepType.LAST, time_step_.step_type)
    self.assertEqual(-1, time_step_.observation)
    self.assertEqual(2.0, time_step_.reward)
    self.assertEqual(1.0, time_step_.discount)

  def testRestartIsFirst(self):
    observation = tf.constant(-1)
    time_step = ts.restart(observation)
    is_first = time_step.is_first()
    self.assertEqual(True, self.evaluate(is_first))

  def testTransitionIsMid(self):
    observation = tf.constant(-1)
    reward = tf.constant(2.0)
    time_step = ts.transition(observation, reward)
    is_mid = time_step.is_mid()
    self.assertEqual(True, self.evaluate(is_mid))

  def testTerminationIsLast(self):
    observation = tf.constant(-1)
    reward = tf.constant(2.0)
    time_step = ts.termination(observation, reward)
    is_last = time_step.is_last()
    self.assertEqual(True, self.evaluate(is_last))


class TFTimeStepSpecTest(tf.test.TestCase):

  def testObservationSpec(self):
    observation_spec = [tensor_spec.TensorSpec((1, 2, 3), tf.int32, 'obs1'),
                        tensor_spec.TensorSpec((1, 2, 3), tf.float32, 'obs2')]
    time_step_spec = ts.time_step_spec(observation_spec)

    self.assertEqual(time_step_spec.observation, observation_spec)
    self.assertEqual(time_step_spec.step_type,
                     tensor_spec.TensorSpec([], tf.int32, name='step_type'))
    self.assertEqual(time_step_spec.reward,
                     tensor_spec.TensorSpec([], tf.float32, name='reward'))
    self.assertEqual(time_step_spec.discount,
                     tensor_spec.BoundedTensorSpec([], tf.float32,
                                                   minimum=0.0, maximum=1.0,
                                                   name='discount'))


if __name__ == '__main__':
  tf.test.main()
