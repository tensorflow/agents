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

# Lint as: python3
"""Tests for tf_agents.agents.random.random_agent."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow.compat.v2 as tf

from tf_agents.agents.random import random_agent
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import test_utils

tf.enable_v2_behavior()


class RandomAgentTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(RandomAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

  def testCreateAgent(self):
    agent = random_agent.RandomAgent(
        self._time_step_spec,
        self._action_spec,)
    agent.initialize()

  def testTrain(self):
    # Define the train step counter.
    counter = common.create_variable('test_train_counter')
    agent = random_agent.RandomAgent(
        self._time_step_spec,
        self._action_spec,
        train_step_counter=counter,
        num_outer_dims=2)
    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)

    time_steps = ts.TimeStep(
        step_type=tf.constant([[1] * 3] * 2, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, (),
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)

    # Assert that counter starts out at zero.
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(0, self.evaluate(counter))

    agent.train(experience)

    # Now we should have one iteration.
    self.assertEqual(1, self.evaluate(counter))

  def testPolicy(self):
    agent = random_agent.RandomAgent(
        self._time_step_spec,
        self._action_spec,)
    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=1)
    action_step = agent.policy.action(time_steps)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    actions = self.evaluate(action_step.action)
    self.assertEqual(list(actions.shape), [1, 1])

if __name__ == '__main__':
  tf.test.main()
