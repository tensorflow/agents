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

"""Tests for tf_agents.environments.random_tf_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import random_tf_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class RandomTFEnvironmentTest(test_utils.TestCase):

  def setUp(self):
    self.observation_spec = tensor_spec.TensorSpec((2, 3), tf.float32)
    self.reward_spec = tensor_spec.TensorSpec((2,), tf.float32)
    self.time_step_spec = ts.time_step_spec(self.observation_spec,
                                            reward_spec=self.reward_spec)
    self.action_spec = tensor_spec.TensorSpec((2,), tf.float32)
    self.random_env = random_tf_environment.RandomTFEnvironment(
        self.time_step_spec, self.action_spec)

  def test_state_saved_after_reset(self):
    initial_time_step = self.evaluate(self.random_env.reset())
    current_time_step = self.evaluate(self.random_env.current_time_step())

    np.testing.assert_almost_equal(initial_time_step.step_type,
                                   current_time_step.step_type)
    np.testing.assert_almost_equal(initial_time_step.observation,
                                   current_time_step.observation)
    np.testing.assert_almost_equal(initial_time_step.discount,
                                   current_time_step.discount)
    np.testing.assert_almost_equal(initial_time_step.reward,
                                   current_time_step.reward)

  def test_state_saved_after_step(self):
    self.evaluate(self.random_env.reset())
    random_action = self.evaluate(
        tensor_spec.sample_spec_nest(self.action_spec, outer_dims=(1,)))

    expected_time_step = self.evaluate(self.random_env.step(random_action))
    current_time_step = self.evaluate(self.random_env.current_time_step())

    np.testing.assert_almost_equal(expected_time_step.step_type,
                                   current_time_step.step_type)
    np.testing.assert_almost_equal(expected_time_step.observation,
                                   current_time_step.observation)
    np.testing.assert_almost_equal(expected_time_step.discount,
                                   current_time_step.discount)
    np.testing.assert_almost_equal(expected_time_step.reward,
                                   current_time_step.reward)

  def test_auto_reset(self):
    time_step = self.evaluate(self.random_env.reset())
    random_action = self.evaluate(
        tensor_spec.sample_spec_nest(self.action_spec, outer_dims=(1,)))

    attempts = 0

    # With a 1/10 chance of resetting on each step, the probability of failure
    # after 500 attempts should be 0.9^500, roughly 1e-23. If we miss more than
    # 500 attempts, we can safely assume the test is broken.
    while not time_step.is_last() and attempts < 500:
      time_step = self.evaluate(self.random_env.step(random_action))
      attempts += 1

    self.assertLess(attempts, 500)
    self.assertTrue(time_step.is_last())

    current_time_step = self.evaluate(self.random_env.current_time_step())
    self.assertTrue(current_time_step.is_last())

    first_time_step = self.evaluate(self.random_env.step(random_action))
    self.assertTrue(first_time_step.is_first())

  def test_step_batched_action(self):
    self.evaluate(self.random_env.reset())
    random_action = self.evaluate(
        tensor_spec.sample_spec_nest(self.action_spec, outer_dims=(5,)))

    self.evaluate(self.random_env.step(random_action))


if __name__ == '__main__':
  tf.test.main()
