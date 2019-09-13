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

"""Tests for tf_agents.bandits.environments.classification_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.bandits.environments import classification_environment as ce
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import  # TF internal

tfd = tfp.distributions


def deterministic_reward_distribution(reward_table):
  """Returns a deterministic distribution centered at `reward_table`."""
  return tfd.Independent(tfd.Deterministic(loc=reward_table),
                         reinterpreted_batch_ndims=2)


@test_util.run_all_in_graph_and_eager_modes
class ClassificationEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_3x2x3',
           tbl=[[[0, 1, 2],
                 [3, 4, 5]],
                [[6, 7, 8],
                 [9, 10, 11]],
                [[12, 13, 14],
                 [15, 16, 17]]],
           row=[0, 1, 1],
           col=[0, 2, 0],
           expected=[0, 11, 15]),
      )
  def testBatchedTableLookup(self, tbl, row, col, expected):
    actual = ce._batched_table_lookup(tbl, row, col)
    np.testing.assert_almost_equal(expected, self.evaluate(actual))

  @parameterized.named_parameters(
      dict(
          testcase_name='_scalar_batch_1',
          context=np.array([[0], [1]]),
          labels=np.array([0, 1]),
          batch_size=1),
      dict(
          testcase_name='_multi_dim_batch_23',
          context=np.arange(100).reshape(10, 10),
          labels=np.arange(10),
          batch_size=23),
  )
  def testObservationShapeAndValue(self, context, labels, batch_size):
    """Test that observations have correct shape and values from `context`."""
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (context, labels)).repeat().shuffle(4 * batch_size))
    # Rewards of 1. is given when action == label
    reward_distribution = deterministic_reward_distribution(
        tf.eye(len(set(labels))))
    env = ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size)
    expected_observation_shape = [batch_size] + list(context.shape[1:])
    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(100):
      observation = self.evaluate(env.reset().observation)
      np.testing.assert_array_equal(observation.shape,
                                    expected_observation_shape)
      for o in observation:
        self.assertIn(o, context)

  def testReturnsCorrectRewards(self):
    """Test that rewards are being returned correctly for a simple case."""
    # Reward of 1 is given if action == (context % 3)
    context = tf.reshape(tf.range(128), shape=[128, 1])
    labels = tf.math.mod(context, 3)
    batch_size = 32
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (context, labels)).repeat().shuffle(4 * batch_size))
    reward_distribution = deterministic_reward_distribution(tf.eye(3))
    env = ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(10):
      # Take the 'correct' action
      observation = env.reset().observation
      action = tf.math.mod(observation, 3)
      reward = env.step(action).reward
      np.testing.assert_almost_equal(self.evaluate(reward),
                                     self.evaluate(tf.ones_like(reward)))

    for _ in range(10):
      # Take the 'incorrect' action
      observation = env.reset().observation
      action = tf.math.mod(observation + 1, 3)
      reward = env.step(action).reward
      np.testing.assert_almost_equal(self.evaluate(reward),
                                     self.evaluate(tf.zeros_like(reward)))

  def testPreviousLabelIsSetCorrectly(self):
    """Test that the previous label is set correctly for a simple case."""
    # Reward of 1 is given if action == (context % 3)
    context = tf.reshape(tf.range(128), shape=[128, 1])
    labels = tf.math.mod(context, 3)
    batch_size = 4
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (context, labels)).repeat().shuffle(4 * batch_size))
    reward_distribution = deterministic_reward_distribution(tf.eye(3))
    env = ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size)
    self.evaluate(tf.compat.v1.global_variables_initializer())

    time_step = env.reset()
    time_step_label = tf.squeeze(tf.math.mod(time_step.observation, 3))
    action = tf.math.mod(time_step.observation, 3)
    next_time_step = env.step(action)
    next_time_step_label = tf.squeeze(
        tf.math.mod(next_time_step.observation, 3))

    if tf.executing_eagerly():
      np.testing.assert_almost_equal(
          self.evaluate(time_step_label),
          self.evaluate(env._previous_label))
      np.testing.assert_almost_equal(
          self.evaluate(next_time_step_label),
          self.evaluate(env._current_label))
    else:
      with self.cached_session() as sess:
        time_step_label_value, next_time_step_label_value = (
            sess.run([time_step_label, next_time_step_label]))

        previous_label_value = self.evaluate(env._previous_label)
        np.testing.assert_almost_equal(
            time_step_label_value, previous_label_value)
        current_label_value = self.evaluate(env._current_label)
        np.testing.assert_almost_equal(
            next_time_step_label_value,
            current_label_value)


if __name__ == '__main__':
  tf.test.main()
