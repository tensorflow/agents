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

"""Tests for tf_agents.bandits.environments.classification_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from absl.testing.absltest import mock
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.environments import classification_environment as ce

tfd = tfp.distributions


def deterministic_reward_distribution(reward_table):
  """Returns a deterministic distribution centered at `reward_table`."""
  return tfd.Independent(tfd.Deterministic(loc=reward_table),
                         reinterpreted_batch_ndims=2)


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

  def testShuffle(self):
    """Test that dataset is being shuffled when asked."""
    # Reward of 1 is given if action == (context % 3)
    context = tf.reshape(tf.range(128), shape=[128, 1])
    labels = tf.math.mod(context, 3)
    batch_size = 32
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (context, labels)).repeat().shuffle(4 * batch_size))
    reward_distribution = deterministic_reward_distribution(tf.eye(3))

    # Note - shuffle should hapen *first* in call chain, so this
    # test will fail if shuffle is called e.g. after batch or prefetch.
    dataset.shuffle = mock.Mock(spec=dataset.shuffle,
                                side_effect=dataset.shuffle)
    ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size)
    dataset.shuffle.assert_not_called()
    ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size, shuffle_buffer_size=3, seed=7)
    dataset.shuffle.assert_called_with(
        buffer_size=3, reshuffle_each_iteration=True, seed=7)

  @mock.patch('tf_agents.bandits.environments.classification_environment'+
              '.eager_utils.dataset_iterator')
  def testPrefetch(self, mock_dataset_iterator):
    """Test that dataset is being prefetched when asked."""
    mock_dataset_iterator.return_value = 'mock_iterator_result'
    # Reward of 1 is given if action == (context % 3)
    context = tf.reshape(tf.range(128), shape=[128, 1])
    labels = tf.math.mod(context, 3)
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices((context, labels))
    reward_distribution = deterministic_reward_distribution(tf.eye(3))

    # Operation order should be batch() then prefetch(), have to jump
    # through a couple hoops to get this sequence tested correctly.

    # Save dataset.prefetch in temp mock_prefetch, return batched dataset to
    # make down-stream logic work correctly with batch dimensions.
    batched_dataset = dataset.batch(batch_size)
    mock_prefetch = mock.Mock(spec=dataset.prefetch,
                              return_value=batched_dataset)
    # Replace dataset.batch with mock batch that returns original dataset,
    # in order to make mocking out it's prefetch call easier.
    dataset.batch = mock.Mock(spec=batched_dataset,
                              return_value=batched_dataset)
    # Replace dataset.prefetch with mock_prefetch.
    batched_dataset.prefetch = mock_prefetch
    env = ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size, repeat_dataset=False)
    dataset.batch.assert_called_with(batch_size, drop_remainder=True)
    batched_dataset.prefetch.assert_not_called()
    mock_dataset_iterator.assert_called_with(batched_dataset)
    self.assertEqual(env._data_iterator, 'mock_iterator_result')
    env = ce.ClassificationBanditEnvironment(
        dataset, reward_distribution, batch_size, repeat_dataset=False,
        prefetch_size=3)
    dataset.batch.assert_called_with(batch_size, drop_remainder=True)
    batched_dataset.prefetch.assert_called_with(3)
    mock_dataset_iterator.assert_called_with(batched_dataset)
    self.assertEqual(env._data_iterator, 'mock_iterator_result')


if __name__ == '__main__':
  tf.test.main()
