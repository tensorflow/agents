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

"""Tests for episodic_replay_buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import test_utils

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.ops import list_ops  # TF internal
# pylint:enable=g-direct-tensorflow-import


# Shorthand for converting arrays to np.arrays.
_a = np.asarray


def sample_as_dataset(replay_buffer,
                      num_steps=None,
                      batch_size=None):
  ds = replay_buffer.as_dataset(
      num_steps=num_steps,
      sample_batch_size=batch_size)
  # Note, we don't use tf.contrib.data.get_single_element because the dataset
  # contains more than one element.
  if tf.executing_eagerly():
    itr = iter(ds)
    return next(itr)
  else:
    itr = tf.compat.v1.data.make_initializable_iterator(ds)
    with tf.control_dependencies([itr.initializer]):
      return itr.get_next()


def iterator_from_dataset(
    replay_buffer,
    num_steps=None,
    batch_size=None,
    single_deterministic_pass=None,
    session=None):
  ds = replay_buffer.as_dataset(
      num_steps=num_steps,
      sample_batch_size=batch_size,
      single_deterministic_pass=single_deterministic_pass)

  if tf.executing_eagerly():
    for value in iter(ds):
      yield tf.nest.map_structure(lambda v: v.numpy(), value)
  else:
    itr = tf.compat.v1.data.make_initializable_iterator(ds)
    gn = itr.get_next()
    initialized = [False]
    while True:
      if not initialized[0]:
        session.run(itr.initializer)
        initialized[0] = True
      yield session.run(gn)


# We access a lot of protected methods on replay_buffer to test them.
# pylint: disable=protected-access


class EpisodicReplayBufferTest(test_utils.TestCase, parameterized.TestCase):

  def _assertContains(self, list1, list2):
    self.assertTrue(test_utils.contains(list1, list2))

  def _assertNestedCloseness(self, closeness_fn, expected, actual):
    tf.nest.map_structure(closeness_fn, expected, actual)

  def testCreateEpisodeId(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)

    episode_ids = replay_buffer.create_episode_ids()

    self.assertEqual(self.evaluate(episode_ids), -1)

  def testCreateBatchEpisodeIds(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=5)

    episodes_0 = replay_buffer.create_episode_ids(2)
    episodes_1 = replay_buffer.create_episode_ids(3)

    self.assertAllEqual([-1] * 2, self.evaluate(episodes_0))
    self.assertAllEqual([-1] * 3, self.evaluate(episodes_1))

  def testCreateTooManyBatchEpisodeIdsRaisesError(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)

    with self.assertRaisesRegexp(
        ValueError, 'Buffer cannot create episode_ids when '
        'num_episodes 3 > capacity 2.'):
      replay_buffer.create_episode_ids(num_episodes=3)

  def testGetEpisodeId(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)
    episode_id = replay_buffer.create_episode_ids()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertEqual(self.evaluate(episode_id), -1)

    # episode_id should be updated when begin_episode is True.
    new_episode_id_0 = replay_buffer._get_episode_id(
        episode_id, begin_episode=True)

    # episode_id should not be updated when begin_episode is False and the input
    # episode_id is not negative.
    new_episode_id_1 = replay_buffer._get_episode_id(
        new_episode_id_0, begin_episode=False)

    # episode_id should be updated when begin_episode is True even when the
    # input episode_id is not negative.
    new_episode_id_2 = replay_buffer._get_episode_id(
        new_episode_id_1, begin_episode=True)

    (new_episode_id_0_value,
     new_episode_id_1_value,
     new_episode_id_2_value) = self.evaluate(
         (new_episode_id_0, new_episode_id_1, new_episode_id_2))

    self.assertEqual(new_episode_id_0_value, 0)
    self.assertEqual(new_episode_id_1_value, 0)
    self.assertEqual(new_episode_id_2_value, 1)

  def testGetBatchEpisodeIds(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=5)
    episode_ids = [replay_buffer.create_episode_ids(num_episodes=3)]
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    expected_ids = [[-1, -1, -1]]

    # batch_episode_ids should be updated the first time regardless of
    # begin_episode.
    episode_ids.append(replay_buffer._get_batch_episode_ids(
        episode_ids[-1], begin_episode=False))
    expected_ids.append([0, 1, 2])

    # batch_episode_ids should not be updated when begin_episode is False.
    episode_ids.append(replay_buffer._get_batch_episode_ids(
        episode_ids[-1], begin_episode=False))
    expected_ids.append([0, 1, 2])

    episode_ids.append(replay_buffer._get_batch_episode_ids(
        episode_ids[-1], begin_episode=[False, False, False]))
    expected_ids.append([0, 1, 2])

    # batch_episode_ids should be updated only when begin_episode is True.
    episode_ids.append(replay_buffer._get_batch_episode_ids(
        episode_ids[-1], begin_episode=[True, False, False]))
    expected_ids.append([3, 1, 2])

    episode_ids.append(replay_buffer._get_batch_episode_ids(
        episode_ids[-1], begin_episode=[False, True, False]))
    expected_ids.append([3, 4, 2])

    episode_ids.append(replay_buffer._get_batch_episode_ids(
        episode_ids[-1], begin_episode=[False, True, True]))
    expected_ids.append([3, 5, 6])

    self.assertAllEqual(expected_ids, self.evaluate(episode_ids))

  def testGetEpisodeIdBeginEpisodeFalse(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)
    episode_id = replay_buffer.create_episode_ids()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertEqual(self.evaluate(episode_id), -1)

    # episode_id should be updated regardless begin_episode being False since
    # the initial value of the episode_id_var is not valid (id >= 0).
    episode_id_0 = replay_buffer._get_episode_id(
        episode_id, begin_episode=False)
    expected_id_0 = 0

    # episode_id should be updated because begin_episode is True.
    episode_id_1 = replay_buffer._get_episode_id(
        episode_id_0, begin_episode=True)
    expected_id_1 = 1

    # episode_id should not be updated because begin_episode is False.
    episode_id_2 = replay_buffer._get_episode_id(
        episode_id_1, begin_episode=False)
    expected_id_2 = 1

    self.assertAllEqual(
        (expected_id_0, expected_id_1, expected_id_2),
        self.evaluate(
            (episode_id_0, episode_id_1, episode_id_2)))

  def testGetTwoEpisodeId(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)
    episode_id_0 = replay_buffer.create_episode_ids()
    episode_id_1 = replay_buffer.create_episode_ids()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertEqual(self.evaluate(episode_id_0), -1)
    self.assertEqual(self.evaluate(episode_id_1), -1)

    # episode_id will be updated regardless of begin_episode being False since
    # it doesn't have a valid value >= 0.
    episode_id_0 = replay_buffer._get_episode_id(
        replay_buffer.create_episode_ids())
    with tf.control_dependencies([episode_id_0]):
      episode_id_1 = replay_buffer._get_episode_id(
          replay_buffer.create_episode_ids())
    expected_id_0 = 0
    expected_id_1 = 1

    with tf.control_dependencies([episode_id_0, episode_id_1]):
      # Now episode_id should be updated only when begin_episode is True.
      episode_id_0_next = replay_buffer._get_episode_id(
          episode_id_0, begin_episode=False)
      with tf.control_dependencies([episode_id_0_next]):
        episode_id_1_next = replay_buffer._get_episode_id(
            episode_id_1, begin_episode=True)
      expected_id_0_next = 0
      expected_id_1_next = 2

    self.assertEqual(
        (expected_id_0, expected_id_1,
         expected_id_0_next, expected_id_1_next),
        self.evaluate((
            episode_id_0, episode_id_1,
            episode_id_0_next, episode_id_1_next)))

  def testMaybeEndEpisode(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)
    episode_id = replay_buffer.create_episode_ids()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    episode_id = replay_buffer._get_episode_id(episode_id)
    completed = replay_buffer._maybe_end_episode(episode_id)
    with tf.control_dependencies([completed]):
      completed_episodes = replay_buffer._completed_episodes()

    episode_id_value, completed_value, completed_episodes_value = self.evaluate(
        (episode_id, completed, completed_episodes))
    self.assertEqual(0, episode_id_value)
    self.assertEqual(False, completed_value)
    self.assertEmpty(completed_episodes_value)

    # Mark episode as completed.
    completed = replay_buffer._maybe_end_episode(0, end_episode=True)
    with tf.control_dependencies([completed]):
      completed_episodes = replay_buffer._completed_episodes()
    completed_value, completed_episodes_value = self.evaluate(
        (completed, completed_episodes))
    self.assertEqual(True, completed_value)
    self.assertEqual([0], completed_episodes_value)

    # Invalid episode cannot be completed.
    completed = replay_buffer._maybe_end_episode(1, end_episode=True)
    with tf.control_dependencies([completed]):
      completed_episodes = replay_buffer._completed_episodes()
    completed_value, completed_episodes_value = self.evaluate(
        (completed, completed_episodes))
    self.assertEqual(False, completed_value)
    self.assertEqual([0], completed_episodes_value)

  def testGetBatchEpisodeIdsEndEpisode(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3)
    batch_episode_ids = [replay_buffer.create_episode_ids(num_episodes=3)]
    expected_episode_ids = [[-1, -1, -1]]
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    batch_episode_ids.append(
        replay_buffer._get_batch_episode_ids(batch_episode_ids[-1]))
    with tf.control_dependencies([batch_episode_ids[-1]]):
      completed = [
          replay_buffer._maybe_end_batch_episodes(batch_episode_ids[-1])]
    with tf.control_dependencies(completed):
      completed_episodes = [replay_buffer._completed_episodes()]
    expected_episode_ids.append([0, 1, 2])
    expected_completed = [[False, False, False]]
    expected_completed_episodes = [[]]

    # Mark all episodes as completed.
    with tf.control_dependencies([completed_episodes[-1]]):
      batch_episode_ids.append(
          replay_buffer._get_batch_episode_ids(
              batch_episode_ids[-1], end_episode=True))
    with tf.control_dependencies([batch_episode_ids[-1]]):
      completed_episodes.append(replay_buffer._completed_episodes())
    expected_episode_ids.append([0, 1, 2])
    expected_completed_episodes.append([0, 1, 2])

    # Begin new episodes, it would overwrite the previous ones.
    with tf.control_dependencies([completed_episodes[-1]]):
      batch_episode_ids.append(
          replay_buffer._get_batch_episode_ids(
              batch_episode_ids[-1], begin_episode=True))
    with tf.control_dependencies([batch_episode_ids[-1]]):
      completed_episodes.append(replay_buffer._completed_episodes())
    expected_episode_ids.append([3, 4, 5])
    expected_completed_episodes.append([])

    # Mark one of the new episodes as completed.
    with tf.control_dependencies([completed_episodes[-1]]):
      batch_episode_ids.append(
          replay_buffer._get_batch_episode_ids(
              batch_episode_ids[-1], end_episode=[False, True, False]))
    with tf.control_dependencies([batch_episode_ids[-1]]):
      completed_episodes.append(replay_buffer._completed_episodes())
    expected_episode_ids.append([3, 4, 5])
    expected_completed_episodes.append([4])

    (batch_episode_ids_values,
     completed_episodes_values,
     completed_values) = self.evaluate(
         (batch_episode_ids, completed_episodes, completed))

    self.assertAllEqual(expected_episode_ids, batch_episode_ids_values)
    tf.nest.map_structure(
        self.assertAllEqual,
        [np.array(x) for x in expected_completed_episodes],
        completed_episodes_values)
    self.assertAllEqual(expected_completed, completed_values)

  def testAddSingleSample(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)
    episode_ids = replay_buffer.create_episode_ids(num_episodes=1)

    action = 1 * np.ones(spec[0].shape.as_list(), dtype=np.float32)
    lidar = 2 * np.ones(spec[1][0].shape.as_list(), dtype=np.float32)
    camera = 3 * np.ones(spec[1][1].shape.as_list(), dtype=np.float32)
    values = [action, [lidar, camera]]
    values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1), values)

    episode_len_1_values = tf.nest.map_structure(
        lambda arr: np.expand_dims(arr, 0), values)

    new_episode_ids = replay_buffer.add_batch(values_batched, episode_ids)
    sample, _ = replay_buffer.get_next()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    new_episode_ids_value = self.evaluate(new_episode_ids)
    self.assertEqual(new_episode_ids_value, 0)
    sample_ = self.evaluate(sample)
    self._assertNestedCloseness(self.assertAllClose, episode_len_1_values,
                                sample_)

  def testNumFrames(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=5,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)

    episode_ids_var = common.create_variable(
        'episode_id', initial_value=-1,
        shape=(2,), use_local_variable=True)

    action = 1 * np.ones(spec[0].shape.as_list(), dtype=np.float32)
    lidar = 2 * np.ones(spec[1][0].shape.as_list(), dtype=np.float32)
    camera = 3 * np.ones(spec[1][1].shape.as_list(), dtype=np.float32)
    values = [action, [lidar, camera]]
    values_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 2), values)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    @common.function_in_tf1()
    def add_to_buffer(values_batched):
      new_episode_ids = replay_buffer.add_batch(values_batched, episode_ids_var)
      episode_ids_var.assign(new_episode_ids)
      return new_episode_ids

    for _ in range(4):
      self.evaluate(add_to_buffer(values_batched))

    num_frames = replay_buffer.num_frames()
    num_frames_value = self.evaluate(num_frames)
    self.assertEqual(num_frames_value, 8)

  def testGetNextEmpty(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'),
        [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError, 'EpisodicReplayBuffer is empty. Make '
        'sure to add items before sampling the buffer.'):
      sample, _ = replay_buffer.get_next()
      self.evaluate(sample)

  def testIterateEmpty(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError, 'EpisodicReplayBuffer is empty. Make '
        'sure to add items before sampling the buffer.'):
      sample_two_steps = sample_as_dataset(
          replay_buffer, num_steps=2, batch_size=1)[0]
      self.evaluate(sample_two_steps)

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError, 'EpisodicReplayBuffer is empty. Make '
        'sure to add items before sampling the buffer.'):
      sample_episode = sample_as_dataset(replay_buffer, num_steps=2,
                                         batch_size=1)[0]
      self.evaluate(sample_episode)

    with self.assertRaisesRegexp(
        tf.errors.InvalidArgumentError, 'EpisodicReplayBuffer is empty. Make '
        'sure to add items before sampling the buffer.'):
      sample_episode = sample_as_dataset(
          replay_buffer, num_steps=2, batch_size=1)[0]
      self.evaluate(sample_episode)

  def testAsDatasetBatchSizeFullEpisodeRaisesError(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=10)
    with self.assertRaises(ValueError):
      sample_as_dataset(replay_buffer, num_steps=None, batch_size=10)

  def testAddSteps(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)
    episode_id = replay_buffer.create_episode_ids()

    values = np.ones(spec.shape.as_list())
    values = np.stack([values, 10 * values, 100 * values])
    episode_id = replay_buffer.add_sequence(values, episode_id)
    sample = lambda: sample_as_dataset(replay_buffer, 3, 10)[0]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.evaluate(episode_id)
    sample_ = self.evaluate(sample())
    self._assertContains([values], list(sample_))

  def testAddStepsGetEpisode(self):
    spec = specs.TensorSpec([5], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)

    episode_id = replay_buffer.create_episode_ids()

    values = np.ones(spec.shape.as_list())
    # Stack 3 steps.
    num_steps = 3
    values = np.stack([values * step for step in range(num_steps)])
    episode_id = replay_buffer.add_sequence(values, episode_id)
    episode = replay_buffer._get_episode(episode_id)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertAllEqual(values, self.evaluate(episode))

  def testAddStepsGetEpisodes(self):
    spec = specs.TensorSpec([5], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)
    episode_id_0 = replay_buffer.create_episode_ids()
    episode_id_1 = replay_buffer.create_episode_ids()

    values = np.ones(spec.shape.as_list())
    # Add episode_0 with 3 steps
    values_0 = np.stack((values, 10 * values, 100 * values))
    episode_id_0 = replay_buffer.add_sequence(values_0, episode_id_0)
    episode_0 = replay_buffer._get_episode(episode_id_0)
    # Add episode_1 with 2 steps
    values_1 = np.stack((2 * values, 20 * values))
    episode_id_1 = replay_buffer.add_sequence(values_1, episode_id_1)
    episode_1 = replay_buffer._get_episode(episode_id_1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    episode_0_value = self.evaluate(episode_0)
    episode_1_value = self.evaluate(episode_1)

    self.assertEqual(1, self.evaluate(replay_buffer._get_last_episode_id()))
    self.assertAllEqual(values_0, episode_0_value)
    self.assertAllEqual(values_1, episode_1_value)

  def testAddStepsUnknownBatchDims(self):
    if tf.executing_eagerly():
      self.skipTest('b/123770194')

    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)
    episode_ids = replay_buffer.create_episode_ids()

    values = np.ones(spec.shape.as_list())
    values = np.stack([values, 10 * values, 100 * values])
    batch = tf.compat.v1.placeholder(shape=[None, 3], dtype=tf.int32)
    sample, _ = sample_as_dataset(replay_buffer, num_steps=3, batch_size=10)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      sess.run(replay_buffer.add_sequence(batch, episode_ids),
               {batch: values})
      sample_ = sess.run(sample)
      self._assertContains([values], list(sample_))

  def testMultipleAddBatch(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)

    values = np.ones(spec.shape.as_list(), dtype=np.int32)
    values = np.stack([values, 10 * values, 100 * values])
    episode_ids = replay_buffer.create_episode_ids(num_episodes=3)
    # In this case all episodes are valid, so it will add all the values.
    # Add 1 to ep0, 10 to ep1, 100 to ep2
    new_episode_ids = replay_buffer.add_batch(values, episode_ids)
    # In this case the second episode will be invalid because we're starting
    # two new episodes with a capacity of 3, so ep1 with location 1 is
    # now going to be replaced by ep4 (in location 1).  As a result we won't
    # add its values.
    #
    # Add 1 to ep3 (was ep0), ., 100 to ep4 (was ep1).
    replay_buffer._begin_episode_fn = lambda _: [True, False, True]
    new_episode_ids_2 = replay_buffer.add_batch(values, new_episode_ids)
    # In this case all episodes are valid, so it will add all the values.
    # Add 1 to ep3, 10 to ep5 (was ep2), 100 to ep4.
    replay_buffer._begin_episode_fn = lambda _: [False, True, False]
    new_episode_ids_3 = replay_buffer.add_batch(values, new_episode_ids_2)

    # End result: ep3 with [1, 1], ep4 with [100, 100], ep5 with [10].
    items = [replay_buffer._get_episode(i) for i in [3, 4, 5]]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertAllEqual(
        ([0, 1, 2],
         [3, 1, 4],
         [3, 5, 4]),
        self.evaluate((new_episode_ids,
                       new_episode_ids_2,
                       new_episode_ids_3)))
    self.assertEqual(5, self.evaluate(replay_buffer._get_last_episode_id()))
    items_ = self.evaluate(items)
    self.assertAllEqual(items_[0], [values[0], values[0]])
    self.assertAllEqual(items_[1], [values[2], values[2]])
    self.assertAllEqual(items_[2], [values[1]])

  def testAddBatchGetEpisodes(self):
    num_episodes = 3
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=num_episodes, begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)
    episode_ids = replay_buffer.create_episode_ids(num_episodes)

    values = np.ones(spec.shape.as_list(), dtype=np.int32)
    values = np.stack([values, 10 * values, 100 * values])
    episode_ids = replay_buffer.add_batch(values, episode_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    episode_ids = self.evaluate(episode_ids)
    for episode_id in range(num_episodes):
      episode = replay_buffer._get_episode(episode_ids[episode_id])
      self.assertAllEqual([values[episode_id]], self.evaluate(episode))

  def testAddBatchUnknownBatchDims(self):
    if tf.executing_eagerly():
      self.skipTest('b/123770194')

    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3, begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)
    episode_ids = replay_buffer.create_episode_ids(3)

    values = np.ones(spec.shape.as_list(), dtype=np.int32)
    values = np.stack([values, 10 * values, 100 * values])
    batch = tf.compat.v1.placeholder(shape=[None, 3], dtype=tf.int32)
    items = replay_buffer.gather_all()
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(tf.compat.v1.local_variables_initializer())
      sess.run(
          replay_buffer.add_batch(batch, episode_ids), {batch: values})
      self.assertEqual(2, sess.run(replay_buffer._get_last_episode_id()))
      self.assertAllEqual(values, sess.run(items)[0])
      # Make sure it's safe to run the tf.data pipeline for gather_all twice!
      self.assertAllEqual(values, sess.run(items)[0])

  def testGatherAllEmpty(self):
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(spec)

    items = replay_buffer.gather_all()
    expected = []

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    items_ = self.evaluate(items)
    self.assertAllClose(expected, items_)

  def testParallelAdds(self):
    spec = specs.TensorSpec([], tf.int32, 'action')
    num_adds = 10
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=num_adds, begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)
    expected_items = range(num_adds)
    items_batched = tf.nest.map_structure(lambda t: tf.stack([t] * 1),
                                          expected_items)
    add_ops = []
    for item in items_batched:
      episode_ids = replay_buffer.create_episode_ids(1)
      add_ops.append(replay_buffer.add_batch(item, episode_ids))
    items = replay_buffer.gather_all()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.evaluate(add_ops)
    self.assertEqual(self.evaluate(replay_buffer._get_last_episode_id()), 9)
    items_ = self.evaluate(items)[0]
    self.assertSameElements(expected_items, items_)

  def testExtractNoClear(self):
    num_episodes = 5
    episode_length = 3
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=num_episodes - 1,
        begin_episode_fn=lambda _: True, end_episode_fn=lambda _: False)
    episode_id = replay_buffer.create_episode_ids()
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    for i in range(num_episodes):
      episode_id = replay_buffer.add_sequence(
          i * tf.ones([episode_length], dtype=tf.int32),
          episode_id)
    self.evaluate(episode_id)
    episodes = replay_buffer.extract([1, 0], clear_data=False)
    [
        episodes_length_, episodes_completed_, extracted_first_,
        extracted_second_
    ] = self.evaluate([
        episodes.length,
        episodes.completed,
        list_ops.tensor_list_stack(episodes.tensor_lists[0], spec.dtype),
        list_ops.tensor_list_stack(episodes.tensor_lists[1], spec.dtype),
    ])
    self.assertAllEqual(episodes_length_, [episode_length, episode_length])
    self.assertAllEqual(episodes_completed_, [False, False])
    self.assertAllClose(extracted_first_, [1] * episode_length)
    self.assertAllClose(extracted_second_, [4] * episode_length)

    # The location associated with episode ID 0 is not cleared.
    self.assertAllClose(
        self.evaluate(replay_buffer._get_episode(num_episodes - 1)),
        [num_episodes - 1] * episode_length)
    # The episode ID 1 (extracted) was not cleared.
    self.assertAllClose(
        self.evaluate(replay_buffer._get_episode(1)), [1.0] * episode_length)

  def testExtractAndClear(self):
    num_episodes = 5
    episode_length = 3
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=num_episodes - 1,
        begin_episode_fn=lambda _: True, end_episode_fn=lambda _: False)
    episode_id = replay_buffer.create_episode_ids()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    for i in range(num_episodes):
      episode_id = replay_buffer.add_sequence(
          i * tf.ones([episode_length], dtype=tf.int32),
          episode_id)
    self.evaluate(episode_id)  # Run the insertions.

    episodes = replay_buffer.extract([1, 0], clear_data=True)
    [
        episodes_length_, episodes_completed_, extracted_first_,
        extracted_second_
    ] = self.evaluate([
        episodes.length,
        episodes.completed,
        list_ops.tensor_list_stack(episodes.tensor_lists[0], spec.dtype),
        list_ops.tensor_list_stack(episodes.tensor_lists[1], spec.dtype),
    ])
    self.assertAllEqual(episodes_length_, [episode_length, episode_length])
    self.assertAllEqual(episodes_completed_, [False, False])
    self.assertAllClose(extracted_first_, [1] * episode_length)
    self.assertAllClose(extracted_second_, [4] * episode_length)

    # The location associated with episode ID 0 (extracted) was cleared.
    self.assertAllEqual(
        self.evaluate(
            tf.size(input=replay_buffer._get_episode(num_episodes - 1))), 0)
    # The episode ID 1 (extracted) was cleared.
    self.assertEqual(
        self.evaluate(tf.size(input=replay_buffer._get_episode(1))), 0)

  @parameterized.parameters([dict(stateless=False), dict(stateless=True)])
  def testExtend(self, stateless):
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3, begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)

    if stateless:
      extend_ids = replay_buffer.create_episode_ids(3)
    else:
      stateful_replay_buffer = (
          episodic_replay_buffer.StatefulEpisodicReplayBuffer(
              replay_buffer, num_episodes=3))
      extend_ids = stateful_replay_buffer.episode_ids

    episodes1 = episodic_replay_buffer.Episodes(
        length=tf.constant([2, 1, 3], dtype=tf.int64),
        completed=tf.constant([0, 0, 1], dtype=tf.uint8),
        tensor_lists=tf.stack([
            list_ops.tensor_list_from_tensor(
                tf.constant([100, 200], dtype=spec.dtype),
                element_shape=spec.shape),
            list_ops.tensor_list_from_tensor(
                tf.constant([999], dtype=spec.dtype), element_shape=spec.shape),
            list_ops.tensor_list_from_tensor(
                tf.constant([1, 2, 3], dtype=spec.dtype),
                element_shape=spec.shape),
        ]))

    episodes2 = episodic_replay_buffer.Episodes(
        length=tf.constant([1, 3], dtype=tf.int64),
        completed=tf.constant([0, 1], dtype=tf.uint8),
        tensor_lists=tf.stack([
            list_ops.tensor_list_from_tensor(
                tf.constant([888], dtype=spec.dtype), element_shape=spec.shape),
            list_ops.tensor_list_from_tensor(
                tf.constant([4, 5, 6], dtype=spec.dtype),
                element_shape=spec.shape),
        ]))

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    if stateless:
      extended_ids_1 = replay_buffer.extend_episodes(
          extend_ids, [0, 1, 2], episodes1)
      extend_ids = extended_ids_1
      replay_buffer._begin_episode_fn = lambda _: [False, True]
      extended_ids_2 = replay_buffer.extend_episodes(
          extend_ids, [1, 2], episodes2)
    else:
      replay_buffer._begin_episode_fn = lambda _: False
      extended_ids_1 = stateful_replay_buffer.extend_episodes(
          [0, 1, 2], episodes1)
      replay_buffer._begin_episode_fn = lambda _: [False, True]
      extended_ids_2 = stateful_replay_buffer.extend_episodes(
          [1, 2], episodes2)

    if stateless:
      extended_ids_1_value, extended_ids_2_value = (
          self.evaluate((extended_ids_1, extended_ids_2)))
    else:
      extended_ids_1_value = self.evaluate(extended_ids_1)
      extended_ids_2_value = self.evaluate(extended_ids_2)

    self.assertAllEqual(extended_ids_1_value, [0, 1, 2])
    self.assertAllEqual(extended_ids_2_value, [0, 1, 3])

    if not stateless:
      self.assertAllEqual(self.evaluate(extend_ids), [0, 1, 3])

    self.assertAllEqual(self.evaluate(
        replay_buffer._get_episode(1)), [999, 888])
    self.assertAllEqual(self.evaluate(replay_buffer._get_episode(2)), [1, 2, 3])
    self.assertAllEqual(self.evaluate(replay_buffer._get_episode(3)), [4, 5, 6])

  def testClearAll(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    values = tf.expand_dims(np.ones(spec.shape.as_list(), dtype=np.int32), 0)

    empty_values = np.empty((0, 3), dtype=np.int32)
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)

    episode_id_1 = replay_buffer.create_episode_ids(1)
    episode_id_2 = replay_buffer.create_episode_ids(1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    self.assertAllEqual(self.evaluate(replay_buffer.gather_all()), empty_values)
    self.assertEqual(self.evaluate(replay_buffer._last_episode), -1)
    self.assertAllEqual(self.evaluate(replay_buffer._episode_lengths), [0, 0])

    self.evaluate(replay_buffer.add_batch(values, episode_id_1))
    self.assertAllEqual(self.evaluate(replay_buffer.gather_all())[0], values)
    self.assertEqual(self.evaluate(replay_buffer._last_episode), 0)
    self.assertAllEqual(self.evaluate(replay_buffer._episode_lengths), [1, 0])

    self.evaluate(replay_buffer.clear())
    self.assertAllEqual(self.evaluate(replay_buffer.gather_all()), empty_values)
    self.assertEqual(self.evaluate(replay_buffer._last_episode), 0)
    self.assertAllEqual(self.evaluate(replay_buffer._episode_lengths), [0, 0])

    self.evaluate(replay_buffer.add_batch(values, episode_id_2))
    self.assertAllEqual(self.evaluate(replay_buffer.gather_all())[0], values)
    self.evaluate(replay_buffer._clear(clear_all_variables=True))
    self.assertAllEqual(self.evaluate(replay_buffer.gather_all()), empty_values)
    self.assertEqual(self.evaluate(replay_buffer._last_episode), -1)
    self.assertAllEqual(self.evaluate(replay_buffer._episode_lengths), [0, 0])

  def _create_rb_and_add_3N_episodes(
      self, drop_remainder=False, window_shift=None, repeat=1):
    spec = {'a': specs.TensorSpec([], tf.int32, 'spec')}
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec,
        capacity=10,
        dataset_drop_remainder=drop_remainder,
        dataset_window_shift=window_shift,
        begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    episode_id = replay_buffer.create_episode_ids()

    values = np.ones(spec['a'].shape.as_list())
    values = np.stack([values, 10 * values, 100 * values])

    original_episode_id = self.evaluate(episode_id)

    for _ in range(repeat):
      # Add an episode with frames [1, 10, 100]
      replay_buffer._begin_episode_fn = lambda _: True
      episode_id = replay_buffer.add_sequence({'a': values}, episode_id)
      # Add an episode with frames [2, 20, 200, 3, 30, 300]
      episode_id = replay_buffer.add_sequence({'a': 2 * values}, episode_id)
      replay_buffer._begin_episode_fn = lambda _: False
      episode_id = replay_buffer.add_sequence({'a': 3 * values}, episode_id)
      # Add an episode with frames [-1, -10, -100]
      replay_buffer._begin_episode_fn = lambda _: True
      episode_id = replay_buffer.add_sequence({'a': -values}, episode_id)
    new_episode_id = self.evaluate(episode_id)

    assert new_episode_id >= original_episode_id + repeat * 3, (
        '{} vs. {}'.format(new_episode_id, original_episode_id + repeat * 3))

    return replay_buffer

  def testSingleDeterministicPassAsDataset(self):
    replay_buffer = self._create_rb_and_add_3N_episodes()
    with self.cached_session() as session:
      pass
    itr = iterator_from_dataset(
        replay_buffer,
        single_deterministic_pass=True,
        session=session)

    tf.nest.map_structure(
        self.assertAllEqual, {'a': _a([1, 10, 100])}, next(itr))
    tf.nest.map_structure(
        self.assertAllEqual, {'a': _a([2, 20, 200, 3, 30, 300])}, next(itr))
    tf.nest.map_structure(
        self.assertAllEqual, {'a': _a([-1, -10, -100])}, next(itr))
    with self.assertRaises((tf.errors.OutOfRangeError, StopIteration)):
      next(itr)

  @parameterized.parameters(
      [dict(drop_remainder=False), dict(drop_remainder=True)])
  def testSingleDeterministicPassAsDatasetWithNumSteps(
      self, drop_remainder):
    replay_buffer = self._create_rb_and_add_3N_episodes(
        drop_remainder=drop_remainder)
    with self.cached_session() as session:
      pass
    itr = iterator_from_dataset(
        replay_buffer,
        num_steps=5,
        single_deterministic_pass=True,
        session=session)

    tf.nest.map_structure(
        self.assertAllEqual, {'a': _a([1, 10, 100, 2, 20])}, next(itr))
    tf.nest.map_structure(
        self.assertAllEqual, {'a': _a([200, 3, 30, 300, -1])}, next(itr))
    if not drop_remainder:
      tf.nest.map_structure(
          self.assertAllEqual, {'a': _a([-10, -100])}, next(itr))
    with self.assertRaises((tf.errors.OutOfRangeError, StopIteration)):
      next(itr)

  def testSingleDeterministicPassAsDatasetWithNumStepsBatchSize(self):
    # Add 6 episodes, repeating 3 episodes twice.
    replay_buffer = self._create_rb_and_add_3N_episodes(
        window_shift=None, drop_remainder=False, repeat=2)
    with self.cached_session() as session:
      pass
    itr = iterator_from_dataset(
        replay_buffer,
        batch_size=3,
        num_steps=5,
        single_deterministic_pass=True,
        session=session)

    # NOTE(ebrevdo): Here, the final steps of the final episodes get cut off.
    tf.nest.map_structure(self.assertAllEqual,
                          {'a': _a([[1, 10, 100, 1, 10],
                                    [2, 20, 200, 3, 30],
                                    [-1, -10, -100, -1, -10]])},
                          next(itr))
    # Note here we are missing the final [100] from episode 4, the final [30,
    # 300] from episode 5, and the final [-100] from episode 6.
    tf.nest.map_structure(
        self.assertAllEqual, {'a': _a([[300, 2, 20, 200, 3]])}, next(itr))
    with self.assertRaises((tf.errors.OutOfRangeError, StopIteration)):
      next(itr)

  def testSingleDeterministicPassAsDatasetWithNumStepsBatchSizeAndShift(self):
    # Add 6 episodes, repeating 3 episodes twice.
    replay_buffer = self._create_rb_and_add_3N_episodes(
        window_shift=1, drop_remainder=False, repeat=2)
    with self.cached_session() as session:
      pass
    itr = iterator_from_dataset(
        replay_buffer,
        batch_size=3,
        num_steps=5,
        single_deterministic_pass=True,
        session=session)

    # Due to window_shift == 1, we see two instances of each block: the
    # original, and one shifted left by 1.  For example, if we have two episodes
    # in a batch entry, [1, 10, 100] and [1, 10, 100] again, then with num_steps
    # == 5 we'll see one block that's [1, 10, 100, 1, 10] and a second one
    # that's shifted: [10, 100, 1, 10, 100].
    tf.nest.map_structure(self.assertAllEqual,
                          {'a': _a([[1, 10, 100, 1, 10],
                                    [2, 20, 200, 3, 30],
                                    [-1, -10, -100, -1, -10]])},
                          next(itr))
    tf.nest.map_structure(self.assertAllEqual,
                          {'a': _a([[10, 100, 1, 10, 100],
                                    [20, 200, 3, 30, 300],
                                    [-10, -100, -1, -10, -100]])},
                          next(itr))
    tf.nest.map_structure(self.assertAllEqual,
                          {'a': _a([[200, 3, 30, 300, 2],
                                    [3, 30, 300, 2, 20],
                                    [30, 300, 2, 20, 200]])},
                          next(itr))
    tf.nest.map_structure(self.assertAllEqual,
                          {'a': _a([[300, 2, 20, 200, 3],
                                    [2, 20, 200, 3, 30],
                                    [20, 200, 3, 30, 300]])},
                          next(itr))
    with self.assertRaises((tf.errors.OutOfRangeError, StopIteration)):
      next(itr)


class StatefulEpisodicReplayBufferTest(test_utils.TestCase):

  def _assertContains(self, list1, list2):
    self.assertTrue(test_utils.contains(list1, list2))

  def _assertCircularOrdering(self, expected_order, given_order):
    for i in range(len(given_order)):
      self.assertIn(given_order[i], expected_order)
      if i > 0:
        prev_idx = expected_order.index(given_order[i - 1])
        cur_idx = expected_order.index(given_order[i])
        self.assertEqual(cur_idx, (prev_idx + 1) % len(expected_order))

  def testCreateEpisodeId(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)
    replay_buffer_stateful_0 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(replay_buffer))
    replay_buffer_stateful_1 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(replay_buffer))

    episode_0 = replay_buffer_stateful_0.episode_ids
    episode_1 = replay_buffer_stateful_1.episode_ids

    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertIsNot(episode_0, episode_1)
    self.assertEqual(self.evaluate(episode_0), -1)
    self.assertEqual(self.evaluate(episode_1), -1)

  def testCreateBatchEpisodeIds(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=5)

    replay_buffer_stateful_0 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=2))
    replay_buffer_stateful_1 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=3))
    episodes_0 = replay_buffer_stateful_0.episode_ids
    episodes_1 = replay_buffer_stateful_1.episode_ids

    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertIsNot(episodes_0, episodes_1)
    self.assertAllEqual([-1] * 2, self.evaluate(episodes_0))
    self.assertAllEqual([-1] * 3, self.evaluate(episodes_1))

  def testCreateTooManyBatchEpisodeIdsRaisesError(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2)

    with self.assertRaisesRegexp(
        ValueError, 'Buffer cannot create episode_ids when '
        'num_episodes 3 > capacity 2.'):
      episodic_replay_buffer.StatefulEpisodicReplayBuffer(
          replay_buffer, num_episodes=3)

  def testAddSingleMultipleTimesSampleAsDatasetBatched(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=10, begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)
    replay_buffer_stateful = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    values = np.expand_dims(np.ones(spec.shape.as_list(), dtype=np.int32), 0)
    values = [values, 10 * values, 100 * values]
    simple_values = [1, 10, 100]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.evaluate(replay_buffer_stateful.add_batch(values[0]))
    self.evaluate(replay_buffer_stateful.add_batch(values[1]))
    self.evaluate(replay_buffer_stateful.add_batch(values[2]))
    sample_next = (sample_as_dataset(replay_buffer, num_steps=3,
                                     batch_size=100)[0])

    self.assertEqual(tf.compat.dimension_value(sample_next.shape[2]), 3)
    sample_ = self.evaluate(sample_next)

    # Shape should be batch_size x episode_length x tensor spec.
    # In this case, all episodes have length 3, which is why batching works.
    self.assertEqual(sample_.shape, (100, 3, 3))

    for multi_item in sample_:
      self._assertCircularOrdering(simple_values,
                                   [item[0] for item in multi_item])

    self._assertContains(
        list(values), [multi_item[0] for multi_item in sample_])

  def testAddSingleMultipleTimesSampleAsDatasetBatchedMultiStep(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=2, begin_episode_fn=lambda _: False,
        end_episode_fn=lambda _: False)
    replay_buffer_stateful = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    values = np.expand_dims(np.ones(spec.shape.as_list(), dtype=np.int32), 0)
    values = [values, 10 * values, 100 * values]
    simple_values = [1, 10, 100]
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.evaluate(replay_buffer_stateful.add_batch(values[0]))
    self.evaluate(replay_buffer_stateful.add_batch(values[1]))
    self.evaluate(replay_buffer_stateful.add_batch(values[2]))
    sample_next = (
        sample_as_dataset(replay_buffer, num_steps=2, batch_size=100)[0])
    self.assertEqual(sample_next.shape[1:].as_list(), [2, 3])
    sample_ = self.evaluate(sample_next)

    # Shape should be batch_size x num_steps x tensor spec.
    self.assertEqual(sample_.shape, (100, 2, 3))

    for multi_item in sample_:
      self._assertCircularOrdering(simple_values,
                                   [item[0] for item in multi_item])

    self._assertContains(
        list(values), [multi_item[0] for multi_item in sample_])

  def testAddBatch(self):
    spec = specs.TensorSpec([3], tf.int32, 'lidar')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)

    stateful_replay_buffer = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=3))
    values = np.ones(spec.shape.as_list(), dtype=np.int32)
    values = np.stack([values, 10 * values, 100 * values])
    new_episode_ids = stateful_replay_buffer.add_batch(values)
    item_0 = replay_buffer._get_episode(0)
    item_1 = replay_buffer._get_episode(1)
    item_2 = replay_buffer._get_episode(2)
    # We appended one time step to each of 3 episodes.  If we tf.stack
    # the line below, we would end up with shape
    #  (batch_size, time_steps, depth) == (3, 1, 3).
    #
    # Instead we concat here to make testing easier below.
    items = tf.nest.map_structure(lambda *x: tf.concat(x, 0), item_0, item_1,
                                  item_2)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    self.assertAllEqual([0, 1, 2], self.evaluate(new_episode_ids))
    self.assertEqual(2, self.evaluate(replay_buffer._get_last_episode_id()))
    self.assertAllEqual(values, self.evaluate(items))

  def testMultipleAddBatch(self):
    spec = (specs.TensorSpec([3], tf.int32, 'lidar'),
            specs.TensorSpec([], tf.bool, 'begin_episode'))
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3,
        begin_episode_fn=lambda value_and_begin: value_and_begin[1],
        end_episode_fn=lambda _: False)
    stateful_replay_buffer = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=3))
    episode_ids_var = stateful_replay_buffer.episode_ids

    values = np.ones(spec[0].shape.as_list(), dtype=np.int32)
    values = np.stack([values, 10 * values, 100 * values])
    # In this case all episodes are valid, so it will add all the values.
    # Add 1 to ep0, 10 to ep1, 100 to ep2
    new_episode_ids = stateful_replay_buffer.add_batch(
        (values, tf.constant([False, False, False])))
    with tf.control_dependencies([new_episode_ids]):
      items = [[replay_buffer._get_episode(i) for i in [0, 1, 2]]]
    # In this case the second episode is invalid, so it won't add its values.
    # Add 1 to ep3 (was ep0), ., 100 to ep4 (was ep1).
    with tf.control_dependencies(items[-1][0]):
      new_episode_ids_2 = stateful_replay_buffer.add_batch(
          (values, tf.constant([True, False, True])))
    with tf.control_dependencies([new_episode_ids_2]):
      items.append([replay_buffer._get_episode(i) for i in [2, 3, 4]])
    # In this case all episodes are valid, so it will add all the values.
    # Add 1 to ep3, 10 to ep5 (was ep2), 100 to ep4.
    with tf.control_dependencies(items[-1][0]):
      new_episode_ids_3 = stateful_replay_buffer.add_batch(
          (values, tf.constant([False, True, False])))
    with tf.control_dependencies([new_episode_ids_3]):
      # End result: ep3 with [1, 1], ep4 with [100, 100], ep5 with [10].
      items.append([replay_buffer._get_episode(i) for i in [3, 4, 5]])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    (new_episode_ids,
     new_episode_ids_2,
     new_episode_ids_3,
     items) = self.evaluate(
         (new_episode_ids,
          new_episode_ids_2,
          new_episode_ids_3,
          items))
    self.assertAllEqual([0, 1, 2], new_episode_ids)
    self.assertAllEqual([3, 1, 4], new_episode_ids_2)
    self.assertAllEqual([3, 5, 4], new_episode_ids_3)
    episode_ids_value = self.evaluate(episode_ids_var)
    self.assertAllEqual([3, 5, 4], episode_ids_value)
    self.assertEqual(5, self.evaluate(replay_buffer._get_last_episode_id()))

    # First add_batch: ep0, ep1, ep2.
    self.assertAllEqual(items[0][0][0], [values[0]])
    self.assertAllEqual(items[0][1][0], [values[1]])
    self.assertAllEqual(items[0][2][0], [values[2]])
    # Second add_batch: ep2, ep3, ep4.
    self.assertAllEqual(items[1][0][0], [values[2]])
    self.assertAllEqual(items[1][1][0], [values[0]])
    self.assertAllEqual(items[1][2][0], [values[2]])
    # Third add_batch: ep3, ep4, ep5.
    self.assertAllEqual(items[2][0][0], [values[0], values[0]])
    self.assertAllEqual(items[2][1][0], [values[2], values[2]])
    self.assertAllEqual(items[2][2][0], [values[1]])

  def testGatherAll(self):
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)
    replay_buffer_stateful = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))

    action_state = common.create_variable('counter', -1, dtype=spec.dtype)
    action = lambda: tf.expand_dims(action_state.assign_add(1), 0)

    # pylint: disable=unnecessary-lambda
    items = lambda: replay_buffer.gather_all()
    expected = [list(range(10))]
    if tf.executing_eagerly():
      add_op = lambda: replay_buffer_stateful.add_batch(action())
    else:
      add_op = replay_buffer_stateful.add_batch(action())

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    for _ in range(10):
      self.evaluate(add_op)

    items_ = self.evaluate(items())
    self.assertAllClose(expected, items_)

  def testAddOverCapacity(self):
    num_adds = 5
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3,
        begin_episode_fn=lambda _: True, end_episode_fn=lambda _: False)

    a_state = common.create_variable('counter', -1, dtype=spec.dtype)
    a = lambda: tf.expand_dims(a_state.assign_add(1), 0)
    replay_buffer_stateful = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))

    if tf.executing_eagerly():
      # pylint: disable=g-long-lambda
      # pylint: disable=unnecessary-lambda
      add = lambda: replay_buffer_stateful.add_batch(a())
      items = replay_buffer.gather_all
    else:
      add = replay_buffer_stateful.add_batch(a())
      items = replay_buffer.gather_all()

    expected = [2, 3, 4]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    for _ in range(num_adds):
      self.evaluate(add)
    items_ = self.evaluate(items)[0]
    self.assertAllClose(expected, items_)

  def testOverwriteOldEpisodes(self):
    num_adds = 5
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=3,
        begin_episode_fn=lambda _: True, end_episode_fn=lambda _: False)

    a_state = common.create_variable('counter', -1, dtype=spec.dtype)
    a = lambda: tf.expand_dims(a_state.assign_add(1), 0)

    replay_buffer_stateful = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    episode_id_var = replay_buffer_stateful.episode_ids
    if tf.executing_eagerly():
      def add():
        return replay_buffer_stateful.add_batch(a())
      items = replay_buffer.gather_all
    else:
      add = replay_buffer_stateful.add_batch(a())
      items = replay_buffer.gather_all()
    expected = [2, 3, 4]

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    for i in range(num_adds):
      self.evaluate(add)
      self.assertEqual(self.evaluate(episode_id_var), i)

    items_ = self.evaluate(items)[0]
    self.assertAllClose(items_, expected)

  def testAddToStaleEpisodeIDIsAvoided(self):
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=1,
        begin_episode_fn=lambda _: False, end_episode_fn=lambda _: False)

    a_state = common.create_variable('counter', -1, dtype=spec.dtype)
    a = lambda: tf.expand_dims(a_state.assign_add(1), 0)

    replay_buffer_stateful_0 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    replay_buffer_stateful_1 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    episode_id_var_0 = replay_buffer_stateful_0.episode_ids
    episode_id_var_1 = replay_buffer_stateful_1.episode_ids

    items_fn = replay_buffer.gather_all

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())

    self.evaluate(replay_buffer_stateful_0.add_batch(a()))
    self.assertEqual(self.evaluate(episode_id_var_0), 0)
    self.assertAllClose(self.evaluate(items_fn())[0], [0])

    self.evaluate(replay_buffer_stateful_1.add_batch(a()))
    self.assertEqual(self.evaluate(episode_id_var_1), 1)
    self.assertAllClose(self.evaluate(items_fn())[0], [1])

    # Adding to episode 0 will be avoided as it is no longer there.
    self.evaluate(replay_buffer_stateful_0.add_batch(a()))
    self.assertEqual(self.evaluate(episode_id_var_0), 0)
    self.assertAllClose(self.evaluate(items_fn())[0], [1])

  def testParallelAddOverCapacity(self):
    num_adds = 3
    spec = specs.TensorSpec([], tf.int32, 'action')
    replay_buffer = episodic_replay_buffer.EpisodicReplayBuffer(
        spec, capacity=4,
        begin_episode_fn=lambda _: True, end_episode_fn=lambda _: False)

    a = common.create_variable('a', 0, dtype=spec.dtype)
    b = common.create_variable('b', 10, dtype=spec.dtype)

    replay_buffer_stateful_0 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    replay_buffer_stateful_1 = (
        episodic_replay_buffer.StatefulEpisodicReplayBuffer(
            replay_buffer, num_episodes=1))
    episode_id_var_0 = replay_buffer_stateful_0.episode_ids
    episode_id_var_1 = replay_buffer_stateful_1.episode_ids

    expected = [2, 12, 1, 11]

    @common.function
    def add_elem_0(variable):
      elem = tf.expand_dims(variable, 0)
      replay_buffer_stateful_0.add_batch(elem)

    @common.function
    def add_elem_1(variable):
      elem = tf.expand_dims(variable, 0)
      replay_buffer_stateful_1.add_batch(elem)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.local_variables_initializer())
    for _ in range(num_adds):
      self.evaluate([add_elem_0(a),
                     add_elem_1(b)])
      self.evaluate([a.assign_add(1), b.assign_add(1)])
    items = replay_buffer.gather_all()
    items_ = self.evaluate(items)[0]
    episode_ids = self.evaluate([episode_id_var_0, episode_id_var_1])
    self.assertSameElements(episode_ids, [4, 5])
    self.assertSameElements(items_, expected)


if __name__ == '__main__':
  tf.test.main()
