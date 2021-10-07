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

# Lint as: python3
"""Tests for tf_agents.replay_buffers.reverb_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools

import time
from absl.testing import parameterized
import mock
import reverb
from six.moves import range
import tensorflow as tf

from tf_agents.drivers import py_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import test_envs
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class ReverbTableTests(test_utils.TestCase):

  def test_queue_table(self):
    table_name = 'test_queue_table'
    queue_table = reverb.Table.queue(table_name, 3)
    reverb_server = reverb.Server([queue_table])
    data_spec = tensor_spec.TensorSpec((), dtype=tf.int64)
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        table_name,
        local_server=reverb_server,
        sequence_length=1,
        dataset_buffer_size=1)

    with replay.py_client.trajectory_writer(num_keep_alive_refs=1) as writer:
      for i in range(3):
        writer.append(i)
        trajectory = writer.history[-1:]
        writer.create_item(table_name, trajectory=trajectory, priority=1)

    dataset = replay.as_dataset(
        sample_batch_size=1, num_steps=None, num_parallel_calls=1)

    iterator = iter(dataset)
    for i in range(3):
      sample = next(iterator)[0]
      self.assertEqual(sample, i)

  def test_uniform_table(self):
    table_name = 'test_uniform_table'
    queue_table = reverb.Table(
        table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1000,
        rate_limiter=reverb.rate_limiters.MinSize(3))
    reverb_server = reverb.Server([queue_table])
    data_spec = tensor_spec.TensorSpec((), dtype=tf.int64)
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        table_name,
        local_server=reverb_server,
        sequence_length=1,
        dataset_buffer_size=1)

    with replay.py_client.trajectory_writer(num_keep_alive_refs=1) as writer:
      for i in range(3):
        writer.append(i)
        trajectory = writer.history[-1:]
        writer.create_item(table_name, trajectory=trajectory, priority=1)

    dataset = replay.as_dataset(
        sample_batch_size=1, num_steps=None, num_parallel_calls=1)

    iterator = iter(dataset)
    counts = [0] * 3
    for i in range(1000):
      item_0 = next(iterator)[0].numpy()  # This is a matrix shaped 1x1.
      counts[int(item_0)] += 1

    # Comparing against 200 to avoid flakyness
    self.assertGreater(counts[0], 200)
    self.assertGreater(counts[1], 200)
    self.assertGreater(counts[2], 200)

  def test_uniform_table_max_sample(self):
    table_name = 'test_uniform_table'
    table = reverb.Table(
        table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=3,
        max_times_sampled=10,
        rate_limiter=reverb.rate_limiters.MinSize(1))
    reverb_server = reverb.Server([table])
    data_spec = tensor_spec.TensorSpec((), dtype=tf.int64)
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        table_name,
        local_server=reverb_server,
        sequence_length=1,
        dataset_buffer_size=1)

    with replay.py_client.trajectory_writer(1) as writer:
      for i in range(3):
        writer.append(i)
        writer.create_item(table_name, trajectory=writer.history[-1:],
                           priority=1)

    dataset = replay.as_dataset(sample_batch_size=3, num_parallel_calls=3)

    self.assertTrue(table.can_sample(3))
    iterator = iter(dataset)
    counts = [0] * 3
    for i in range(10):
      item_0 = next(iterator)[0].numpy()  # This is a matrix shaped 1x3.
      for item in item_0:
        counts[int(item)] += 1
    self.assertFalse(table.can_sample(3))

    # Same number of counts due to limit on max_times_sampled
    self.assertEqual(counts[0], 10)
    self.assertEqual(counts[1], 10)
    self.assertEqual(counts[2], 10)

  def test_prioritized_table(self):
    table_name = 'test_prioritized_table'
    queue_table = reverb.Table(
        table_name,
        sampler=reverb.selectors.Prioritized(1.0),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        max_size=3)
    reverb_server = reverb.Server([queue_table])
    data_spec = tensor_spec.TensorSpec((), dtype=tf.int64)
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        table_name,
        sequence_length=1,
        local_server=reverb_server,
        dataset_buffer_size=1)

    with replay.py_client.trajectory_writer(1) as writer:
      for i in range(3):
        writer.append(i)
        writer.create_item(table_name, trajectory=writer.history[-1:],
                           priority=i)

    dataset = replay.as_dataset(
        sample_batch_size=1, num_steps=None, num_parallel_calls=1)

    iterator = iter(dataset)
    counts = [0] * 3
    for i in range(1000):
      item_0 = next(iterator)[0].numpy()  # This is a matrix shaped 1x1.
      counts[int(item_0)] += 1

    self.assertEqual(counts[0], 0)  # priority 0
    self.assertGreater(counts[1], 250)  # priority 1
    self.assertGreater(counts[2], 600)  # priority 2

  def test_prioritized_table_max_sample(self):
    table_name = 'test_prioritized_table'
    table = reverb.Table(
        table_name,
        sampler=reverb.selectors.Prioritized(1.0),
        remover=reverb.selectors.Fifo(),
        max_times_sampled=10,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        max_size=3)
    reverb_server = reverb.Server([table])
    data_spec = tensor_spec.TensorSpec((), dtype=tf.int64)
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec,
        table_name,
        sequence_length=1,
        local_server=reverb_server,
        dataset_buffer_size=1)

    with replay.py_client.trajectory_writer(1) as writer:
      for i in range(3):
        writer.append(i)
        writer.create_item(table_name, trajectory=writer.history[-1:],
                           priority=i)

    dataset = replay.as_dataset(sample_batch_size=3, num_parallel_calls=3)

    self.assertTrue(table.can_sample(3))
    iterator = iter(dataset)
    counts = [0] * 3
    for i in range(10):
      item_0 = next(iterator)[0].numpy()  # This is a matrix shaped 1x3.
      for item in item_0:
        counts[int(item)] += 1
    self.assertFalse(table.can_sample(3))

    # Same number of counts due to limit on max_times_sampled
    self.assertEqual(counts[0], 10)  # priority 0
    self.assertEqual(counts[1], 10)  # priority 1
    self.assertEqual(counts[2], 10)  # priority 2


def _create_add_trajectory_observer_fn(*args, **kwargs):

  @contextlib.contextmanager
  def _create_and_yield(client):
    yield reverb_utils.ReverbAddTrajectoryObserver(client, *args, **kwargs)

  return _create_and_yield


def _create_add_episode_observer_fn(*args, **kwargs):

  @contextlib.contextmanager
  def _create_and_yield(client):
    yield reverb_utils.ReverbAddEpisodeObserver(client, *args, **kwargs)

  return _create_and_yield


def _create_add_sequence_observer_fn(*args, **kwargs):

  @contextlib.contextmanager
  def _create_and_yield(client):
    yield reverb_utils.ReverbTrajectorySequenceObserver(client, *args, **kwargs)

  return _create_and_yield


def _env_creator(episode_len=3):
  return functools.partial(test_envs.CountingEnv, steps_per_episode=episode_len)


def _create_env_spec(episode_len=3):
  return ts.time_step_spec(_env_creator(episode_len)().observation_spec())


def _parallel_env_creator(collection_batch_size=1, episode_len=3):
  return functools.partial(
      parallel_py_environment.ParallelPyEnvironment,
      env_constructors=[
          _env_creator(episode_len) for _ in range(collection_batch_size)
      ])


def _create_random_policy_from_env(env):
  return random_py_policy.RandomPyPolicy(
      ts.time_step_spec(env.observation_spec()), env.action_spec())


class ReverbObserverTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._mock_client = mock.MagicMock()
    self._mock_writer = mock.MagicMock()
    self._mock_client.trajectory_writer = self._mock_writer
    self._mock_writer.return_value = self._mock_writer

    self._table_name = 'uniform_table'
    self._table = reverb.Table(
        self._table_name,
        max_size=100,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))
    self._reverb_server = reverb.Server([self._table], port=None)
    self._reverb_client = self._reverb_server.localhost_client()

  @parameterized.named_parameters(
      (
          'add_trajectory_observer',
          _create_add_trajectory_observer_fn(
              table_name='test_table', sequence_length=2),
          _env_creator(episode_len=3),
          3,  # expected_items
          1,  # writer_call_counts
          4,  # max_steps
          5),  # append_count
      (
          'add_trajectory_episode_observer',
          _create_add_episode_observer_fn(
              table_name='test_table', max_sequence_length=8, priority=3),
          _env_creator(episode_len=3),
          2,  # expected_items
          1,  # writer_call_counts
          8,  # max_steps
          10),  # append_count
      (
          'add_trajectory_observer_stride2',
          _create_add_trajectory_observer_fn(
              table_name='test_table', sequence_length=2, stride_length=2),
          _env_creator(episode_len=3),
          2,  # expected_items
          1,  # writer_call_counts
          4,  # max_steps
          5),  # append_count
      (
          'add_trajectory_observer_with_padding_stride_one',
          _create_add_trajectory_observer_fn(
              table_name='test_table',
              sequence_length=4,
              stride_length=1,
              pad_end_of_episodes=True,
              tile_end_of_episodes=True),
          _env_creator(episode_len=5),
          12,  # expected_items
          1,   # writer_call_counts
          11,  # max_steps
          19,  # append_count
      ),
      (
          'add_trajectory_observer_with_padding_stride_two',
          _create_add_trajectory_observer_fn(
              table_name='test_table',
              sequence_length=4,
              stride_length=2,
              pad_end_of_episodes=True,
              tile_end_of_episodes=True),
          _env_creator(episode_len=5),
          6,  # expected_items
          1,  # writer_call_counts
          11,  # max_steps
          19,  # append_count
      ),
      (
          'add_trajectory_observer_with_padding_stride_three',
          _create_add_trajectory_observer_fn(
              table_name='test_table',
              sequence_length=4,
              stride_length=3,
              pad_end_of_episodes=True,
              tile_end_of_episodes=True),
          _env_creator(episode_len=5),
          4,  # expected_items
          1,  # writer_call_counts
          11,  # max_steps
          19,  # append_count
      ),
      (
          'add_trajectory_observer_with_padding_stride_sequence_length',
          _create_add_trajectory_observer_fn(
              table_name='test_table',
              sequence_length=4,
              stride_length=4,
              pad_end_of_episodes=True,
              tile_end_of_episodes=True),
          _env_creator(episode_len=5),
          4,  # expected_items
          1,  # writer_call_counts
          11,  # max_steps
          19,  # append_count
      ),
      (
          'add_sequence_observer',
          _create_add_sequence_observer_fn(
              table_name='test_table', sequence_length=2, stride_length=2),
          _env_creator(episode_len=3),
          2,  # expected_items
          1,  # writer_call_counts
          4,  # max_steps
          5)  # append_count
  )
  def test_observer_writes(self, create_observer_fn, env_fn, expected_items,
                           writer_call_counts, max_steps, append_count):
    env = env_fn()
    with create_observer_fn(self._mock_client) as observer:
      policy = _create_random_policy_from_env(env)
      driver = py_driver.PyDriver(
          env, policy, observers=[observer], max_steps=max_steps)
      driver.run(env.reset())

    self.assertEqual(writer_call_counts, self._mock_writer.call_count)
    self.assertEqual(append_count, self._mock_writer.append.call_count)
    self.assertEqual(expected_items, self._mock_writer.create_item.call_count)

  @parameterized.named_parameters(
      (
          'add_trajectory_observer_reset_without_writing_cache',
          _create_add_trajectory_observer_fn(
              table_name='test_table', sequence_length=4, stride_length=4),
          False,  # reset_with_write_cached_steps
          13,  # append_count
          2,  # expected_items
          0,  # append_count_from_reset
          0,  # expected_items_from_reset
      ),
      (
          'add_trajectory_observer_reset_with_writing_cache_with_padding',
          _create_add_trajectory_observer_fn(
              table_name='test_table',
              sequence_length=4,
              stride_length=4,
              pad_end_of_episodes=True,
              tile_end_of_episodes=True),
          True,  # reset_with_write_cached_steps
          19,  # append_count
          4,  # expected_items
          3,  # append_count_from_reset
          1,  # expected_items_from_reset
      ),
      (
          'add_trajectory_observer_reset_writing_cache_padding_no_tile',
          _create_add_trajectory_observer_fn(
              table_name='test_table',
              sequence_length=4,
              stride_length=4,
              pad_end_of_episodes=True,
              tile_end_of_episodes=False,
          ),
          True,  # reset_with_write_cached_steps
          13,  # append_count
          2,  # expected_items
          3,  # append_count_from_reset
          1,  # expected_items_from_reset
      ),
  )
  def test_observer_resets(self, create_observer_fn,
                           reset_with_write_cached_steps, append_count,
                           expected_items, append_count_from_reset,
                           expected_items_from_reset):
    env = _env_creator(5)()
    with create_observer_fn(self._mock_client) as observer:
      policy = _create_random_policy_from_env(env)
      driver = py_driver.PyDriver(
          env, policy, observers=[observer], max_steps=11)
      driver.run(env.reset())

      self.assertEqual(append_count, self._mock_writer.append.call_count)
      self.assertEqual(expected_items, self._mock_writer.create_item.call_count)
      observer.reset(write_cached_steps=reset_with_write_cached_steps)
      self.assertEqual(append_count + append_count_from_reset,
                       self._mock_writer.append.call_count)
      self.assertEqual(expected_items + expected_items_from_reset,
                       self._mock_writer.create_item.call_count)

  def test_observer_writes_multi_tables(self):
    episode_length = 3
    collect_step_count = 6
    table_count = 2
    create_observer_fn = _create_add_sequence_observer_fn(
        table_name=['test_table1', 'test_table2'],
        sequence_length=episode_length,
        stride_length=episode_length)
    env = _env_creator(episode_length)()
    with create_observer_fn(self._mock_client) as observer:
      policy = _create_random_policy_from_env(env)
      driver = py_driver.PyDriver(
          env, policy, observers=[observer], max_steps=collect_step_count)
      driver.run(env.reset())

    self.assertEqual(table_count * int(collect_step_count / episode_length),
                     self._mock_writer.create_item.call_count)

  def test_trajectory_observer_no_mock(self):
    create_observer_fn = _create_add_trajectory_observer_fn(
        table_name=self._table_name,
        sequence_length=2)
    env = _env_creator(episode_len=6)()

    self._reverb_client.reset(self._table_name)
    with create_observer_fn(self._reverb_client) as observer:
      policy = _create_random_policy_from_env(env)
      driver = py_driver.PyDriver(
          env, policy, observers=[observer], max_steps=5)
      driver.run(env.reset())
      # Give it some time for the items to reach Reverb.
      time.sleep(1)

      self.assertEqual(observer._cached_steps, 5)
      self.assertEqual(self._table.info.current_size, 4)

  def test_episodic_observer_overflow_episode_bypass(self):
    env1 = _env_creator(episode_len=3)()
    env2 = _env_creator(episode_len=4)()
    with _create_add_episode_observer_fn(
        table_name='test_table', max_sequence_length=4,
        priority=1,
        bypass_partial_episodes=True)(self._mock_client) as observer:
      policy = _create_random_policy_from_env(env1)
      # env1 -> writes only ONE episode. Note that `max_sequence_length`
      # must be one more than episode length. As in TF-Agents, we append
      # a trajectory as the `LAST` step.
      driver = py_driver.PyDriver(
          env1, policy, observers=[observer], max_steps=6)
      driver.run(env1.reset())
      # env2 -> writes NO episodes (all of them has length >
      # `max_sequence_length`)
      policy = _create_random_policy_from_env(env2)
      driver = py_driver.PyDriver(
          env2, policy, observers=[observer], max_steps=6)
      driver.run(env2.reset())
    self.assertEqual(1, self._mock_writer.create_item.call_count)

  def test_episodic_observer_overflow_episode_raise_value_error(self):
    env = _env_creator(episode_len=3)()
    with _create_add_episode_observer_fn(
        table_name='test_table', max_sequence_length=2,
        priority=1)(self._mock_client) as observer:
      policy = _create_random_policy_from_env(env)
      driver = py_driver.PyDriver(
          env, policy, observers=[observer], max_steps=4)
      with self.assertRaises(ValueError):
        driver.run(env.reset())

  def test_episodic_observer_assert_sequence_length_positive(self):
    with self.assertRaises(ValueError):
      _ = reverb_utils.ReverbAddEpisodeObserver(
          self._mock_client,
          table_name='test_table',
          max_sequence_length=-1,
          priority=3)

  def test_episodic_observer_update_priority(self):
    observer = reverb_utils.ReverbAddEpisodeObserver(
        self._mock_client,
        table_name='test_table',
        max_sequence_length=1,
        priority=3)
    self.assertEqual(observer._priority, 3)
    observer.update_priority(4)
    self.assertEqual(observer._priority, 4)

  def test_episodic_observer_no_mock(self):
    create_observer_fn = _create_add_episode_observer_fn(
        table_name=self._table_name,
        max_sequence_length=8,
        priority=3)
    env = _env_creator(episode_len=3)()

    self._reverb_client.reset(self._table_name)
    with create_observer_fn(self._reverb_client) as observer:
      policy = _create_random_policy_from_env(env)
      driver = py_driver.PyDriver(
          env, policy, observers=[observer], max_steps=10)
      driver.run(env.reset())
      # Give it some time for the items to reach Reverb.
      time.sleep(1)

      # We run the driver for 3 full episode and one step.
      self.assertEqual(observer._cached_steps, 1)
      self.assertEqual(self._table.info.current_size, 3)


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
