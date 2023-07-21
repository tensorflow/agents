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

"""Tests for tf_agents.replay_buffers.reverb_replay_buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import reverb
import tensorflow as tf

from tf_agents.drivers import py_driver
from tf_agents.environments import test_envs
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import test_utils


class ReverbReplayBufferTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(ReverbReplayBufferTest, self).setUp()

    # Prepare the environment (and the corresponding specs).
    self._env = test_envs.EpisodeCountingEnv(steps_per_episode=3)
    tensor_time_step_spec = tf.nest.map_structure(tensor_spec.from_spec,
                                                  self._env.time_step_spec())
    tensor_action_spec = tensor_spec.from_spec(self._env.action_spec())
    self._data_spec = trajectory.Trajectory(
        step_type=tensor_time_step_spec.step_type,
        observation=tensor_time_step_spec.observation,
        action=tensor_action_spec,
        policy_info=(),
        next_step_type=tensor_time_step_spec.step_type,
        reward=tensor_time_step_spec.reward,
        discount=tensor_time_step_spec.discount,
    )
    # TODO(b/188427258) Add time dimension when using Reverb.TrajectoryWriters.
    # table_signature = tensor_spec.add_outer_dim(self._data_spec)
    self._array_data_spec = tensor_spec.to_nest_array_spec(self._data_spec)

    # Initialize and start a Reverb server (and set up a client to it).
    self._table_name = 'test_table'
    uniform_table = reverb.Table(
        self._table_name,
        max_size=100,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        # signature=table_signature,
    )
    self._server = reverb.Server([uniform_table])
    self._py_client = reverb.Client('localhost:{}'.format(self._server.port))

  def tearDown(self):
    if self._server:
      # Stop the Reverb server if it is running.
      self._server.stop()
      self._server = None
    super(ReverbReplayBufferTest, self).tearDown()

  def _insert_random_data(self,
                          env,
                          num_steps,
                          sequence_length=2,
                          additional_observers=None):
    """Insert `num_step` random observations into Reverb server."""
    observers = [] if additional_observers is None else additional_observers
    traj_obs = reverb_utils.ReverbAddTrajectoryObserver(
        self._py_client, self._table_name, sequence_length=sequence_length)
    observers.append(traj_obs)
    policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                             env.action_spec())
    driver = py_driver.PyDriver(env,
                                policy,
                                observers=observers,
                                max_steps=num_steps)
    time_step = env.reset()
    driver.run(time_step)
    traj_obs.close()

  @parameterized.named_parameters(
      ('_sequence_length_none', None),
      ('_sequence_length_eq_num_steps', 2),
      ('_sequence_length_gt_num_steps', 4))
  def test_dataset_samples_sequential(self, sequence_length):

    def validate_data_observer(traj):
      if not array_spec.check_arrays_nest(traj, self._array_data_spec):
        raise ValueError('Trajectory incompatible with array_data_spec')

    # Observe 20 steps from the env.  This isn't the num_steps we're testing.
    self._insert_random_data(
        self._env, num_steps=20,
        additional_observers=[validate_data_observer],
        sequence_length=sequence_length or 4)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec, self._table_name, local_server=self._server,
        sequence_length=sequence_length)

    # Make sure observations belong to the same episode and their step are off
    # by 1.
    for sample, _ in replay.as_dataset(num_steps=2).take(100):
      episode, step = sample.observation
      self.assertEqual(episode[0], episode[1])
      self.assertEqual(step[0] + 1, step[1])

  def test_dataset_with_variable_sequence_length_truncates(self):
    spec = tf.TensorSpec((), tf.int64)
    table_spec = tf.TensorSpec((None,), tf.int64)
    table = reverb.Table(
        name=self._table_name,
        sampler=reverb.selectors.Fifo(),
        remover=reverb.selectors.Fifo(),
        max_times_sampled=1,
        max_size=100,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=table_spec,
    )
    server = reverb.Server([table])
    py_client = reverb.Client('localhost:{}'.format(server.port))

    # Insert two episodes: one of length 3 and one of length 5
    with py_client.trajectory_writer(10) as writer:
      writer.append(1)
      writer.append(2)
      writer.append(3)
      writer.create_item(
          self._table_name, trajectory=writer.history[-3:], priority=5)

    with py_client.trajectory_writer(10) as writer:
      writer.append(10)
      writer.append(20)
      writer.append(30)
      writer.append(40)
      writer.append(50)
      writer.create_item(
          self._table_name, trajectory=writer.history[-5:], priority=5)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        spec, self._table_name, local_server=server, sequence_length=None,
        rate_limiter_timeout_ms=100)
    ds = replay.as_dataset(single_deterministic_pass=True, num_steps=2)
    it = iter(ds)

    # Expect [1, 2]
    data, _ = next(it)
    self.assertAllEqual(data, [1, 2])

    # Expect [10, 20]
    data, _ = next(it)
    self.assertAllEqual(data, [10, 20])

    # Expect [30, 40]
    data, _ = next(it)
    self.assertAllEqual(data, [30, 40])

    with self.assertRaises(StopIteration):
      next(it)

  def test_dataset_with_preprocess(self):

    def validate_data_observer(traj):
      if not array_spec.check_arrays_nest(traj, self._array_data_spec):
        raise ValueError('Trajectory incompatible with array_data_spec')

    def preprocess(traj):
      episode, step = traj.observation
      return traj.replace(observation=(episode, step + 1))

    # Observe 10 steps from the env.  This isn't the num_steps we're testing.
    self._insert_random_data(
        self._env,
        num_steps=10,
        additional_observers=[validate_data_observer],
        sequence_length=4)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        local_server=self._server,
        sequence_length=4)

    dataset = replay.as_dataset(num_steps=2)
    for sample, _ in dataset.take(5):
      episode, step = sample.observation
      self.assertEqual(episode[0], episode[1])
      self.assertEqual(step[0] + 1, step[1])
      # From even to odd steps
      self.assertEqual(0, step[0].numpy() % 2)
      self.assertEqual(1, step[1].numpy() % 2)

    dataset = replay.as_dataset(
        num_steps=2, sample_batch_size=1, sequence_preprocess_fn=preprocess)
    for sample, _ in dataset.take(5):
      episode, step = sample.observation
      self.assertEqual(episode[0, 0], episode[0, 1])
      self.assertEqual(step[0, 0] + 1, step[0, 1])
      # Makes sure the preprocess has happened.
      # From odd to even steps
      self.assertEqual(1, step[0, 0].numpy() % 2)
      self.assertEqual(0, step[0, 1].numpy() % 2)

  def test_single_episode_dataset(self):
    sequence_length = 3
    self._insert_random_data(
        self._env,
        num_steps=sequence_length,
        sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=None,
        local_server=self._server)

    # Make sure observations are off by 1 given we are counting transitions in
    # the env observations.
    dataset = replay.as_dataset()
    for sample, _ in dataset.take(5):
      episode, step = sample.observation
      self.assertEqual((sequence_length,), episode.shape)
      self.assertEqual((sequence_length,), step.shape)
      self.assertAllEqual([0] * sequence_length, episode - episode[:1])
      self.assertAllEqual(list(range(sequence_length)), step - step[:1])

  def test_variable_length_episodes_dataset(self):
    # Add one episode of each length.
    for sequence_length in range(1, 10):
      env = test_envs.EpisodeCountingEnv(steps_per_episode=sequence_length)
      self._insert_random_data(
          env,
          num_steps=sequence_length,
          sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=None,
        local_server=self._server)

    # Make sure observations are off by 1 given we are counting transitions in
    # the env observations.
    dataset = replay.as_dataset(sample_batch_size=1)
    for sample, _ in dataset.take(5):
      episode, step = sample.observation
      self.assertIn(episode.shape[1], range(1, 10))
      self.assertIn(step.shape[1], range(1, 10))
      length = episode.shape[1]
      # All Episode id are 0.
      self.assertAllEqual([[0] * length], episode)
      # Steps id is sequential up its length.
      self.assertAllEqual([list(range(length))], step)

  @parameterized.named_parameters(
      ('_sequence_length_1', 1),
      ('_sequence_length_2', 2),
      ('_sequence_length_5', 5))
  def test_batched_episodes_dataset(self, sequence_length):
    # Observe batch_size * sequence_length steps to have at least 3 episodes
    batch_size = 3
    env = test_envs.EpisodeCountingEnv(steps_per_episode=sequence_length)
    self._insert_random_data(
        env,
        num_steps=batch_size * sequence_length,
        sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=None,
        local_server=self._server)

    dataset = replay.as_dataset(batch_size)
    for sample, _ in dataset.take(5):
      episode, step = sample.observation
      self.assertEqual((batch_size, sequence_length), episode.shape)
      self.assertEqual((batch_size, sequence_length), step.shape)
      for n in range(sequence_length):
        # All elements in the same batch should belong to the same episode.
        self.assertAllEqual(episode[:, 0], episode[:, n])
        # All elements in the same batch should have consecutive steps.
        self.assertAllEqual(step[:, 0] + n, step[:, n])

  @parameterized.named_parameters(
      ('_num_steps_1', 1),
      ('_num_steps_2', 2),
      ('_num_steps_5', 5),
      ('_num_steps_10', 10),
      ('_num_steps_None', None))
  def test_sequential_ordering(self, num_steps):
    sequence_length = 10
    batch_size = 5
    env = test_envs.EpisodeCountingEnv(steps_per_episode=sequence_length)
    self._insert_random_data(
        env,
        num_steps=batch_size * sequence_length,
        sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=sequence_length,
        local_server=self._server)

    dataset = replay.as_dataset(batch_size, num_steps=num_steps)
    num_steps = num_steps or sequence_length
    for sample, _ in dataset.take(10):
      episode, step = sample.observation
      self.assertEqual((batch_size, num_steps), episode.shape)
      self.assertEqual((batch_size, num_steps), step.shape)
      for n in range(num_steps):
        # All elements in the same batch should belong to the same episode.
        self.assertAllEqual(episode[:, 0], episode[:, n])
        # All elements in the batch should have consecutive steps.
        self.assertAllEqual(step[:, 0] + n, step[:, n])

  def test_sample_single_episode(self):
    num_episodes = 1
    sequence_length = 100
    batch_size = 10
    num_steps = 5
    env = test_envs.EpisodeCountingEnv(steps_per_episode=sequence_length)
    # Insert only one episode in the RB.
    self._insert_random_data(
        env,
        num_steps=num_episodes * sequence_length,
        sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=sequence_length,
        local_server=self._server)

    dataset = replay.as_dataset(batch_size, num_steps=num_steps)
    n_samples = 0
    for sample, _ in dataset.take(10):
      n_samples += 1
      episode, step = sample.observation
      # The episode should always be 0.
      episode_id = tf.constant(0, dtype=episode.dtype, shape=episode.shape)
      # All elements in the same batch should belong to the same episode.
      self.assertAllEqual(episode_id, episode)
      for n in range(num_steps):
        # All elements in the batch should have consecutive steps.
        self.assertAllEqual(step[:, 0] + n, step[:, n])
    # Ensure we can actually sampled 10 times.
    self.assertEqual(10, n_samples)

  def test_capacity_set(self):
    table_name = 'test_table'
    capacity = 100

    uniform_table = reverb.Table(
        table_name,
        max_size=capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(3))
    server = reverb.Server([uniform_table])
    data_spec = tensor_spec.TensorSpec((), tf.float32)
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec, table_name, local_server=server, sequence_length=None)

    self.assertEqual(capacity, replay.capacity)
    server.stop()

  def test_size_empty(self):
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec, self._table_name, local_server=self._server,
        sequence_length=None)
    self.assertEqual(replay.num_frames(), 0)

  @parameterized.named_parameters(
      ('_sequence_length_none', None),
      ('_sequence_length_eq_num_steps', 20))
  def test_size_with_data_inserted(self, sequence_length):
    num_steps = 20
    self._insert_random_data(self._env, num_steps=num_steps)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec, self._table_name, local_server=self._server,
        sequence_length=sequence_length)

    # The number of observations are off by 1 given we are counting transitions
    # in the env observations.
    self.assertEqual(replay.num_frames(), 19)

  def test_raises_if_ask_for_num_steps_gt_sequence_length(self):
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec, self._table_name, local_server=self._server,
        sequence_length=2)

    with self.assertRaisesRegex(ValueError, r'num_steps > sequence_length'):
      replay.as_dataset(num_steps=4)

  def test_raises_if_ask_for_num_steps_not_multiple_sequence_length(self):
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec, self._table_name, local_server=self._server,
        sequence_length=4)

    with self.assertRaisesRegex(ValueError, r'not a multiple of num_steps'):
      replay.as_dataset(num_steps=3)

  def test_raises_deterministic_dataset_from_random_table(self):
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec, self._table_name, local_server=self._server,
        sequence_length=None)

    with self.assertRaisesRegex(
        ValueError, r'either the sampler or the remover is not deterministic'):
      replay.as_dataset(single_deterministic_pass=True)

  def test_deterministic_dataset_from_heap_sampler_remover(self):

    uniform_sampler_min_heap_remover_table = reverb.Table(
        name=self._table_name,
        sampler=reverb.selectors.MaxHeap(),
        remover=reverb.selectors.MinHeap(),
        max_size=100,
        max_times_sampled=0,
        rate_limiter=reverb.rate_limiters.MinSize(1))
    server = reverb.Server([uniform_sampler_min_heap_remover_table])
    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        local_server=server,
        sequence_length=None)
    replay.as_dataset(single_deterministic_pass=True)
    server.stop()

  @parameterized.named_parameters(
      ('_default', tf.distribute.get_strategy()),
      ('_one_device', tf.distribute.OneDeviceStrategy('/cpu:0')),
      ('_mirrored', tf.distribute.MirroredStrategy(devices=('/cpu:0',
                                                            '/cpu:1'))))
  def test_experimental_distribute_dataset(self, strategy):
    sequence_length = 3
    batch_size = 10
    self._insert_random_data(
        self._env,
        num_steps=sequence_length,
        sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=sequence_length,
        local_server=self._server)

    dataset = replay.as_dataset(batch_size)

    with strategy.scope():
      dataset = strategy.experimental_distribute_dataset(dataset)
      iterator = iter(dataset)

    def train_step():
      with strategy.scope():
        sample, _ = next(iterator)
        _, step = sample.observation
        loss = strategy.run(lambda x: tf.reduce_mean(x, axis=-1), args=(step,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=0)

    # Test running eagerly
    for _ in range(5):
      with strategy.scope():
        loss = train_step()
        self.assertEqual(batch_size, loss)

    # Test with wrapping into a tf.function
    train_step_fn = common.function(train_step)
    for _ in range(5):
      with strategy.scope():
        loss = train_step_fn()
        self.assertEqual(batch_size, loss)

  @parameterized.named_parameters(
      ('_default', tf.distribute.get_strategy()),
      ('_one_device', tf.distribute.OneDeviceStrategy('/cpu:0')),
      ('_mirrored', tf.distribute.MirroredStrategy(devices=('/cpu:0',
                                                            '/cpu:1'))))
  def test_experimental_distribute_datasets_from_function(self, strategy):
    sequence_length = 3
    batch_size = 10
    self._insert_random_data(
        self._env,
        num_steps=sequence_length,
        sequence_length=sequence_length)

    replay = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        self._table_name,
        sequence_length=sequence_length,
        local_server=self._server)

    num_replicas = strategy.num_replicas_in_sync
    with strategy.scope():
      dataset = strategy.experimental_distribute_datasets_from_function(
          lambda _: replay.as_dataset(batch_size // num_replicas))
      iterator = iter(dataset)

    @common.function()
    def train_step():
      with strategy.scope():
        sample, _ = next(iterator)
        _, step = sample.observation
        loss = strategy.run(lambda x: tf.reduce_mean(x, axis=-1), args=(step,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=0)

    # Test running eagerly
    for _ in range(5):
      with strategy.scope():
        loss = train_step()
        self.assertEqual(batch_size, loss)

    # Test with wrapping into a tf.function
    train_step_fn = common.function(train_step)
    for _ in range(5):
      with strategy.scope():
        loss = train_step_fn()
        self.assertEqual(batch_size, loss)


if __name__ == '__main__':
  test_utils.main()
