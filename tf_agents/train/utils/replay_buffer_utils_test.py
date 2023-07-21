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

"""Tests for tf_agents.train.replay_buffer_utils."""

import tensorflow as tf

from tf_agents import specs
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train.utils import replay_buffer_utils
from tf_agents.utils import lazy_loader
from tf_agents.utils import test_utils

# Lazy loading since not all users have the reverb package installed.
reverb = lazy_loader.LazyLoader('reverb', globals(), 'reverb')


class ReplayBufferUtilsTest(test_utils.TestCase):

  def _get_mock_spec(self):
    spec = [
        specs.TensorSpec([3], tf.float32, 'action'), [
            specs.TensorSpec([5], tf.float32, 'lidar'),
            specs.TensorSpec([3, 2], tf.float32, 'camera')
        ]
    ]
    return spec

  def test_returns_correct_instances(self):
    rb, observer = replay_buffer_utils.get_reverb_buffer_and_observer(
        self._get_mock_spec(),
        sequence_length=1,
        port=None)
    self.assertIsInstance(rb, reverb_replay_buffer.ReverbReplayBuffer)
    self.assertIsInstance(observer, reverb_utils.ReverbAddTrajectoryObserver)
    rb.local_server.stop()

  def test_non_default_table(self):
    table_name = 'test_prioritized_table'
    test_table = reverb.Table(
        table_name,
        max_size=333,
        sampler=reverb.selectors.Prioritized(1.0),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))
    rb, _ = replay_buffer_utils.get_reverb_buffer_and_observer(
        self._get_mock_spec(),
        sequence_length=1,
        table_name=table_name,
        table=test_table,
        port=None)

    server_info = rb._py_client.server_info()
    self.assertEqual(
        server_info['test_prioritized_table'].name,
        'test_prioritized_table')
    self.assertEqual(
        server_info['test_prioritized_table'].max_size, 333)
    rb.local_server.stop()

if __name__ == '__main__':
  test_utils.main()
