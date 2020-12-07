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
"""Tests for tf_agents.train.strategy_utils."""

from absl.testing.absltest import mock
import tensorflow.compat.v2 as tf

from tf_agents.train.utils import strategy_utils
from tf_agents.utils import test_utils


class StrategyUtilsTest(test_utils.TestCase):

  def test_get_distribution_strategy_default(self):
    # Get a default strategy to compare against.
    default_strategy = tf.distribute.get_strategy()

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
    self.assertIsInstance(strategy, type(default_strategy))

  @mock.patch.object(tf.distribute.experimental, 'TPUStrategy')
  @mock.patch.object(tf.tpu.experimental, 'initialize_tpu_system')
  @mock.patch.object(tf.config, 'experimental_connect_to_cluster')
  @mock.patch.object(tf.distribute.cluster_resolver, 'TPUClusterResolver')
  def test_tpu_strategy(self, mock_tpu_cluster_resolver,
                        mock_experimental_connect_to_cluster,
                        mock_initialize_tpu_system, mock_tpu_strategy):
    resolver = mock.MagicMock()
    mock_tpu_cluster_resolver.return_value = resolver
    mock_strategy = mock.MagicMock()
    mock_tpu_strategy.return_value = mock_strategy

    strategy = strategy_utils.get_strategy(tpu='bns_address', use_gpu=False)

    mock_tpu_cluster_resolver.assert_called_with(tpu='bns_address')
    mock_experimental_connect_to_cluster.assert_called_with(resolver)
    mock_initialize_tpu_system.assert_called_with(resolver)
    self.assertIs(strategy, mock_strategy)

  @mock.patch.object(tf.distribute, 'MirroredStrategy')
  def test_mirrored_strategy(self, mock_mirrored_strategy):
    mirrored_strategy = mock.MagicMock()
    mock_mirrored_strategy.return_value = mirrored_strategy

    strategy = strategy_utils.get_strategy(False, use_gpu=True)
    self.assertIs(strategy, mirrored_strategy)


if __name__ == '__main__':
  test_utils.main()
