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

"""Tests for tf_agents.bandits.networks.global_and_arm_feature_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.networks import global_and_arm_feature_network as gafn
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import test_utils


parameters = parameterized.named_parameters(
    {
        'testcase_name': 'batch2feat4act3',
        'batch_size': 2,
        'feature_dim': 4,
        'num_actions': 3
    }, {
        'testcase_name': 'batch1feat7act9',
        'batch_size': 1,
        'feature_dim': 7,
        'num_actions': 9
    })


class GlobalAndArmFeatureNetworkTest(parameterized.TestCase,
                                     test_utils.TestCase):

  @parameters
  def testCreateFeedForwardCommonTowerNetwork(self, batch_size, feature_dim,
                                              num_actions):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        7, feature_dim, num_actions)
    net = gafn.create_feed_forward_common_tower_network(obs_spec, (4, 3, 2),
                                                        (6, 5, 4), (7, 6, 5))
    input_nest = tensor_spec.sample_spec_nest(
        obs_spec, outer_dims=(batch_size,))
    output, _ = net(input_nest)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    output = self.evaluate(output)
    self.assertAllEqual(output.shape, (batch_size, num_actions))

  @parameters
  def testCreateFeedForwardDotProductNetwork(self, batch_size, feature_dim,
                                             num_actions):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        7, feature_dim, num_actions)
    net = gafn.create_feed_forward_dot_product_network(obs_spec, (4, 3, 4),
                                                       (6, 5, 4))
    input_nest = tensor_spec.sample_spec_nest(
        obs_spec, outer_dims=(batch_size,))
    output, _ = net(input_nest)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    output = self.evaluate(output)
    self.assertAllEqual(output.shape, (batch_size, num_actions))

  def testCreateFeedForwardCommonTowerNetworkWithFeatureColumns(
      self, batch_size=2, feature_dim=4, num_actions=3):
    obs_spec = {
        'global': {
            'dense':
                tensor_spec.TensorSpec(shape=(feature_dim,), dtype=tf.float32),
            'composer':
                tensor_spec.TensorSpec((), tf.string)
        },
        'per_arm': {
            'name': tensor_spec.TensorSpec((num_actions,), tf.string),
            'fruit': tensor_spec.TensorSpec((num_actions,), tf.string)
        }
    }
    columns_dense = tf.feature_column.numeric_column(
        'dense', shape=(feature_dim,))
    columns_composer = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'composer', ['wolfgang', 'amadeus', 'mozart']))

    columns_name = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'name', ['bob', 'george', 'wanda']))
    columns_fruit = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'fruit', ['banana', 'kiwi', 'pear']))

    net = gafn.create_feed_forward_common_tower_network(
        observation_spec=obs_spec,
        global_layers=(4, 3, 2),
        arm_layers=(6, 5, 4),
        common_layers=(7, 6, 5),
        global_preprocessing_combiner=tf.compat.v2.keras.layers.DenseFeatures(
            [columns_dense, columns_composer]),
        arm_preprocessing_combiner=tf.compat.v2.keras.layers.DenseFeatures(
            [columns_name, columns_fruit]))
    input_nest = {
        'global': {
            'dense':
                tf.constant(
                    np.random.rand(batch_size, feature_dim).astype(np.float32)),
            'composer':
                tf.constant(['wolfgang', 'mozart'])
        },
        'per_arm': {
            'name':
                tf.constant([[['george'], ['george'], ['george']],
                             [['bob'], ['bob'], ['bob']]]),
            'fruit':
                tf.constant([[['banana'], ['banana'], ['banana']],
                             [['kiwi'], ['kiwi'], ['kiwi']]])
        }
    }

    output, _ = net(input_nest)
    self.evaluate([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.tables_initializer()
    ])
    output = self.evaluate(output)
    self.assertAllEqual(output.shape, (batch_size, num_actions))


if __name__ == '__main__':
  tf.test.main()
