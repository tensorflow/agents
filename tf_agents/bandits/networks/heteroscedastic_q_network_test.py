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

"""Tests for tf_agents.network.heteroscedastic_q_network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.specs import tensor_spec


class SingleObservationSingleActionTest(tf.test.TestCase):

  def setUp(self):
    super(SingleObservationSingleActionTest, self).setUp()
    gin.clear_config()

  def testBuild(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    preds, _ = network(states)
    q_values, log_variances = preds.q_value_logits, preds.log_variance
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertAllEqual(log_variances.shape.as_list(),
                        [batch_size, num_actions])
    self.assertEqual(len(network.trainable_weights), 8)

  def testChangeHiddenLayers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        fc_layer_params=(40,))
    preds, _ = network(states)
    q_values, log_variances = preds.q_value_logits, preds.log_variance
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertAllEqual(log_variances.shape.as_list(),
                        [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 6)

  def testAddConvLayers(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, 5, 5, num_state_dims])
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([5, 5, num_state_dims],
                                                 tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        conv_layer_params=((16, 3, 2),))
    preds, _ = network(states)
    q_values, log_variances = preds.q_value_logits, preds.log_variance
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertAllEqual(log_variances.shape.as_list(),
                        [batch_size, num_actions])
    self.assertEqual(len(network.trainable_variables), 10)

  def testAddPreprocessingLayers(self):
    batch_size = 3
    num_actions = 2
    states = (tf.random.uniform([batch_size,
                                 1]), tf.random.uniform([batch_size]))
    preprocessing_layers = (tf.keras.layers.Dense(4),
                            tf.keras.Sequential([
                                tf.keras.layers.Reshape((1,)),
                                tf.keras.layers.Dense(4)
                            ]))
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=(tensor_spec.TensorSpec([1], tf.float32),
                           tensor_spec.TensorSpec([], tf.float32)),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add(),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0,
                                                  num_actions - 1))
    preds, _ = network(states)
    q_values, log_variances = preds.q_value_logits, preds.log_variance
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertAllEqual(log_variances.shape.as_list(),
                        [batch_size, num_actions])
    # At least 2 variables each for the preprocessing layers.
    self.assertGreater(len(network.trainable_variables), 6)

  def testCorrectOutputShape(self):
    batch_size = 3
    num_state_dims = 5
    num_actions = 2
    states = tf.random.uniform([batch_size, num_state_dims])
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    preds, _ = network(states)
    q_values, log_variances = preds.q_value_logits, preds.log_variance
    self.assertAllEqual(q_values.shape.as_list(), [batch_size, num_actions])
    self.assertAllEqual(log_variances.shape.as_list(),
                        [batch_size, num_actions])

  def testNetworkVariablesAreReused(self):
    batch_size = 3
    num_state_dims = 5
    states = tf.ones([batch_size, num_state_dims])
    next_states = tf.ones([batch_size, num_state_dims])
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1))
    preds, _ = network(states)
    q_values, log_variances = preds.q_value_logits, preds.log_variance
    preds, _ = network(next_states)
    next_q_values, next_log_variances = preds.q_value_logits, preds.log_variance
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(q_values, next_q_values)
    self.assertAllClose(log_variances, next_log_variances)

  def testNumericFeatureColumnInput(self):
    key = 'feature_key'
    batch_size = 3
    state_dims = 5
    column = tf.feature_column.numeric_column(key, [state_dims])
    state = {key: tf.ones([batch_size, state_dims], tf.int32)}
    state_spec = {key: tensor_spec.TensorSpec([state_dims], tf.int32)}

    dense_features = tf.compat.v2.keras.layers.DenseFeatures([column])
    online_network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=state_spec,
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        preprocessing_combiner=dense_features)
    target_network = online_network.copy(name='TargetNetwork')
    q_online = online_network(state)[0].q_value_logits
    q_target = target_network(state)[0].q_value_logits
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(q_online, q_target, rtol=1.0, atol=1.0)

  def testIndicatorFeatureColumnInput(self):
    key = 'feature_key'
    vocab_list = [2, 3, 4]
    column = tf.feature_column.categorical_column_with_vocabulary_list(
        key, vocab_list)
    column = tf.feature_column.indicator_column(column)
    feature_tensor = tf.convert_to_tensor([3, 2, 2, 4, 3])
    state = {key: tf.expand_dims(feature_tensor, -1)}
    state_spec = {key: tensor_spec.TensorSpec([1], tf.int32)}

    dense_features = tf.compat.v2.keras.layers.DenseFeatures([column])
    online_network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=state_spec,
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        preprocessing_combiner=dense_features)
    target_network = online_network.copy(name='TargetNetwork')
    q_online = online_network(state)[0].q_value_logits
    q_target = target_network(state)[0].q_value_logits
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.initializers.tables_initializer())
    self.assertAllClose(q_online, q_target, rtol=1.0, atol=1.0)

  def testEmbeddingFeatureColumnInput(self):
    key = 'feature_key'
    vocab_list = ['a', 'b']
    column = tf.feature_column.categorical_column_with_vocabulary_list(
        key, vocab_list)
    column = tf.feature_column.embedding_column(column, 3)
    feature_tensor = tf.convert_to_tensor(['a', 'b', 'c', 'a', 'c'])
    state = {key: tf.expand_dims(feature_tensor, -1)}
    state_spec = {key: tensor_spec.TensorSpec([1], tf.string)}

    dense_features = tf.compat.v2.keras.layers.DenseFeatures([column])
    online_network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=state_spec,
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        preprocessing_combiner=dense_features)
    target_network = online_network.copy(name='TargetNetwork')
    q_online = online_network(state)[0].q_value_logits
    q_target = target_network(state)[0].q_value_logits
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.initializers.tables_initializer())
    self.assertAllClose(q_online, q_target, rtol=1.0, atol=1.0)

  def testCombinedFeatureColumnInput(self):
    columns = {}
    state_tensors = {}
    state_specs = {}
    expected_dim = 0

    indicator_key = 'indicator_key'
    vocab_list = [2, 3, 4]
    column1 = tf.feature_column.categorical_column_with_vocabulary_list(
        indicator_key, vocab_list)
    columns[indicator_key] = tf.feature_column.indicator_column(column1)
    state_tensors[indicator_key] = tf.expand_dims([3, 2, 2, 4, 3], -1)
    state_specs[indicator_key] = tensor_spec.TensorSpec([1], tf.int32)
    expected_dim += len(vocab_list)

    embedding_key = 'embedding_key'
    embedding_dim = 3
    vocab_list = [2, 3, 4]
    column2 = tf.feature_column.categorical_column_with_vocabulary_list(
        embedding_key, vocab_list)
    columns[embedding_key] = tf.feature_column.embedding_column(
        column2, embedding_dim)
    state_tensors[embedding_key] = tf.expand_dims([3, 2, 2, 4, 3], -1)
    state_specs[embedding_key] = tensor_spec.TensorSpec([1], tf.int32)
    expected_dim += embedding_dim

    numeric_key = 'numeric_key'
    batch_size = 5
    state_dims = 3
    input_shape = (batch_size, state_dims)
    columns[numeric_key] = tf.feature_column.numeric_column(
        numeric_key, [state_dims])
    state_tensors[numeric_key] = tf.ones(input_shape, tf.int32)
    state_specs[numeric_key] = tensor_spec.TensorSpec([state_dims], tf.int32)
    expected_dim += state_dims

    num_actions = 4
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0,
                                                num_actions - 1)
    dense_features = tf.compat.v2.keras.layers.DenseFeatures(columns.values())
    online_network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        state_specs, action_spec, preprocessing_combiner=dense_features)
    target_network = online_network.copy(name='TargetNetwork')
    q_online = online_network(state_tensors)[0].q_value_logits
    q_target = target_network(state_tensors)[0].q_value_logits
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tf.compat.v1.initializers.tables_initializer())

    expected_shape = (batch_size, num_actions)
    self.assertEqual(expected_shape, q_online.shape)
    self.assertEqual(expected_shape, q_target.shape)
    self.assertAllClose(q_online, q_target, rtol=1.0, atol=1.0)

  def testPreprocessingLayersSingleObservations(self):
    """Tests using preprocessing_layers without preprocessing_combiner."""
    num_state_dims = 5
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        preprocessing_layers=tf.keras.layers.Lambda(lambda x: x),
        preprocessing_combiner=None)
    preds, _ = network(tf.ones((3, num_state_dims)))
    q_logits, log_variances = preds.q_value_logits, preds.log_variance
    self.assertAllEqual(q_logits.shape.as_list(), [3, 2])
    self.assertAllEqual(log_variances.shape.as_list(), [3, 2])

  def testVarianceBoundaryConditions(self):
    """Tests that min/max variance conditions are satisfied."""
    batch_size = 3
    num_state_dims = 5
    min_variance = 1.0
    max_variance = 2.0
    eps = 0.0001
    states = tf.random.uniform([batch_size, num_state_dims])
    network = heteroscedastic_q_network.HeteroscedasticQNetwork(
        input_tensor_spec=tensor_spec.TensorSpec([num_state_dims], tf.float32),
        action_spec=tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1),
        min_variance=min_variance,
        max_variance=max_variance)
    log_variances = network(states)[0].log_variance
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllGreater(log_variances, math.log(min_variance) - eps)
    self.assertAllLess(log_variances, math.log(max_variance) + eps)


if __name__ == '__main__':
  tf.test.main()
