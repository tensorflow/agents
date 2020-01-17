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

"""Tests for tf_agents.utils.tensor_normalizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import tensor_spec
from tf_agents.utils import tensor_normalizer


class EMATensorNormalizerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(EMATensorNormalizerTest, self).setUp()
    tf.compat.v1.reset_default_graph()
    self._tensor_spec = tensor_spec.TensorSpec([3], tf.float32, 'obs')
    self._tensor_normalizer = tensor_normalizer.EMATensorNormalizer(
        tensor_spec=self._tensor_spec)
    self._dict_tensor_spec = {'a': self._tensor_spec, 'b': self._tensor_spec}
    self._dict_tensor_normalizer = tensor_normalizer.EMATensorNormalizer(
        tensor_spec=self._dict_tensor_spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())

  def testGetVariables(self):
    means_var, variances_var = self._tensor_normalizer.variables
    self.assertAllEqual(means_var.shape.as_list(),
                        self._tensor_spec.shape.as_list())
    self.assertAllEqual(variances_var.shape.as_list(),
                        self._tensor_spec.shape.as_list())

  def testUpdateVariables(self):
    # Get original mean and variance.
    original_means, original_variances = self.evaluate(
        self._tensor_normalizer.variables)

    # Construct and evaluate normalized tensor. Should update mean &
    #   variance.
    tensor = tf.constant([[1.3, 4.2, 7.5]], dtype=tf.float32)
    update_norm_vars = self._tensor_normalizer.update(tensor)
    self.evaluate(update_norm_vars)

    # Get new mean and variance, and make sure they changed.
    new_means, new_variances = self.evaluate(
        self._tensor_normalizer.variables)
    for new_val, old_val in (list(zip(new_means, original_means)) +
                             list(zip(new_variances, original_variances))):
      self.assertNotEqual(new_val, old_val)

  def testUpdateVariablesDictNest(self):
    # Get original mean and variance.
    original_means, original_variances = self.evaluate(
        self._dict_tensor_normalizer.variables)

    # Construct and evaluate normalized tensor. Should update mean &
    #   variance.
    tensor = {'a': tf.constant([[1.3, 4.2, 7.5]], dtype=tf.float32),
              'b': tf.constant([[1.3, 4.2, 7.5]], dtype=tf.float32)}
    update_norm_vars = self._dict_tensor_normalizer.update(tensor)
    self.evaluate(update_norm_vars)

    # Get new mean and variance, and make sure they changed.
    new_means, new_variances = self.evaluate(
        self._dict_tensor_normalizer.variables)

    def _assert_dict_changed(dict1, dict2):
      self.assertAllEqual(sorted(dict1.keys()), sorted(dict2.keys()))
      for k in dict1.keys():
        for i in range(len(dict1[k])):
          self.assertNotEqual(dict1[k][i], dict2[k][i])

    _assert_dict_changed(original_means, new_means)
    _assert_dict_changed(original_variances, new_variances)

  @parameterized.named_parameters(
      ('OneReduceAxis', 1),
      ('TwoReduceAxes', 2),
  )
  def testNormalization(self, num_outer_dims):
    means_var, variance_var = self._tensor_normalizer.variables
    self.evaluate([
        tf.compat.v1.assign(means_var, [10.0] * 3),
        tf.compat.v1.assign(variance_var, [0.1] * 3)
    ])

    vector = [9.0, 10.0, 11.0]
    # Above, the estimated mean was set to 10, and variance to 0.1. Thus the
    # estimated stddev is sqrt(0.1) = 0.3162.
    # The middle sample falls on the mean, so should be normalized to 0.0. Each
    # of the other samples is 1 away from the mean. 1 / 0.3162 = 3.162
    expected = [-3.1622776601, 0.0, 3.1622776601]
    for _ in range(num_outer_dims - 1):
      vector = [vector] * 2
      expected = [expected] * 2
    tensor = tf.constant(vector)

    norm_obs = self._tensor_normalizer.normalize(
        tensor, variance_epsilon=0.0)
    self.assertAllClose(expected, self.evaluate(norm_obs), atol=0.0001)

  def testNormalizationDictNest(self):
    means_var, variance_var = self._dict_tensor_normalizer.variables
    self.evaluate(  # For each var in nest, assign initial value.
        [tf.compat.v1.assign(var, [10.0] * 3) for var in means_var.values()] +
        [tf.compat.v1.assign(var, [.1] * 3) for var in variance_var.values()])

    vector = [9.0, 10.0, 11.0]
    expected = {'a': [-3.1622776601, 0.0, 3.1622776601],
                'b': [-3.1622776601, 0.0, 3.1622776601]}
    tensor = {'a': tf.constant(vector), 'b': tf.constant(vector)}

    norm_obs = self._dict_tensor_normalizer.normalize(
        tensor, variance_epsilon=0.0)
    self.assertAllClose(expected, self.evaluate(norm_obs), atol=0.0001)

  def testShouldNotCenterMean(self):
    means_var, variance_var = self._tensor_normalizer.variables
    self.evaluate([
        tf.compat.v1.assign(means_var, [10.0] * 3),
        tf.compat.v1.assign(variance_var, [0.01] * 3)
    ])
    tensor = tf.constant([[9.0, 10.0, 11.0]])
    norm_obs = self._tensor_normalizer.normalize(
        tensor, center_mean=False,
        variance_epsilon=0.0, clip_value=0.0)
    expected = [[90.0, 100.0, 110.0]]
    self.assertAllClose(expected, self.evaluate(norm_obs))


class StreamingTensorNormalizerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(StreamingTensorNormalizerTest, self).setUp()
    tf.compat.v1.reset_default_graph()
    self._tensor_spec = tensor_spec.TensorSpec([3], tf.float32, 'obs')
    self._tensor_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=self._tensor_spec)
    self._dict_tensor_spec = {'a': self._tensor_spec, 'b': self._tensor_spec}
    self._dict_tensor_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=self._dict_tensor_spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())

  def testGetVariables(self):
    count_var, means_var, variances_var = (self._tensor_normalizer.variables)
    self.assertAllEqual(count_var.shape, self._tensor_spec.shape)
    self.assertAllEqual(means_var.shape, self._tensor_spec.shape)
    self.assertAllEqual(variances_var.shape, self._tensor_spec.shape)

  def testUpdateVariables(self):
    # Get original mean and variance.
    original_count, original_mean_sum, original_variance_sum = self.evaluate(
        self._tensor_normalizer.variables)

    # Construct and evaluate normalized tensor. Should update mean &
    #   variance.
    np_array = np.array([[1.3, 4.2, 7.5],
                         [8.3, 2.2, 9.5],
                         [3.3, 5.2, 6.5]], np.float32)
    tensor = tf.constant(np_array, dtype=tf.float32)
    update_norm_vars = self._tensor_normalizer.update(tensor)
    self.evaluate(update_norm_vars)

    # Get new mean and variance, and make sure they changed.
    new_count, new_mean_sum, new_variance_sum = self.evaluate(
        self._tensor_normalizer.variables)

    self.assertAllEqual(new_count,
                        np.array([3, 3, 3], dtype=np.float32) + original_count)
    self.assertAllClose(new_mean_sum,
                        np.sum(np_array, axis=0) + original_mean_sum)
    self.assertAllClose(
        new_variance_sum,
        np.sum(np.square(np_array - original_mean_sum), axis=0) +
        original_variance_sum)

  def testUpdateVariablesDictNest(self):
    # Get original mean and variance.
    original_count, original_mean_sum, original_variance_sum = self.evaluate(
        self._dict_tensor_normalizer.variables)

    # Construct and evaluate normalized tensor. Should update mean &
    #   variance.
    np_array = np.array([[1.3, 4.2, 7.5],
                         [8.3, 2.2, 9.5],
                         [3.3, 5.2, 6.5]], np.float32)
    tensor = {'a': tf.constant(np_array, dtype=tf.float32),
              'b': tf.constant(np_array, dtype=tf.float32)}
    update_norm_vars = self._dict_tensor_normalizer.update(tensor)
    self.evaluate(update_norm_vars)

    # Get new mean and variance, and make sure they changed.
    new_count, new_mean_sum, new_variance_sum = self.evaluate(
        self._dict_tensor_normalizer.variables)

    expected_count = {k: (np.array([3, 3, 3], dtype=np.float32) +
                          original_count[k]) for k in original_count}
    expected_mean_sum = {k: (np.sum(np_array, axis=0) +
                             original_mean_sum[k]) for k in original_mean_sum}
    expected_variance_sum = {
        k: (np.sum(np.square(np_array - original_mean_sum[k]), axis=0) +
            original_variance_sum[k]) for k in original_variance_sum}

    def _assert_dicts_close(dict1, dict2):
      self.assertAllEqual(sorted(dict1.keys()), sorted(dict2.keys()))
      self.assertAllClose([dict1[k] for k in dict1.keys()],
                          [dict2[k] for k in dict1.keys()])

    _assert_dicts_close(new_count, expected_count)
    _assert_dicts_close(new_mean_sum, expected_mean_sum)
    _assert_dicts_close(new_variance_sum, expected_variance_sum)

  @parameterized.named_parameters(
      ('OneReduceAxis', 1),
      ('TwoReduceAxes', 2),
  )
  def testNormalization(self, num_outer_dims):
    count_var, means_var, variance_var = self._tensor_normalizer.variables
    self.evaluate([
        tf.compat.v1.assign(count_var, [1.0] * 3),
        tf.compat.v1.assign(means_var, [10.0] * 3),
        tf.compat.v1.assign(variance_var, [0.1] * 3)
    ])

    vector = [9.0, 10.0, 11.0]
    # Above, the estimated mean was set to 10, and variance to 0.1. Thus the
    # estimated stddev is sqrt(0.1) = 0.3162.
    # The middle sample falls on the mean, so should be normalized to 0.0. Each
    # of the other samples is 1 away from the mean. 1 / 0.3162 = 3.162
    expected = [-3.1622776601, 0.0, 3.1622776601]
    for _ in range(num_outer_dims - 1):
      vector = [vector] * 2
      expected = [expected] * 2
    tensor = tf.constant(vector)

    norm_obs = self._tensor_normalizer.normalize(
        tensor, variance_epsilon=0.0)
    self.assertAllClose(expected, self.evaluate(norm_obs), atol=0.0001)

  def testNormalizationDictNest(self):
    count_var, means_var, variance_var = self._dict_tensor_normalizer.variables
    self.evaluate(  # For each var in nest, assign initial value.
        [tf.compat.v1.assign(var, [1.0] * 3) for var in count_var.values()] +
        [tf.compat.v1.assign(var, [10.0] * 3) for var in means_var.values()] +
        [tf.compat.v1.assign(var, [.1] * 3) for var in variance_var.values()])

    vector = [9.0, 10.0, 11.0]
    expected = {'a': [-3.1622776601, 0.0, 3.1622776601],
                'b': [-3.1622776601, 0.0, 3.1622776601]}
    tensor = {'a': tf.constant(vector), 'b': tf.constant(vector)}

    norm_obs = self._dict_tensor_normalizer.normalize(
        tensor, variance_epsilon=0.0)
    self.assertAllClose(expected, self.evaluate(norm_obs), atol=0.0001)

  def testShouldNotCenterMean(self):
    count_var, means_var, variance_var = self._tensor_normalizer.variables
    self.evaluate([
        tf.compat.v1.assign(count_var, [1.0] * 3),
        tf.compat.v1.assign(means_var, [10.0] * 3),
        tf.compat.v1.assign(variance_var, [0.01] * 3)
    ])
    tensor = tf.constant([[9.0, 10.0, 11.0]])
    norm_obs = self._tensor_normalizer.normalize(
        tensor, center_mean=False,
        variance_epsilon=0.0, clip_value=0.0)
    expected = [[90.0, 100.0, 110.0]]
    self.assertAllClose(expected, self.evaluate(norm_obs))

if __name__ == '__main__':
  tf.test.main()
