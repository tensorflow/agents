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

"""Tests for tf_agents.utils.tensor_normalizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.utils import tensor_normalizer


class ParallelVarianceTest(tf.test.TestCase):

  def testParallelVarianceOneAtATime(self):
    x = np.random.randn(5, 10)
    n, avg, m2, m2_c = 1, x[0], 0, 0
    for row in range(1, 5):
      n, avg, m2, m2_c = tensor_normalizer.parallel_variance_calculation(
          n_a=n, avg_a=avg, m2_a=m2,
          n_b=1, avg_b=x[row], m2_b=0, m2_b_c=m2_c)
    var = m2 / n
    self.assertAllClose(avg, x.mean(axis=0))
    self.assertAllClose(var, x.var(axis=0))

  def testParallelVarianceForOneGroup(self):
    x = tf.constant(np.random.randn(5, 10))
    n = 5
    avg, var = tf.nn.moments(x, axes=[0])
    m2 = var * n
    new_n, new_avg, new_m2, _ = tensor_normalizer.parallel_variance_calculation(
        n, avg, m2, n_b=0, avg_b=0, m2_b=0, m2_b_c=0)
    new_var = new_m2 / n
    (avg, var, new_avg, new_var) = self.evaluate(
        (avg, var, new_avg, new_var))
    self.assertEqual(new_n, 5)
    self.assertAllClose(new_avg, avg)
    self.assertAllClose(new_var, var)

  def testParallelVarianceCombinesGroups(self):
    x1 = tf.constant(np.random.randn(5, 10))
    x2 = tf.constant(np.random.randn(15, 10))
    n1 = 5
    n2 = 15
    avg1 = tf.math.reduce_mean(x1, axis=[0])
    m2_1 = tf.math.reduce_sum(tf.math.squared_difference(x1, avg1), axis=[0])
    avg2 = tf.math.reduce_mean(x2, axis=[0])
    m2_2 = tf.math.reduce_sum(tf.math.squared_difference(x2, avg2), axis=[0])
    m2_c = m2_2 * 0.
    n, avg, m2, _ = tensor_normalizer.parallel_variance_calculation(
        n1, avg1, m2_1,
        n2, avg2, m2_2, m2_c)
    var = m2 / n
    avg_true, var_true = tf.nn.moments(tf.concat((x1, x2), axis=0), axes=[0])
    avg, var, avg_true, var_true = self.evaluate((
        avg, var, avg_true, var_true))
    self.assertAllClose(avg, avg_true)
    self.assertAllClose(var, var_true)


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

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testGetVariables(self, dtype):
    spec = tensor_spec.TensorSpec([3], dtype, 'obs')
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    variables = t_normalizer.variables
    for var in variables:
      self.assertEqual(var.shape, spec.shape)
      self.assertEqual(var.dtype, spec.dtype)

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testReset(self, dtype):
    # Get original mean and variance.
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    original_vars = self.evaluate(t_normalizer.variables)

    # Construct and evaluate normalized tensor. Updates statistics.
    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    update_norm_vars = t_normalizer.update(tensor)
    self.evaluate(update_norm_vars)

    # Verify that the internal variables have been successfully reset.
    self.evaluate(t_normalizer.reset())
    reset_vars = self.evaluate(t_normalizer.variables)
    self.assertAllClose(original_vars, reset_vars)

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testUpdate(self, dtype):
    # Construct and evaluate normalized tensor. Should update mean &
    #   variance.
    spec = tensor_spec.TensorSpec([3], dtype, 'obs')
    dict_tensor_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec={
            'a': spec,
            'b': spec
        })
    self.evaluate(tf.compat.v1.global_variables_initializer())

    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensors = {
        'a': tf.constant(np_array, dtype=dtype),
        'b': tf.constant(np_array, dtype=dtype)
    }

    def _compare(data):
      n = data.shape[0]
      expected_avg = data.mean(axis=0)
      expected_var = data.var(axis=0)
      expected_m2 = expected_var * n
      expected_count = np.array([n] * 3)

      new_count, new_avg, new_m2, _ = self.evaluate(
          dict_tensor_normalizer.variables)

      tf.nest.map_structure(lambda v: self.assertAllClose(v, expected_count),
                            new_count)
      tf.nest.map_structure(lambda v: self.assertAllClose(v, expected_avg),
                            new_avg)
      tf.nest.map_structure(lambda v: self.assertAllClose(v, expected_m2),
                            new_m2)

    update_norm_vars = dict_tensor_normalizer.update(tensors)
    self.evaluate(update_norm_vars)
    _compare(data=np_array)

    update_norm_vars = dict_tensor_normalizer.update(
        tf.nest.map_structure(lambda t: t + 1.0, tensors))
    self.evaluate(update_norm_vars)
    _compare(data=np.concatenate((np_array, np_array + 1.0), axis=0))

    update_norm_vars = dict_tensor_normalizer.update(
        tf.nest.map_structure(lambda t: t - 1.0, tensors))
    self.evaluate(update_norm_vars)
    _compare(
        data=np.concatenate((np_array, np_array + 1.0, np_array - 1.0), axis=0))

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testNormalization(self, dtype):
    spec = tensor_spec.TensorSpec([3], dtype, 'obs')
    dict_tensor_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec={
            'a': spec,
            'b': spec
        })
    self.evaluate(tf.compat.v1.global_variables_initializer())
    as_tensor = functools.partial(tf.convert_to_tensor, dtype=dtype)

    # Update with some initial values.
    norm_obs = {'a': np.random.randn(6, 2, 3),
                'b': np.random.randn(6, 2, 3)}
    norm_obs_t = tf.nest.map_structure(as_tensor, norm_obs)
    self.evaluate(dict_tensor_normalizer.update(norm_obs_t))

    view_obs = {'a': np.random.randn(4, 3),
                'b': np.random.randn(4, 3)}
    view_obs_t = tf.nest.map_structure(as_tensor, view_obs)
    observed = self.evaluate(
        dict_tensor_normalizer.normalize(
            view_obs_t, clip_value=-1, variance_epsilon=1e-6))

    norm_obs_avg = tf.nest.map_structure(
        lambda a: a.mean(axis=(0, 1)), norm_obs)
    norm_obs_std = tf.nest.map_structure(
        lambda a: a.std(axis=(0, 1)), norm_obs)
    expected = tf.nest.map_structure(
        lambda obs, avg, std: (obs - avg)/std,
        view_obs, norm_obs_avg, norm_obs_std)

    self.assertAllClose(observed, expected)

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testNormalizeVSNumpy(self, dtype):
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    self.evaluate(t_normalizer.update(tensor))

    epsilon = 1e-6
    # Get new mean and variance, and make sure they changed.
    norm_obs = t_normalizer.normalize(tensor, variance_epsilon=epsilon)

    exp_obs = ((np_array - np_array.mean(axis=0)) /
               (np_array.std(axis=0) + epsilon))

    self.assertAllClose(norm_obs, exp_obs)

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testMeanVariance(self, dtype):
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    self.evaluate(t_normalizer.update(tensor))

    # Get new mean and variance, and make sure they changed.
    new_mean, new_variance = self.evaluate(
        t_normalizer._get_mean_var_estimates())

    self.assertAllClose(np_array.mean(axis=0), new_mean[0])

    self.assertAllClose(np_array.var(axis=0), new_variance[0])

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testIncrementalMean(self, dtype):
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    # It fails when run more than 62 iterations.
    for i in range(62):
      self.evaluate(t_normalizer.update(tensor + 100 * i))

      # Get new mean and variance, and make sure they changed.
      new_mean, _ = self.evaluate(t_normalizer._get_mean_var_estimates())
      new_array = np.concatenate([np_array + 100*j for j in range(i+1)], axis=0)

      self.assertAllClose(new_array.mean(axis=0), new_mean[0])

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testFixedMean(self, dtype):
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    # It fails when run more than 41 iterations.
    for i in range(41):
      self.evaluate(t_normalizer.update(tensor))

      # Get new mean and variance, and make sure they changed.
      new_mean, _ = self.evaluate(t_normalizer._get_mean_var_estimates())
      new_array = np.concatenate([np_array for _ in range(i+1)], axis=0)

      self.assertAllClose(new_array.mean(axis=0), new_mean[0])

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testIncrementalVariance(self, dtype):
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np_array = np.array([[1.3, 4.2, 7.5], [8.3, 2.2, 9.5], [3.3, 5.2, 6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    # It fails when run more than 383 iterations.
    for i in range(383):
      self.evaluate(t_normalizer.update(tensor + 100 * i))

      # Get new mean and variance, and make sure they changed.
      _, new_variance = self.evaluate(t_normalizer._get_mean_var_estimates())
      new_array = np.concatenate([np_array + 100*j for j in range(i+1)], axis=0)

      self.assertAllClose(new_array.var(axis=0), new_variance[0])

  @parameterized.named_parameters(
      ('float32', tf.float32),
      ('float64', tf.float64),
  )
  def testFixedVariance(self, dtype):
    t_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec=tensor_spec.TensorSpec([3], dtype, 'obs'))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    np_array = np.array([[-1.3, 4.2, 7.5], [8.3, -2.2, 9.5], [3.3, 5.2, -6.5]],
                        dtype.as_numpy_dtype)
    tensor = tf.constant(np_array, dtype=dtype)
    # It fails done more than 54 iterations.
    for i in range(54):
      self.evaluate(t_normalizer.update(tensor))

      # Get new mean and variance, and make sure they changed.
      _, new_variance = self.evaluate(t_normalizer._get_mean_var_estimates())
      new_array = np.concatenate([np_array for _ in range(i+1)], axis=0)

      self.assertAllClose(new_array.var(axis=0), new_variance[0])


if __name__ == '__main__':
  tf.test.main()
