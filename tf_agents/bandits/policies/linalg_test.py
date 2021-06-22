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

"""Tests for tf_agents.bandits.agents.linalg."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.policies import linalg

tfd = tfp.distributions
tf.compat.v1.enable_v2_behavior()


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10',
          'batch_size': 1,
          'context_dim': 10,
      }, {
          'testcase_name': '_batch4_contextdim5',
          'batch_size': 4,
          'context_dim': 5,
      })


class LinalgTest(tf.test.TestCase, parameterized.TestCase):

  @test_cases()
  def testAInvUpdate(self, batch_size, context_dim):
    a_array = 2 * np.eye(context_dim) + np.array(
        range(context_dim * context_dim)).reshape((context_dim, context_dim))
    a_array = a_array + a_array.T
    a_inv_array = np.linalg.inv(a_array)
    x_array = np.array(range(batch_size * context_dim)).reshape(
        (batch_size, context_dim))
    expected_a_inv_updated_array = np.linalg.inv(
        a_array + np.matmul(np.transpose(x_array), x_array))

    a_inv = tf.constant(
        a_inv_array, dtype=tf.float32, shape=[context_dim, context_dim])
    x = tf.constant(x_array, dtype=tf.float32, shape=[batch_size, context_dim])
    a_inv_update = linalg.update_inverse(a_inv, x)
    self.assertAllClose(expected_a_inv_updated_array,
                        self.evaluate(a_inv + a_inv_update))

  @test_cases()
  def testAInvUpdateEmptyObservations(self, batch_size, context_dim):
    a_array = 2 * np.eye(context_dim) + np.array(
        range(context_dim * context_dim)).reshape((context_dim, context_dim))
    a_array = a_array + a_array.T
    a_inv_array = np.linalg.inv(a_array)
    expected_a_inv_update_array = np.zeros([context_dim, context_dim],
                                           dtype=np.float32)

    a_inv = tf.constant(
        a_inv_array, dtype=tf.float32, shape=[context_dim, context_dim])
    x = tf.constant([], dtype=tf.float32, shape=[0, context_dim])
    a_inv_update = linalg.update_inverse(a_inv, x)
    self.assertAllClose(expected_a_inv_update_array,
                        self.evaluate(a_inv_update))


def cg_test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_n_1',
          'n': 1,
          'rhs': 1,
      }, {
          'testcase_name': '_n_10',
          'n': 10,
          'rhs': 1,
      }, {
          'testcase_name': '_n_100',
          'n': 100,
          'rhs': 5,
      })


class ConjugateGradientTest(tf.test.TestCase, parameterized.TestCase):

  @cg_test_cases()
  def testConjugateGradientBasic(self, n, rhs):
    x_obs = tf.constant(np.random.rand(n, 2), dtype=tf.float32, shape=[n, 2])
    a_mat = tf.eye(n) + tf.matmul(x_obs, tf.linalg.matrix_transpose(x_obs))
    x_exact = tf.constant(np.random.rand(n), dtype=tf.float32, shape=[n, 1])
    b = tf.matmul(a_mat, x_exact)
    x_approx = self.evaluate(linalg.conjugate_gradient(a_mat, b))
    x_exact_numpy = self.evaluate(x_exact)
    self.assertAllClose(x_exact_numpy, x_approx, rtol=1e-4, atol=1e-4)

  @cg_test_cases()
  def testConjugateGradientMultipleRHS(self, n, rhs):
    x_obs = tf.constant(np.random.rand(n, 2), dtype=tf.float32, shape=[n, 2])
    a_mat = tf.eye(n) + tf.matmul(x_obs, tf.linalg.matrix_transpose(x_obs))
    x_exact = tf.constant(
        np.random.rand(n, rhs), dtype=tf.float32, shape=[n, rhs])
    b_mat = tf.matmul(a_mat, x_exact)
    x_approx = self.evaluate(
        linalg.conjugate_gradient(a_mat, b_mat))
    x_exact_numpy = self.evaluate(x_exact)
    self.assertAllClose(x_exact_numpy, x_approx, rtol=1e-4, atol=1e-4)

  @cg_test_cases()
  def testConjugateGradientMultipleRHSPlaceholders(self, n, rhs):
    # Test the case where a_mat and b_mat are placeholders and they have unknown
    # dimension values.

    if tf.executing_eagerly():
      return

    x_obs = tf.constant(np.random.rand(n, 2), dtype=tf.float32, shape=[n, 2])
    a_mat = tf.eye(n) + tf.matmul(x_obs, tf.linalg.matrix_transpose(x_obs))
    a_mat_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, None))
    a_mat_value = self.evaluate(a_mat)

    x_exact = tf.constant(
        np.random.rand(n, rhs), dtype=tf.float32, shape=[n, rhs])
    b_mat = tf.matmul(a_mat, x_exact)
    b_mat_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, None))
    b_mat_value = self.evaluate(b_mat)

    x_exact_numpy = self.evaluate(x_exact)
    with self.cached_session() as sess:
      x_approx = linalg.conjugate_gradient(a_mat_ph, b_mat_ph)
      x_approx_value = sess.run(
          x_approx,
          feed_dict={a_mat_ph: a_mat_value, b_mat_ph: b_mat_value})
      self.assertAllClose(x_exact_numpy, x_approx_value, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
