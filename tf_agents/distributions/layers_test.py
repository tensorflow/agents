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

"""Tests for tf_agents.distributions.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import layers
from tf_agents.specs import tensor_spec


def _get_inputs(batch_size, num_input_dims):
  return tf.random_uniform([batch_size, num_input_dims])


class FactoredCategoricalTest(tf.test.TestCase):

  def testShape(self):
    inputs = _get_inputs(batch_size=3, num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [11, 2], tf.int32, 0, 4)

    distribution = layers.factored_categorical(inputs, output_spec)
    self.assertEqual(type(distribution), tfp.distributions.Categorical)
    logits = distribution.logits

    self.assertAllEqual(logits.shape.as_list(),
                        [3] + output_spec.shape.as_list() + [5])


class NormalTest(tf.test.TestCase):

  def testShape(self):
    inputs = _get_inputs(batch_size=3, num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [11, 2], tf.float32, 0, 1)

    distribution = layers.normal(inputs, output_spec)
    self.assertEqual(type(distribution), tfp.distributions.Normal)
    means, stds = distribution.loc, distribution.scale

    self.assertAllEqual(means.shape.as_list(),
                        [3] + output_spec.shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3] + output_spec.shape.as_list())

  def testSquash(self):
    inputs = _get_inputs(batch_size=3, num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [11, 2], tf.float32, 0, 1)

    def gen_projection(value=1):
      def projection(inputs, num_elements, scope=None):
        del scope
        return value * tf.ones([tf.shape(inputs)[0], num_elements])
      return projection

    with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
      # No squashing.
      distribution = layers.normal(
          inputs, output_spec,
          mean_transform=layers.passthrough,
          projection_layer=gen_projection(2))

      self.assertEqual(type(distribution), tfp.distributions.Normal)
      unsquashed_means1 = distribution.loc

      distribution = layers.normal(
          inputs, output_spec,
          mean_transform=layers.passthrough,
          projection_layer=gen_projection(-1))

      self.assertEqual(type(distribution), tfp.distributions.Normal)
      unsquashed_means2 = distribution.loc

      # Yes squashing.
      distribution = layers.normal(
          inputs, output_spec,
          projection_layer=gen_projection(2))

      self.assertEqual(type(distribution), tfp.distributions.Normal)
      squashed_means1 = distribution.loc

      distribution = layers.normal(
          inputs, output_spec,
          projection_layer=gen_projection(-1))

      self.assertEqual(type(distribution), tfp.distributions.Normal)
      squashed_means2 = distribution.loc

    self.evaluate(tf.global_variables_initializer())
    (unsquashed_means1_, unsquashed_means2_,
     squashed_means1_, squashed_means2_) = self.evaluate(
         (unsquashed_means1, unsquashed_means2,
          squashed_means1, squashed_means2))
    self.assertTrue(np.all(unsquashed_means1_ == 2))
    self.assertTrue(np.all(unsquashed_means2_ == -1))
    self.assertTrue(np.all(squashed_means1_ <= 1))
    self.assertTrue(np.all(squashed_means2_ >= 0))

if __name__ == '__main__':
  tf.test.main()
