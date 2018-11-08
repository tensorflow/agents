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

"""Tests for tf_agents.distributions.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import layers
from tf_agents.distributions import utils
from tf_agents.specs import tensor_spec


def _get_inputs(batch_dims, num_input_dims):
  return tf.random_uniform(batch_dims + [num_input_dims])


class ProjectToOutputDistributionTest(tf.test.TestCase):

  def testDiscrete(self):
    inputs = _get_inputs(batch_dims=[3], num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [11], tf.int32, 0, 1)

    distribution = utils.project_to_output_distribution(
        inputs,
        output_spec,
        project_to_discrete=layers.factored_categorical,
        project_to_continuous=layers.normal)

    self.assertEqual(type(distribution), tfp.distributions.Categorical)
    logits = distribution.logits
    self.assertAllEqual(logits.shape.as_list(),
                        [3] + output_spec.shape.as_list() + [2])

  def testDiscreteTwoBatchDims(self):
    inputs = _get_inputs(batch_dims=[3, 13], num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [11], tf.int32, 0, 1)

    distribution = utils.project_to_output_distribution(
        inputs,
        output_spec,
        outer_rank=2,
        project_to_discrete=layers.factored_categorical,
        project_to_continuous=layers.normal)

    self.assertEqual(type(distribution), tfp.distributions.Categorical)
    logits = distribution.logits
    self.assertAllEqual(logits.shape.as_list(),
                        [3, 13] + output_spec.shape.as_list() + [2])

  def test2DDiscrete(self):
    inputs = _get_inputs(batch_dims=[3], num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [11, 4], tf.int32, -2, 2)

    distribution = utils.project_to_output_distribution(
        inputs,
        output_spec,
        project_to_discrete=layers.factored_categorical,
        project_to_continuous=layers.normal)

    self.assertEqual(type(distribution), tfp.distributions.Categorical)
    logits = distribution.logits

    self.assertAllEqual(logits.shape.as_list(),
                        [3] + output_spec.shape.as_list() + [5])

  def testContinuousTwoBatchDims(self):
    inputs = _get_inputs(batch_dims=[3, 13], num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [1], tf.float32, 2., 3.)

    distribution = utils.project_to_output_distribution(
        inputs,
        output_spec,
        outer_rank=2,
        project_to_discrete=layers.factored_categorical,
        project_to_continuous=layers.normal)

    self.assertEqual(type(distribution), tfp.distributions.Normal)
    means, stds = distribution.loc, distribution.scale

    self.assertAllEqual(means.shape.as_list(),
                        [3, 13] + output_spec.shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3, 13] + output_spec.shape.as_list())

  def test2DContinuous(self):
    inputs = _get_inputs(batch_dims=[3], num_input_dims=5)
    output_spec = tensor_spec.BoundedTensorSpec(
        [7, 5], tf.float32, 2., 3.)

    distribution = utils.project_to_output_distribution(
        inputs,
        output_spec,
        project_to_discrete=layers.factored_categorical,
        project_to_continuous=layers.normal)

    self.assertEqual(type(distribution), tfp.distributions.Normal)
    means, stds = distribution.loc, distribution.scale

    self.assertAllEqual(means.shape.as_list(),
                        [3] + output_spec.shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3] + output_spec.shape.as_list())


class ProjectToOutputDistributionsTest(tf.test.TestCase):

  def testListOfSingleoutput(self):
    inputs = _get_inputs(batch_dims=[3], num_input_dims=5)
    output_spec = [tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)]

    distributions = utils.project_to_output_distributions(inputs, output_spec)

    self.assertEqual(len(distributions), 1)
    self.assertEqual(type(distributions[0]), tfp.distributions.Normal)
    means, stds = distributions[0].loc, distributions[0].scale

    self.assertAllEqual(means.shape.as_list(),
                        [3] + output_spec[0].shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3] + output_spec[0].shape.as_list())

  def testDictOfSingleoutput(self):
    inputs = _get_inputs(batch_dims=[3], num_input_dims=5)
    output_spec = {
        'motor': tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.)
    }

    distributions = utils.project_to_output_distributions(inputs, output_spec)

    self.assertEqual(len(distributions), 1)
    self.assertEqual(type(distributions['motor']), tfp.distributions.Normal)
    means, stds = distributions['motor'].loc, distributions['motor'].scale

    self.assertAllEqual(means.shape.as_list(),
                        [3] + output_spec['motor'].shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3] + output_spec['motor'].shape.as_list())

  def testMultipleNestedoutputs(self):
    inputs = _get_inputs(batch_dims=[3], num_input_dims=5)
    output_spec = [
        tensor_spec.BoundedTensorSpec([1], tf.float32, 2., 3.),
        [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)]]

    distributions = utils.project_to_output_distributions(inputs, output_spec)

    # Check nest structure.
    self.assertEqual(len(distributions), 2)
    self.assertEqual(len(distributions[1]), 1)

    # Check distributions.
    self.assertEqual(type(distributions[0]), tfp.distributions.Normal)
    self.assertEqual(type(distributions[1][0]), tfp.distributions.Categorical)
    means, stds = distributions[0].loc, distributions[0].scale
    logits = distributions[1][0].logits

    self.assertAllEqual(means.shape.as_list(),
                        [3] + output_spec[0].shape.as_list())
    self.assertAllEqual(stds.shape.as_list(),
                        [3] + output_spec[0].shape.as_list())
    self.assertAllEqual(logits.shape.as_list(),
                        [3] + output_spec[1][0].shape.as_list() + [2])


if __name__ == '__main__':
  tf.test.main()
