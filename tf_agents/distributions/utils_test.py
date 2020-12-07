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

"""Tests for tf_agents.distributions.utils."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import utils
from tf_agents.specs import tensor_spec


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def testScaleDistribution(self):
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -2, 4)
    distribution = tfp.distributions.Normal(0, 4)
    scaled_distribution = utils.scale_distribution_to_spec(distribution,
                                                           action_spec)
    if tf.executing_eagerly():
      sample = scaled_distribution.sample
    else:
      sample = scaled_distribution.sample()

    for _ in range(1000):
      sample_np = self.evaluate(sample)

      self.assertGreater(sample_np, -2.00001)
      self.assertLess(sample_np, 4.00001)

  def testSquashToSpecNormalModeMethod(self):
    input_dist = tfp.distributions.Normal(loc=1.0, scale=3.0)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -2.0, 4.0)
    squash_to_spec_normal = utils.SquashToSpecNormal(input_dist, action_spec)
    self.assertAlmostEqual(
        self.evaluate(squash_to_spec_normal.mode()), 3.28478247, places=5)

  def testSquashToSpecNormalStdMethod(self):
    input_dist = tfp.distributions.Normal(loc=1.0, scale=3.0)
    action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -2.0, 4.0)
    squash_to_spec_normal = utils.SquashToSpecNormal(input_dist, action_spec)
    self.assertAlmostEqual(
        self.evaluate(squash_to_spec_normal.stddev()), 2.98516426, places=5)

  def compare_params(self, p1, p2, skip_tensor_values=False):
    if tf.is_tensor(p1):
      # When comparing tensors, just make sure both items are tensor-like.  This
      # allows us to consider, e.g., tf.Variable and tf.Tensor equivalent.
      self.assertTrue(tf.is_tensor(p2))
    else:
      self.assertFalse(tf.is_tensor(p2))
      self.assertEqual(type(p1), type(p2))
    if isinstance(p1, utils.Params):
      self.assertEqual(p1.type_, p2.type_)
      tf.nest.map_structure(
          lambda p1_, p2_: self.compare_params(p1_, p2_, skip_tensor_values),
          p1.params, p2.params)
    elif tf.is_tensor(p1):
      if not skip_tensor_values:
        self.assertAllEqual(p1, p2)
    else:
      self.assertEqual(p1, p2)

  def testGetAndMakeFromParameters(self):
    one = tf.constant(1.0)
    d = tfp.distributions.Normal(
        loc=one, scale=3.0, validate_args=True)
    d = tfp.bijectors.Tanh()(d)
    d = tfp.bijectors.Tanh()(d)
    p = utils.get_parameters(d)

    expected_p = utils.Params(
        tfp.distributions.TransformedDistribution,
        params={
            'bijector':
                utils.Params(
                    tfp.bijectors.Chain,
                    params={'bijectors': [
                        utils.Params(tfp.bijectors.Tanh, params={}),
                        utils.Params(tfp.bijectors.Tanh, params={}),
                    ]}),
            'distribution':
                utils.Params(
                    tfp.distributions.Normal,
                    params={'validate_args': True, 'scale': 3.0, 'loc': one})})

    self.compare_params(p, expected_p)

    d_recreated = utils.make_from_parameters(p)
    points = [0.01, 0.25, 0.5, 0.75, 0.99]
    self.assertAllClose(d.log_prob(points), d_recreated.log_prob(points))

  def testGetAndMakeNontrivialBijectorFromParameters(self):
    scale_matrix = tf.Variable([[1.0, 2.0], [-1.0, 0.0]])
    d = tfp.distributions.MultivariateNormalDiag(
        loc=[1.0, 1.0], scale_diag=[2.0, 3.0], validate_args=True)
    b = tfp.bijectors.ScaleMatvecLinearOperator(
        scale=tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix),
        adjoint=True)
    b_d = b(d)
    p = utils.get_parameters(b_d)

    expected_p = utils.Params(
        tfp.distributions.TransformedDistribution,
        params={
            'bijector': utils.Params(
                tfp.bijectors.ScaleMatvecLinearOperator,
                params={'adjoint': True,
                        'scale': utils.Params(
                            tf.linalg.LinearOperatorFullMatrix,
                            params={'matrix': scale_matrix})}),
            'distribution': utils.Params(
                tfp.distributions.MultivariateNormalDiag,
                params={'validate_args': True,
                        'scale_diag': [2.0, 3.0],
                        'loc': [1.0, 1.0]})})

    self.compare_params(p, expected_p)

    b_d_recreated = utils.make_from_parameters(p)

    points = [[-1.0, -2.0],
              [0.0, 0.0],
              [3.0, -5.0],
              [5.0, 5.0],
              [1.0, np.inf],
              [-np.inf, 0.0]]
    self.assertAllClose(
        b_d.log_prob(points), b_d_recreated.log_prob(points))

  @parameterized.named_parameters(('ConvertAll', False),
                                  ('ConvertOnlyNonconstantTensors', True))
  def testParametersToAndFromDict(self, tensors_only):
    scale_matrix = tf.Variable([[1.0, 2.0], [-1.0, 0.0]])
    d = tfp.distributions.MultivariateNormalDiag(
        loc=[1.0, 1.0], scale_diag=[2.0, 3.0], validate_args=True)
    b = tfp.bijectors.ScaleMatvecLinearOperator(
        scale=tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix),
        adjoint=True)
    b_d = b(d)
    p = utils.get_parameters(b_d)

    p_dict = utils.parameters_to_dict(
        p, tensors_only=tensors_only)

    if tensors_only:
      expected_p_dict = {
          'bijector': {'scale': {'matrix': scale_matrix}},
          'distribution': {},
      }
    else:
      expected_p_dict = {
          'bijector': {'adjoint': True,
                       'scale': {'matrix': scale_matrix}},
          'distribution': {'validate_args': True,
                           # These are deeply nested because we passed lists
                           # intead of numpy arrays for `loc` and `scale_diag`.
                           'scale_diag:0': 2.0,
                           'scale_diag:1': 3.0,
                           'loc:0': 1.0,
                           'loc:1': 1.0}}

    tf.nest.map_structure(
        self.assertAllEqual, p_dict, expected_p_dict)

    # This converts the tf.Variable entry in the matrix to a tf.Tensor
    p_dict['bijector']['scale']['matrix'] = (
        p_dict['bijector']['scale']['matrix'] + 1.0)

    # When tensors_only=True, we make sure that we can merge into p
    # from a dict where we dropped everything but tensors.
    p_recreated = utils.merge_to_parameters_from_dict(p, p_dict)

    self.assertAllClose(
        p_recreated.params['bijector'].params['scale'].params['matrix'],
        p.params['bijector'].params['scale'].params['matrix'] + 1.0)

    # Skip the tensor value comparison -- we checked it above.
    self.compare_params(p, p_recreated, skip_tensor_values=True)

  def testParametersFromDictMissingNestedParamsKeyFailure(self):
    one = tf.constant(1.0)
    d = tfp.distributions.Normal(
        loc=one, scale=3.0, validate_args=True)
    d = tfp.bijectors.Tanh()(d)
    d = tfp.bijectors.Tanh()(d)
    p = utils.get_parameters(d)
    p_dict = utils.parameters_to_dict(p)

    # Add a third bijector, changing the structure of the nest.
    self.assertIn('bijectors:0', p_dict['bijector'].keys())
    self.assertIn('bijectors:1', p_dict['bijector'].keys())
    p_dict['bijector']['bijectors:2'] = p_dict['bijector']['bijectors:0']

    # Flattening nested params lost information about the nested structure, so
    # we can't e.g. add a new bijector in the dict and expect to put that back
    # into the bijector list when converting back.
    with self.assertRaisesRegex(
        ValueError,
        r'params_dict had keys that were not part of value.params.*'
        r'params_dict keys: \[\'bijectors:0\', \'bijectors:1\', '
        r'\'bijectors:2\'\], value.params processed keys: \[\'bijectors:0\', '
        r'\'bijectors:1\'\]'):
      utils.merge_to_parameters_from_dict(p, p_dict)

  def testParametersFromDictMissingNestedDictKeyFailure(self):
    one = tf.constant(1.0)
    d = tfp.distributions.Normal(
        loc=one, scale=3.0, validate_args=True)
    d = tfp.bijectors.Tanh()(d)
    d = tfp.bijectors.Tanh()(d)
    p = utils.get_parameters(d)
    p_dict = utils.parameters_to_dict(p)

    # Remove a non-nested key in the dictionary; this is fine.
    del p_dict['distribution']['validate_args']

    # We can reconstruct from this (we just use the default value from p)
    utils.merge_to_parameters_from_dict(p, p_dict)

    # Remove a nested entry in the dictionary; this can lead to subtle errors so
    # we don't allow it.
    del p_dict['bijector']['bijectors:1']

    # Flattening nested params lost information about the nested structure, so
    # we can't e.g. remove a bijector from a list and override just a subset of
    # the nested bijectors list.
    with self.assertRaisesRegex(
        KeyError,
        r'Only saw partial information from the dictionary for nested key '
        r'\'bijectors\' in params_dict.*'
        r'Entries provided: \[\'bijectors:0\'\].*'
        r'Entries required: \[\'bijectors:0\', \'bijectors:1\'\]'):
      utils.merge_to_parameters_from_dict(p, p_dict)


if __name__ == '__main__':
  tf.test.main()
