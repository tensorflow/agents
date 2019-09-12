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

"""Linear Thompson Sampling Policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.bandits.policies import linalg
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

tfd = tfp.distributions


def _get_means_and_variances(parameter_estimators, weight_covariances,
                             observation):
  """Helper function that calculates means and variances for reward sampling."""
  means = []
  variances = []
  for k in range(len(parameter_estimators)):
    obs_x_inv_cov = tf.transpose(
        linalg.conjugate_gradient_solve(
            weight_covariances[k],
            tf.transpose(observation)))
    means.append(tf.linalg.matvec(obs_x_inv_cov, parameter_estimators[k]))
    variances.append(
        tf.linalg.tensor_diag_part(
            tf.matmul(obs_x_inv_cov, observation, transpose_b=True)))
  return means, variances


def _assert_shape(expected_shape, actual_shape, object_name):
  """Asserts shape matches the expected of a given object.

  Args:
    expected_shape: List of ints, the expected shape.
    actual_shape: List of ints, the actual shape of the object.
    object_name: Name of the object under scrutiny, for informative error
      message.

  Raises:
    Value error if the shapes don't match.
  """
  if expected_shape != actual_shape:
    raise ValueError('{} dimension mismatch. Expected shape {}; got {}'.format(
        object_name, expected_shape, actual_shape))


class LinearThompsonSamplingPolicy(tf_policy.Base):
  """Linear Thompson Sampling Policy.

  Implements the Linear Thompson Sampling Policy from the following paper:
  "Thompson Sampling for Contextual Bandits with Linear Payoffs",
  Shipra Agrawal, Navin Goyal, ICML 2013. The actual algorithm implemented is
  `Algorithm 3` from the supplementary material of the paper from
  `http://proceedings.mlr.press/v28/agrawal13-supp.pdf`.

  In a nutshell, the algorithm estimates reward distributions based on
  parameters `B_inv` and `f` for every action, where here we denote `B_inv` as
  `weight_covariance_matrices` and `f` as `parameter_estimators`. Then for each
  action we sample a reward and take the argmax.
  """

  def __init__(
      self,
      action_spec,
      weight_covariance_matrices,
      parameter_estimators,
      name=None):
    """Initializes `LinearThompsonSamplingPolicy`.

    The `weight_covariance_matrices` and `parameter_estimators`
      arguments may either be `Tensor`s or `tf.Variable`s. If they are
      variables, then any assignment to those variables will be reflected in the
      output of the policy.

    Args:
      action_spec: Array spec containing action specification.
      weight_covariance_matrices: A list of `B` inverse matrices from the paper.
        The list has `num_actions` elements of shape
        `[context_dim, context_dim]`.
      parameter_estimators: List of `f` vectors from the paper. The list has
        `num_actions' elements of shape is `[context_dim]`.
      name: The name of this policy.
    """
    if not isinstance(weight_covariance_matrices, (list, tuple)):
      raise ValueError(
          'weight_covariances must be a list of matrices (Tensors).')
    self._weight_covariance_matrices = weight_covariance_matrices

    if not isinstance(parameter_estimators, (list, tuple)):
      raise ValueError(
          'parameter_estimators must be a list of vectors (Tensors).')
    self._parameter_estimators = parameter_estimators

    self._action_spec = action_spec
    self._num_actions = action_spec.maximum + 1
    self._context_dim = tf.compat.dimension_value(
        parameter_estimators[0].shape[0])

    self._variables = [
        x for x in weight_covariance_matrices + parameter_estimators
        if isinstance(x, tf.Variable)
    ]
    for t in self._parameter_estimators:
      _assert_shape([self._context_dim], t.shape.as_list(),
                    'Parameter estimators')
    for t in self._weight_covariance_matrices:
      _assert_shape([self._context_dim, self._context_dim], t.shape.as_list(),
                    'Weight covariance')

    observation_spec = tensor_spec.TensorSpec(
        shape=(self._context_dim), dtype=tf.float32)

    time_step_spec = ts.time_step_spec(observation_spec)
    super(LinearThompsonSamplingPolicy, self).__init__(
        time_step_spec=time_step_spec, action_spec=action_spec)

  def _variables(self):
    return self._variables

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'This policy outputs an action and not a distribution.')

  def _action(self, time_step, policy_state, seed):
    seed_stream = tfd.SeedStream(seed=seed, salt='ts_policy')

    observation = tf.cast(
        time_step.observation, dtype=self._parameter_estimators[0].dtype)
    mean_estimates, scales = _get_means_and_variances(
        self._parameter_estimators, self._weight_covariance_matrices,
        observation)
    mu_sampler = tfd.Normal(
        loc=tf.stack(mean_estimates, axis=-1),
        scale=tf.sqrt(tf.stack(scales, axis=-1)))
    reward_samples = mu_sampler.sample(seed=seed_stream())
    actions = tf.argmax(
        reward_samples, axis=-1, output_type=self._action_spec.dtype)
    return policy_step.PolicyStep(actions, policy_state)
