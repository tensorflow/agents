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

"""Linear Bandit Policy.

LinUCB and Linear Thompson Sampling policies derive from this class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.policies import linalg
from tf_agents.bandits.policies import policy_utilities
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step

tfd = tfp.distributions


class ExplorationStrategy(Enum):
  """Possible exploration strategies."""
  optimistic = 1
  sampling = 2


class LinearBanditPolicy(tf_policy.Base):
  """Linear Bandit Policy to be used by LinUCB, LinTS and possibly others."""

  def __init__(self,
               action_spec,
               cov_matrix,
               data_vector,
               num_samples,
               time_step_spec=None,
               exploration_strategy=ExplorationStrategy.optimistic,
               alpha=1.0,
               eig_vals=(),
               eig_matrix=(),
               tikhonov_weight=1.0,
               add_bias=False,
               emit_policy_info=(),
               emit_log_probability=False,
               observation_and_action_constraint_splitter=None,
               name=None):
    """Initializes `LinearBanditPolicy`.

    The `a` and `b` arguments may be either `Tensor`s or `tf.Variable`s.
    If they are variables, then any assignements to those variables will be
    reflected in the output of the policy.

    Args:
      action_spec: `TensorSpec` containing action specification.
      cov_matrix: list of the covariance matrices A in the paper. There exists
        one A matrix per arm.
      data_vector: list of the b vectors in the paper. The b vector is a
        weighted sum of the observations, where the weight is the corresponding
        reward. Each arm has its own vector b.
      num_samples: list of number of samples per arm.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      exploration_strategy: An Enum of type ExplortionStrategy. The strategy
        used for choosing the actions to incorporate exploration. Currently
        supported strategies are `optimistic` and `sampling`.
      alpha: a float value used to scale the confidence intervals.
      eig_vals: list of eigenvalues for each covariance matrix (one per arm).
      eig_matrix: list of eigenvectors for each covariance matrix (one per arm).
      tikhonov_weight: (float) tikhonov regularization term.
      add_bias: If true, a bias term will be added to the linear reward
        estimation.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      emit_log_probability: Whether to emit log probabilities.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit policy and 2)
        the mask. The mask should be a 0-1 `Tensor` of shape `[batch_size,
        num_actions]`. This function should also work with a `TensorSpec` as
        input, and should output `TensorSpec` objects for the observation and
        mask.
      name: The name of this policy.
    """
    if not isinstance(cov_matrix, (list, tuple)):
      raise ValueError('cov_matrix must be a list of matrices (Tensors).')
    self._cov_matrix = cov_matrix

    if not isinstance(data_vector, (list, tuple)):
      raise ValueError('data_vector must be a list of vectors (Tensors).')
    self._data_vector = data_vector

    if not isinstance(num_samples, (list, tuple)):
      raise ValueError('num_samples must be a list of vectors (Tensors).')
    self._num_samples = num_samples

    if not isinstance(eig_vals, (list, tuple)):
      raise ValueError('eig_vals must be a list of vectors (Tensors).')
    self._eig_vals = eig_vals

    if not isinstance(eig_matrix, (list, tuple)):
      raise ValueError('eig_matrix must be a list of vectors (Tensors).')
    self._eig_matrix = eig_matrix

    self._exploration_strategy = exploration_strategy
    if exploration_strategy == ExplorationStrategy.sampling:
      # We do not have a way to calculate log probabilities for TS yet.
      emit_log_probability = False

    self._alpha = alpha
    self._use_eigendecomp = False
    if eig_matrix:
      self._use_eigendecomp = True
    self._tikhonov_weight = tikhonov_weight
    self._add_bias = add_bias

    if len(cov_matrix) != len(data_vector):
      raise ValueError('The size of list cov_matrix must match the size of '
                       'list data_vector. Got {} for cov_matrix and {} '
                       'for data_vector'.format(
                           len(self._cov_matrix), len((data_vector))))
    if len(num_samples) != len(cov_matrix):
      raise ValueError('The size of num_samples must match the size of '
                       'list cov_matrix. Got {} for num_samples and {} '
                       'for cov_matrix'.format(
                           len(self._num_samples), len((cov_matrix))))
    if tf.nest.is_nested(action_spec):
      raise ValueError('Nested `action_spec` is not supported.')

    self._num_actions = action_spec.maximum + 1
    if self._num_actions != len(cov_matrix):
      raise ValueError(
          'The number of elements in `cov_matrix` ({}) must match '
          'the number of actions derived from `action_spec` ({}).'.format(
              len(cov_matrix), self._num_actions))
    if observation_and_action_constraint_splitter is not None:
      context_shape = observation_and_action_constraint_splitter(
          time_step_spec.observation)[0].shape.as_list()
    else:
      context_shape = time_step_spec.observation.shape.as_list()
    self._context_dim = (
        tf.compat.dimension_value(context_shape[0]) if context_shape else 1)
    if self._add_bias:
      # The bias is added via a constant 1 feature.
      self._context_dim += 1
    cov_matrix_dim = tf.compat.dimension_value(cov_matrix[0].shape[0])
    if self._context_dim != cov_matrix_dim:
      raise ValueError('The dimension of matrix `cov_matrix` must match '
                       'context dimension {}.'
                       'Got {} for `cov_matrix`.'.format(
                           self._context_dim, cov_matrix_dim))

    data_vector_dim = tf.compat.dimension_value(data_vector[0].shape[0])
    if self._context_dim != data_vector_dim:
      raise ValueError('The dimension of vector `data_vector` must match '
                       'context  dimension {}. '
                       'Got {} for `data_vector`.'.format(
                           self._context_dim, data_vector_dim))

    self._dtype = self._data_vector[0].dtype
    self._emit_policy_info = emit_policy_info
    predicted_rewards_mean = ()
    if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info:
      predicted_rewards_mean = tensor_spec.TensorSpec([self._num_actions],
                                                      dtype=self._dtype)
    predicted_rewards_sampled = ()
    if (policy_utilities.InfoFields.PREDICTED_REWARDS_SAMPLED in
        emit_policy_info):
      predicted_rewards_sampled = tensor_spec.TensorSpec([self._num_actions],
                                                         dtype=self._dtype)
    info_spec = policy_utilities.PolicyInfo(
        predicted_rewards_mean=predicted_rewards_mean,
        predicted_rewards_sampled=predicted_rewards_sampled)

    super(LinearBanditPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  def _variables(self):
    all_vars = (self._cov_matrix + self._data_vector + self._num_samples +
                list(self._eig_matrix) + list(self._eig_vals))
    return [v for v in all_vars if isinstance(v, tf.Variable)]

  def _distribution(self, time_step, policy_state):
    observation = time_step.observation
    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)
    if observation_and_action_constraint_splitter is not None:
      observation, mask = observation_and_action_constraint_splitter(
          observation)
    observation = tf.cast(observation, dtype=self._dtype)
    if self._add_bias:
      # The bias is added via a constant 1 feature.
      observation = tf.concat([
          observation,
          tf.ones([tf.shape(observation)[0], 1], dtype=self._dtype)
      ],
                              axis=1)
    # Check the shape of the observation matrix. The observations can be
    # batched.
    if not observation.shape.is_compatible_with([None, self._context_dim]):
      raise ValueError('Observation shape is expected to be {}. Got {}.'.format(
          [None, self._context_dim], observation.shape.as_list()))
    observation = tf.reshape(observation, [-1, self._context_dim])

    est_rewards = []
    confidence_intervals = []
    for k in range(self._num_actions):
      if self._use_eigendecomp:
        q_t_b = tf.matmul(
            self._eig_matrix[k],
            tf.linalg.matrix_transpose(observation),
            transpose_a=True)
        lambda_inv = tf.divide(
            tf.ones_like(self._eig_vals[k]),
            self._eig_vals[k] + self._tikhonov_weight)
        a_inv_x = tf.matmul(
            self._eig_matrix[k], tf.einsum('j,jk->jk', lambda_inv, q_t_b))
      else:
        a_inv_x = linalg.conjugate_gradient_solve(
            self._cov_matrix[k] +
            self._tikhonov_weight * tf.eye(
                self._context_dim, dtype=self._dtype),
            tf.linalg.matrix_transpose(observation))
      est_mean_reward = tf.einsum('j,jk->k', self._data_vector[k], a_inv_x)
      est_rewards.append(est_mean_reward)

      ci = tf.reshape(
          tf.linalg.tensor_diag_part(tf.matmul(observation, a_inv_x)),
          [-1, 1])
      confidence_intervals.append(ci)

    if self._exploration_strategy == ExplorationStrategy.optimistic:
      optimistic_estimates = [
          tf.reshape(mean_reward, [-1, 1]) + self._alpha * tf.sqrt(confidence)
          for mean_reward, confidence in zip(est_rewards, confidence_intervals)
      ]
      # Keeping the batch dimension during the squeeze, even if batch_size == 1.
      rewards_for_argmax = tf.squeeze(
          tf.stack(optimistic_estimates, axis=-1), axis=[1])
    elif self._exploration_strategy == ExplorationStrategy.sampling:
      mu_sampler = tfd.Normal(
          loc=tf.stack(est_rewards, axis=-1),
          scale=self._alpha *
          tf.sqrt(tf.squeeze(tf.stack(confidence_intervals, axis=-1), axis=1)))
      rewards_for_argmax = mu_sampler.sample()
    else:
      raise ValueError('Exploraton strategy %s not implemented.' %
                       self._exploration_strategy)
    if observation_and_action_constraint_splitter is not None:
      chosen_actions = policy_utilities.masked_argmax(
          rewards_for_argmax, mask, output_type=self._action_spec.dtype)
    else:
      chosen_actions = tf.argmax(
          rewards_for_argmax, axis=-1, output_type=self._action_spec.dtype)

    action_distributions = tfp.distributions.Deterministic(loc=chosen_actions)

    policy_info = policy_utilities.PolicyInfo(
        predicted_rewards_sampled=(
            rewards_for_argmax if policy_utilities.InfoFields
            .PREDICTED_REWARDS_SAMPLED in self._emit_policy_info else ()),
        predicted_rewards_mean=(
            tf.stack(est_rewards, axis=-1) if policy_utilities.InfoFields
            .PREDICTED_REWARDS_MEAN in self._emit_policy_info else ()))

    return policy_step.PolicyStep(
        action_distributions, policy_state, policy_info)
