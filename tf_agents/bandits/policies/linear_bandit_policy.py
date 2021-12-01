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

"""Linear Bandit Policy.

LinUCB and Linear Thompson Sampling policies derive from this class.

This linear policy handles two main forms of feature input.
1. A single global feature is received per time step. In this case, the policy
maintains an independent linear reward model for each arm.
2. Apart from the global feature as in case 1, an arm-feature vector is
received for each arm in every time step. In this case, only one model is
maintained by the policy, and the reward estimates are calculated for every arm
by using their own per-arm features.

The above two cases can be triggered by setting the boolean parameter
`accepts_per_arm_features` appropriately.

A detailed explanation for the two above cases can be found in the paper
"Thompson Sampling for Contextual Bandits with Linear Payoffs",
Shipra Agrawal, Navin Goyal, ICML 2013
(http://proceedings.mlr.press/v28/agrawal13.pdf), and its supplementary material
(http://proceedings.mlr.press/v28/agrawal13-supp.pdf).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from typing import Optional, Sequence, Text

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.policies import linalg
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

tfd = tfp.distributions


class ExplorationStrategy(Enum):
  """Possible exploration strategies."""
  optimistic = 1
  sampling = 2


class LinearBanditPolicy(tf_policy.TFPolicy):
  """Linear Bandit Policy to be used by LinUCB, LinTS and possibly others."""

  def __init__(self,
               action_spec: types.BoundedTensorSpec,
               cov_matrix: Sequence[types.Float],
               data_vector: Sequence[types.Float],
               num_samples: Sequence[types.Int],
               time_step_spec: Optional[types.TimeStep] = None,
               exploration_strategy: ExplorationStrategy = ExplorationStrategy
               .optimistic,
               alpha: float = 1.0,
               eig_vals: Sequence[types.Float] = (),
               eig_matrix: Sequence[types.Float] = (),
               tikhonov_weight: float = 1.0,
               add_bias: bool = False,
               emit_policy_info: Sequence[Text] = (),
               emit_log_probability: bool = False,
               accepts_per_arm_features: bool = False,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               name: Optional[Text] = None):
    """Initializes `LinearBanditPolicy`.

    The `a` and `b` arguments may be either `Tensor`s or `tf.Variable`s.
    If they are variables, then any assignements to those variables will be
    reflected in the output of the policy.

    Args:
      action_spec: `TensorSpec` containing action specification.
      cov_matrix: list of the covariance matrices A in the paper. If the policy
        accepts per-arm features, the lenght of this list is 1, as there is only
        one model. Otherwise, there is one A matrix per arm.
      data_vector: list of the b vectors in the paper. The b vector is a
        weighted sum of the observations, where the weight is the corresponding
        reward. If the policy accepts per-arm features, this list should be of
        length 1, as there only 1 reward model maintained. Otherwise, each arm
        has its own vector b.
      num_samples: list of number of samples per arm, unless the policy accepts
      per-arm features, in which case this is just the number of samples seen.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      exploration_strategy: An Enum of type ExplortionStrategy. The strategy
        used for choosing the actions to incorporate exploration. Currently
        supported strategies are `optimistic` and `sampling`.
      alpha: a float value used to scale the confidence intervals.
      eig_vals: list of eigenvalues for each covariance matrix (one per arm,
        unless the policy accepts per-arm features).
      eig_matrix: list of eigenvectors for each covariance matrix (one per arm,
        unless the policy accepts per-arm features).
      tikhonov_weight: (float) tikhonov regularization term.
      add_bias: If true, a bias term will be added to the linear reward
        estimation.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      emit_log_probability: Whether to emit log probabilities.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
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
    policy_utilities.check_no_mask_with_arm_features(
        accepts_per_arm_features, observation_and_action_constraint_splitter)
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
    self._accepts_per_arm_features = accepts_per_arm_features
    if tf.nest.is_nested(action_spec):
      raise ValueError('Nested `action_spec` is not supported.')

    self._num_actions = action_spec.maximum + 1
    self._check_input_variables()
    if observation_and_action_constraint_splitter is not None:
      context_spec, _ = observation_and_action_constraint_splitter(
          time_step_spec.observation)
    else:
      context_spec = time_step_spec.observation
    (self._global_context_dim,
     self._arm_context_dim) = bandit_spec_utils.get_context_dims_from_spec(
         context_spec, accepts_per_arm_features)

    if self._add_bias:
      # The bias is added via a constant 1 feature.
      self._global_context_dim += 1
    self._overall_context_dim = self._global_context_dim + self._arm_context_dim
    cov_matrix_dim = tf.compat.dimension_value(cov_matrix[0].shape[0])
    if self._overall_context_dim != cov_matrix_dim:
      raise ValueError('The dimension of matrix `cov_matrix` must match '
                       'overall context dimension {}. '
                       'Got {} for `cov_matrix`.'.format(
                           self._overall_context_dim, cov_matrix_dim))

    data_vector_dim = tf.compat.dimension_value(data_vector[0].shape[0])
    if self._overall_context_dim != data_vector_dim:
      raise ValueError('The dimension of vector `data_vector` must match '
                       'context  dimension {}. '
                       'Got {} for `data_vector`.'.format(
                           self._overall_context_dim, data_vector_dim))

    self._dtype = self._data_vector[0].dtype
    self._emit_policy_info = emit_policy_info
    info_spec = self._populate_policy_info_spec(
        time_step_spec.observation, observation_and_action_constraint_splitter)

    super(LinearBanditPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  def _variables(self):
    all_vars = [self._cov_matrix,
                self._data_vector,
                self._num_samples,
                self._eig_matrix,
                self._eig_vals]
    return [v for v in tf.nest.flatten(all_vars) if isinstance(v, tf.Variable)]

  def _distribution(self, time_step, policy_state):
    observation = time_step.observation
    if self.observation_and_action_constraint_splitter is not None:
      observation, _ = self.observation_and_action_constraint_splitter(
          observation)
    observation = tf.nest.map_structure(lambda o: tf.cast(o, dtype=self._dtype),
                                        observation)
    global_observation, arm_observations = self._split_observation(observation)

    if self._add_bias:
      # The bias is added via a constant 1 feature.
      global_observation = tf.concat([
          global_observation,
          tf.ones([tf.shape(global_observation)[0], 1], dtype=self._dtype)
      ],
                                     axis=1)
    # Check the shape of the observation matrix. The observations can be
    # batched.
    if not global_observation.shape.is_compatible_with(
        [None, self._global_context_dim]):
      raise ValueError(
          'Global observation shape is expected to be {}. Got {}.'.format(
              [None, self._global_context_dim],
              global_observation.shape.as_list()))
    global_observation = tf.reshape(global_observation,
                                    [-1, self._global_context_dim])

    est_rewards = []
    confidence_intervals = []
    for k in range(self._num_actions):
      current_observation = self._get_current_observation(
          global_observation, arm_observations, k)
      model_index = policy_utilities.get_model_index(
          k, self._accepts_per_arm_features)
      if self._use_eigendecomp:
        q_t_b = tf.matmul(
            self._eig_matrix[model_index],
            tf.linalg.matrix_transpose(current_observation),
            transpose_a=True)
        lambda_inv = tf.divide(
            tf.ones_like(self._eig_vals[model_index]),
            self._eig_vals[model_index] + self._tikhonov_weight)
        a_inv_x = tf.matmul(self._eig_matrix[model_index],
                            tf.einsum('j,jk->jk', lambda_inv, q_t_b))
      else:
        a_inv_x = linalg.conjugate_gradient(
            self._cov_matrix[model_index] + self._tikhonov_weight *
            tf.eye(self._overall_context_dim, dtype=self._dtype),
            tf.linalg.matrix_transpose(current_observation))
      est_mean_reward = tf.einsum('j,jk->k', self._data_vector[model_index],
                                  a_inv_x)
      est_rewards.append(est_mean_reward)

      ci = tf.reshape(
          tf.linalg.tensor_diag_part(tf.matmul(current_observation, a_inv_x)),
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

    mask = constraints.construct_mask_from_multiple_sources(
        time_step.observation, self._observation_and_action_constraint_splitter,
        (), self._num_actions)
    if mask is not None:
      chosen_actions = policy_utilities.masked_argmax(
          rewards_for_argmax,
          mask,
          output_type=tf.nest.flatten(self._action_spec)[0].dtype)
    else:
      chosen_actions = tf.argmax(
          rewards_for_argmax,
          axis=-1,
          output_type=tf.nest.flatten(self._action_spec)[0].dtype)

    action_distributions = tfp.distributions.Deterministic(loc=chosen_actions)

    policy_info = policy_utilities.populate_policy_info(
        arm_observations, chosen_actions, rewards_for_argmax,
        tf.stack(est_rewards, axis=-1), self._emit_policy_info,
        self._accepts_per_arm_features)

    return policy_step.PolicyStep(
        action_distributions, policy_state, policy_info)

  def _check_input_variables(self):
    if len(self._cov_matrix) != len(self._data_vector):
      raise ValueError('The size of list cov_matrix must match the size of '
                       'list data_vector. Got {} for cov_matrix and {} '
                       'for data_vector'.format(
                           len(self._cov_matrix), len((self._data_vector))))
    if len(self._num_samples) != len(self._cov_matrix):
      raise ValueError('The size of num_samples must match the size of '
                       'list cov_matrix. Got {} for num_samples and {} '
                       'for cov_matrix'.format(
                           len(self._num_samples), len((self._cov_matrix))))

    if self._accepts_per_arm_features:
      if len(self._cov_matrix) != 1:
        raise ValueError(
            'If the policy accepts per-arm features, the length of `cov_matrix`'
            ' list must be 1. Got {} instead.'.format(len(self._cov_matrix)))
    else:
      if self._num_actions != len(self._cov_matrix):
        raise ValueError(
            'The number of elements in `cov_matrix` ({}) must match '
            'the number of actions derived from `action_spec` ({}).'.format(
                len(self._cov_matrix), self._num_actions))

  def _populate_policy_info_spec(self, observation_spec,
                                 observation_and_action_constraint_splitter):
    predicted_rewards_mean = ()
    if (policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in
        self._emit_policy_info):
      predicted_rewards_mean = tensor_spec.TensorSpec([self._num_actions],
                                                      dtype=self._dtype)
    predicted_rewards_optimistic = ()
    if (policy_utilities.InfoFields.PREDICTED_REWARDS_OPTIMISTIC in
        self._emit_policy_info):
      predicted_rewards_optimistic = tensor_spec.TensorSpec([self._num_actions],
                                                            dtype=self._dtype)
    predicted_rewards_sampled = ()
    if (policy_utilities.InfoFields.PREDICTED_REWARDS_SAMPLED in
        self._emit_policy_info):
      predicted_rewards_sampled = tensor_spec.TensorSpec([self._num_actions],
                                                         dtype=self._dtype)
    if self._accepts_per_arm_features:
      # The features for the chosen arm is saved to policy_info.
      chosen_arm_features_info = (
          policy_utilities.create_chosen_arm_features_info_spec(
              observation_spec))
      info_spec = policy_utilities.PerArmPolicyInfo(
          predicted_rewards_mean=predicted_rewards_mean,
          predicted_rewards_optimistic=predicted_rewards_optimistic,
          predicted_rewards_sampled=predicted_rewards_sampled,
          chosen_arm_features=chosen_arm_features_info)
    else:
      info_spec = policy_utilities.PolicyInfo(
          predicted_rewards_mean=predicted_rewards_mean,
          predicted_rewards_optimistic=predicted_rewards_optimistic,
          predicted_rewards_sampled=predicted_rewards_sampled)
    return info_spec

  def _get_current_observation(self, global_observation, arm_observations,
                               arm_index):
    """Helper function to construct the observation for a specific arm.

    This function constructs the observation depending if the policy accepts
    per-arm features or not. If not, it simply returns the original observation.
    If yes, it concatenates the global observation with the observation of the
    arm indexed by `arm_index`.

    Args:
      global_observation: A tensor of shape `[batch_size, global_context_dim]`.
        The global part of the observation.
      arm_observations: A tensor of shape `[batch_size, num_actions,
        arm_context_dim]`. The arm part of the observation, for all arms. If the
        policy does not accept per-arm features, this paramater is unused.
      arm_index: (int) The arm for which the observations to be returned.

    Returns:
      A tensor of shape `[batch_size, overall_context_dim]`, containing the
      observation for arm `arm_index`.
    """
    if self._accepts_per_arm_features:
      current_arm = arm_observations[:, arm_index, :]
      current_observation = tf.concat([global_observation, current_arm],
                                      axis=-1)
      return current_observation
    else:
      return global_observation

  def _split_observation(self, observation):
    """Splits the observation into global and arm observations."""
    if self._accepts_per_arm_features:
      global_observation = observation[bandit_spec_utils.GLOBAL_FEATURE_KEY]
      arm_observations = observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]
      if not arm_observations.shape.is_compatible_with(
          [None, self._num_actions, self._arm_context_dim]):
        raise ValueError(
            'Arm observation shape is expected to be {}. Got {}.'.format(
                [None, self._num_actions, self._arm_context_dim],
                arm_observations.shape.as_list()))
    else:
      global_observation = observation
      arm_observations = None
    return global_observation, arm_observations
