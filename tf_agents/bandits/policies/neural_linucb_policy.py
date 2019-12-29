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

"""Neural + LinUCB Policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.bandits.policies import linalg
from tf_agents.bandits.policies import policy_utilities
from tf_agents.distributions import masked
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step

tfd = tfp.distributions


class NeuralLinUCBPolicy(tf_policy.Base):
  """Neural LinUCB Policy.

  Applies LinUCB on top of an encoding network.
  Since LinUCB is a linear method, the encoding network is used to capture the
  non-linear relationship between the context features and the expected rewards.
  The policy starts with exploration based on epsilon greedy and then switches
  to LinUCB for exploring more efficiently.

  Reference:
  Carlos Riquelme, George Tucker, Jasper Snoek,
  `Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep
  Networks for Thompson Sampling`, ICLR 2018.
  """

  def __init__(self,
               encoding_network,
               encoding_dim,
               reward_layer,
               epsilon_greedy,
               actions_from_reward_layer,
               cov_matrix,
               data_vector,
               num_samples,
               time_step_spec=None,
               alpha=1.0,
               emit_policy_info=(),
               emit_log_probability=False,
               observation_and_action_constraint_splitter=None,
               name=None):
    """Initializes `NeuralLinUCBPolicy`.

    Args:
      encoding_network: network that encodes the observations.
      encoding_dim: (int) dimension of the encoded observations.
      reward_layer: final layer that predicts the expected reward per arm.
      epsilon_greedy: (float) representing the probability of choosing a random
        action instead of the greedy action.
      actions_from_reward_layer: (bool) whether to get actions from the reward
        layer or from LinUCB.
      cov_matrix: list of the covariance matrices. There exists one covariance
        matrix per arm.
      data_vector: list of the data vectors. A data vector is a weighted sum
        of the observations, where the weight is the corresponding reward. Each
        arm has its own data vector.
      num_samples: list of number of samples per arm.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      alpha: (float) non-negative weight multiplying the confidence intervals.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      emit_log_probability: (bool) whether to emit log probabilities.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit policy and 2)
        the mask. The mask should be a 0-1 `Tensor` of shape
        `[batch_size, num_actions]`. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      name: The name of this policy.
    """
    encoding_network.create_variables()
    self._encoding_network = encoding_network
    self._reward_layer = reward_layer
    self._encoding_dim = encoding_dim

    if not isinstance(cov_matrix, (list, tuple)):
      raise ValueError('cov_matrix must be a list of matrices (Tensors).')
    self._cov_matrix = cov_matrix

    if not isinstance(data_vector, (list, tuple)):
      raise ValueError('data_vector must be a list of vectors (Tensors).')
    self._data_vector = data_vector

    if not isinstance(num_samples, (list, tuple)):
      raise ValueError('num_samples must be a list of vectors (Tensors).')
    self._num_samples = num_samples

    self._alpha = alpha
    self._actions_from_reward_layer = actions_from_reward_layer
    self._epsilon_greedy = epsilon_greedy
    self._dtype = self._data_vector[0].dtype

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

    self._num_actions = len(cov_matrix)
    assert self._num_actions
    if observation_and_action_constraint_splitter is not None:
      context_shape = observation_and_action_constraint_splitter(
          time_step_spec.observation)[0].shape.as_list()
    else:
      context_shape = time_step_spec.observation.shape.as_list()
    self._context_dim = (
        tf.compat.dimension_value(context_shape[0]) if context_shape else 1)
    cov_matrix_dim = tf.compat.dimension_value(cov_matrix[0].shape[0])
    if self._encoding_dim != cov_matrix_dim:
      raise ValueError('The dimension of matrix `cov_matrix` must match '
                       'encoding dimension {}.'
                       'Got {} for `cov_matrix`.'.format(
                           self._encoding_dim, cov_matrix_dim))
    data_vector_dim = tf.compat.dimension_value(data_vector[0].shape[0])
    if self._encoding_dim != data_vector_dim:
      raise ValueError('The dimension of vector `data_vector` must match '
                       'encoding  dimension {}. '
                       'Got {} for `data_vector`.'.format(
                           self._encoding_dim, data_vector_dim))
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        minimum=0,
        maximum=self._num_actions - 1,
        name='action')

    self._emit_policy_info = emit_policy_info
    predicted_rewards_mean = ()
    if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info:
      predicted_rewards_mean = tensor_spec.TensorSpec(
          [self._num_actions],
          dtype=tf.float32)
    info_spec = policy_utilities.PolicyInfo(
        predicted_rewards_mean=predicted_rewards_mean)

    super(NeuralLinUCBPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        info_spec=info_spec,
        name=name)

  def _variables(self):
    all_variables = (self._cov_matrix + self._data_vector +
                     self._num_samples + self._encoding_network.variables +
                     self._reward_layer.variables)
    return [v for v in all_variables if isinstance(v, tf.Variable)]

  def _get_actions_from_reward_layer(self, encoded_observation, mask):
    # Get the predicted expected reward.
    est_mean_reward = self._reward_layer(encoded_observation)
    if mask is None:
      greedy_actions = tf.argmax(est_mean_reward, axis=-1, output_type=tf.int32)
    else:
      greedy_actions = policy_utilities.masked_argmax(
          est_mean_reward, mask, output_type=tf.int32)

    # Add epsilon greedy on top, if needed.
    if self._epsilon_greedy:
      batch_size = (tf.compat.dimension_value(encoded_observation.shape[0]) or
                    tf.shape(encoded_observation)[0])
      if mask is None:
        random_actions = tf.random.uniform(
            [batch_size], maxval=self._num_actions,
            dtype=tf.int32)
      else:
        zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
        masked_categorical = masked.MaskedCategorical(
            zero_logits, mask, dtype=tf.int32)
        random_actions = masked_categorical.sample()

      rng = tf.random.uniform([batch_size], maxval=1.0)
      cond = tf.greater(rng, self._epsilon_greedy)
      chosen_actions = tf.compat.v1.where(cond, greedy_actions, random_actions)
    else:
      chosen_actions = greedy_actions

    return chosen_actions, est_mean_reward

  def _get_actions_from_linucb(self, encoded_observation, mask):
    encoded_observation = tf.cast(encoded_observation, dtype=self._dtype)

    p_values = []
    est_rewards = []
    for k in range(self._num_actions):
      a_inv_x = linalg.conjugate_gradient_solve(
          self._cov_matrix[k] + tf.eye(self._encoding_dim, dtype=self._dtype),
          tf.linalg.matrix_transpose(encoded_observation))
      mean_reward_est = tf.einsum('j,jk->k', self._data_vector[k], a_inv_x)
      est_rewards.append(mean_reward_est)

      ci = tf.reshape(
          tf.linalg.tensor_diag_part(tf.matmul(encoded_observation, a_inv_x)),
          [-1, 1])
      p_values.append(
          tf.reshape(mean_reward_est, [-1, 1]) + self._alpha * tf.sqrt(ci))

    stacked_p_values = tf.squeeze(tf.stack(p_values, axis=-1), axis=[1])
    if mask is None:
      chosen_actions = tf.argmax(
          stacked_p_values,
          axis=-1,
          output_type=tf.int32)
    else:
      chosen_actions = policy_utilities.masked_argmax(
          stacked_p_values, mask, output_type=tf.int32)

    est_mean_reward = tf.cast(tf.stack(est_rewards, axis=-1), tf.float32)
    return chosen_actions, est_mean_reward

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'This policy outputs an action and not a distribution.')

  def _action(self, time_step, policy_state, seed):
    observation = time_step.observation
    mask = None
    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)
    if observation_and_action_constraint_splitter is not None:
      observation, mask = observation_and_action_constraint_splitter(
          observation)
    # Check the shape of the observation matrix.
    if not observation.shape.is_compatible_with([None, self._context_dim]):
      raise ValueError('Observation shape is expected to be {}. Got {}.'.format(
          [None, self._context_dim], observation.shape.as_list()))

    observation = tf.cast(observation, dtype=self._dtype)

    # Pass the observations through the encoding network.
    encoded_observation, _ = self._encoding_network(observation)

    chosen_actions, est_mean_rewards = tf.cond(
        self._actions_from_reward_layer,
        lambda: self._get_actions_from_reward_layer(encoded_observation, mask),
        lambda: self._get_actions_from_linucb(encoded_observation, mask))

    policy_info = policy_utilities.PolicyInfo(
        predicted_rewards_mean=(
            est_mean_rewards if
            policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in
            self._emit_policy_info else ()))
    return policy_step.PolicyStep(chosen_actions, policy_state, policy_info)
