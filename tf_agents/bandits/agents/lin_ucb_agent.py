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

"""Implements the Linear UCB bandit algorithm.

  Reference:
  "A Contextual Bandit Approach to Personalized News Article Recommendation",
  Lihong Li, Wei Chu, John Langford, Robert Schapire, WWW 2010.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.policies import lin_ucb_policy
from tf_agents.bandits.policies import linalg
from tf_agents.utils import nest_utils


def update_a_and_b_with_forgetting(
    a_prev, b_prev, r, x, gamma, compute_eigendecomp=False):
  r"""Update the covariance matrix `a` and the weighted sum of rewards `b`.

  This function updates the covariance matrix `a` and the sum of weighted
  rewards `b` using a forgetting factor `gamma`.

  Args:
    a_prev: previous estimate of `a`.
    b_prev: previous estimate of `b`.
    r: a `Tensor` of shape [`batch_size`]. This is the rewards of the batched
      observations.
    x: a `Tensor` of shape [`batch_size`, `context_dim`]. This is the matrix
      with the (batched) observations.
    gamma: a float forgetting factor in [0.0, 1.0].
    compute_eigendecomp: whether to compute the eigen-decomposition of the new
      covariance matrix.

  Returns:
    The updated estimates of `a` and `b` and optionally the eigenvalues and
    eigenvectors of `a`.
  """
  a_new = gamma * a_prev + tf.matmul(x, x, transpose_a=True)
  b_new = gamma * b_prev + bandit_utils.sum_reward_weighted_observations(r, x)

  eig_vals = tf.constant([], dtype=a_new.dtype)
  eig_matrix = tf.constant([], dtype=a_new.dtype)
  if compute_eigendecomp:
    eig_vals, eig_matrix = tf.linalg.eigh(a_new)
  return a_new, b_new, eig_vals, eig_matrix


@gin.configurable
class LinearUCBAgent(tf_agent.TFAgent):
  """An agent implementing the Linear UCB bandit algorithm.

  Reference:
  "A Contextual Bandit Approach to Personalized News Article Recommendation",
  Lihong Li, Wei Chu, John Langford, Robert Schapire, WWW 2010.
  """

  def __init__(self,
               time_step_spec,
               action_spec,
               alpha=1.0,
               gamma=1.0,
               use_eigendecomp=False,
               tikhonov_weight=1.0,
               expose_predicted_rewards=False,
               emit_log_probability=False,
               observation_and_action_constraint_splitter=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               enable_summaries=True,
               dtype=tf.float32,
               name=None):
    """Initialize an instance of `LinearUCBAgent`.

    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
      alpha: (float) positive scalar. This is the exploration parameter that
        multiplies the confidence intervals.
      gamma: a float forgetting factor in [0.0, 1.0]. When set to
        1.0, the algorithm does not forget.
      use_eigendecomp: whether to use eigen-decomposition or not. The default
        solver is Conjugate Gradient.
      tikhonov_weight: (float) tikhonov regularization term.
      expose_predicted_rewards: (bool) Whether to expose the predicted rewards
        in the policy info field under the name 'predicted_rewards'.
      emit_log_probability: Whether the LinearUCBPolicy emits log-probabilities
        or not. Since the policy is deterministic, the probability is just 1.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      dtype: The type of the parameters stored and updated by the agent. Should
        be one of `tf.float32` and `tf.float64`. Defaults to `tf.float32`.
      name: a name for this instance of `LinearUCBAgent`.

    Raises:
      ValueError if dtype is not one of `tf.float32` or `tf.float64`.
    """
    tf.Module.__init__(self, name=name)
    self._num_actions = bandit_utils.get_num_actions_from_tensor_spec(
        action_spec)
    if observation_and_action_constraint_splitter:
      context_shape = observation_and_action_constraint_splitter(
          time_step_spec.observation)[0].shape.as_list()
    else:
      context_shape = time_step_spec.observation.shape.as_list()
    self._context_dim = (
        tf.compat.dimension_value(context_shape[0]) if context_shape else 1)
    self._alpha = alpha
    self._cov_matrix_list = []
    self._data_vector_list = []
    self._eig_matrix_list = []
    self._eig_vals_list = []
    # We keep track of the number of samples per arm.
    self._num_samples_list = []
    self._gamma = gamma
    if self._gamma < 0.0 or self._gamma > 1.0:
      raise ValueError('Forgetting factor `gamma` must be in [0.0, 1.0].')
    self._dtype = dtype
    if dtype not in (tf.float32, tf.float64):
      raise ValueError(
          'Agent dtype should be either `tf.float32 or `tf.float64`.')
    self._use_eigendecomp = use_eigendecomp
    self._tikhonov_weight = tikhonov_weight
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)

    for k in range(self._num_actions):
      self._cov_matrix_list.append(
          tf.compat.v2.Variable(
              tf.eye(self._context_dim, dtype=dtype), name='a_' + str(k)))
      self._data_vector_list.append(
          tf.compat.v2.Variable(
              tf.zeros(self._context_dim, dtype=dtype), name='b_' + str(k)))
      self._num_samples_list.append(
          tf.compat.v2.Variable(
              tf.zeros([], dtype=dtype), name='num_samples_' + str(k)))
      if self._use_eigendecomp:
        self._eig_matrix_list.append(
            tf.compat.v2.Variable(
                tf.eye(self._context_dim, dtype=dtype),
                name='eig_matrix' + str(k)))
        self._eig_vals_list.append(
            tf.compat.v2.Variable(
                tf.ones([self._context_dim], dtype=dtype),
                name='eig_vals' + str(k)))
      else:
        self._eig_matrix_list.append(
            tf.compat.v2.Variable(
                tf.constant([], dtype=dtype),
                name='eig_matrix' + str(k)))
        self._eig_vals_list.append(
            tf.compat.v2.Variable(
                tf.constant([], dtype=dtype),
                name='eig_vals' + str(k)))

    policy = lin_ucb_policy.LinearUCBPolicy(
        action_spec=action_spec,
        cov_matrix=self._cov_matrix_list,
        data_vector=self._data_vector_list,
        num_samples=self._num_samples_list,
        time_step_spec=time_step_spec,
        alpha=alpha,
        eig_vals=self._eig_vals_list if self._use_eigendecomp else (),
        eig_matrix=self._eig_matrix_list if self._use_eigendecomp else (),
        tikhonov_weight=self._tikhonov_weight,
        expose_predicted_rewards=expose_predicted_rewards,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
    )
    super(LinearUCBAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy=policy,
        collect_policy=policy,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        train_sequence_length=None)

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def cov_matrix(self):
    return self._cov_matrix_list

  @property
  def eig_matrix(self):
    return self._eig_matrix_list

  @property
  def eig_vals(self):
    return self._eig_vals_list

  @property
  def data_vector(self):
    return self._data_vector_list

  @property
  def num_samples(self):
    return self._num_samples_list

  @property
  def alpha(self):
    return self._alpha

  def update_alpha(self, alpha):
    return tf.compat.v1.assign(self._alpha, alpha)

  @property
  def theta(self):
    """Returns the matrix of per-arm feature weights.

    The returned matrix has shape (num_actions, context_dim).
    It's equivalent to a stacking of theta vectors from the paper.
    """
    thetas = []
    for k in range(self._num_actions):
      thetas.append(
          tf.squeeze(
              linalg.conjugate_gradient_solve(
                  self._cov_matrix_list[k] + self._tikhonov_weight *
                  tf.eye(self._context_dim, dtype=self._dtype),
                  tf.expand_dims(self._data_vector_list[k], axis=-1)),
              axis=-1))

    return tf.stack(thetas, axis=0)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def compute_summaries(self, loss):
    if self.summaries_enabled:
      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='loss', data=loss, step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self.policy.variables():
            var_name = var.name.replace(':', '_')
            tf.compat.v2.summary.histogram(
                name=var_name,
                data=var,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name=var_name + '_value_norm',
                data=tf.linalg.global_norm([var]),
                step=self.train_step_counter)

  def _train(self, experience, weights=None):
    """Updates the policy based on the data in `experience`.

    Note that `experience` should only contain data points that this agent has
    not previously seen. If `experience` comes from a replay buffer, this buffer
    should be cleared between each call to `train`.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`.
      weights: Unused.

    Returns:
        A `LossInfo` containing the loss *before* the training step is taken.
        In most cases, if `weights` is provided, the entries of this tuple will
        have been calculated with the weights.  Note that each Agent chooses
        its own method of applying weights.
    """
    del weights  # unused

    # If the experience comes from a replay buffer, the reward has shape:
    #     [batch_size, time_steps]
    # where `time_steps` is the number of driver steps executed in each
    # training loop.
    # We flatten the tensors below in order to reflect the effective batch size.

    reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._time_step_spec.reward)
    action, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.action, self._action_spec)
    observation, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.observation, self._time_step_spec.observation)

    if self._observation_and_action_constraint_splitter:
      observation, _ = self._observation_and_action_constraint_splitter(
          observation)
    observation = tf.reshape(observation, [-1, self._context_dim])
    observation = tf.cast(observation, self._dtype)
    reward = tf.cast(reward, self._dtype)

    for k in range(self._num_actions):
      diag_mask = tf.linalg.tensor_diag(
          tf.cast(tf.equal(action, k), self._dtype))
      observations_for_arm = tf.matmul(diag_mask, observation)
      rewards_for_arm = tf.matmul(diag_mask, tf.reshape(reward, [-1, 1]))

      num_samples_for_arm_current = tf.reduce_sum(diag_mask)
      tf.compat.v1.assign_add(self._num_samples_list[k],
                              num_samples_for_arm_current)
      num_samples_for_arm_total = self._num_samples_list[k].read_value()

      # Update the matrix A and b.
      # pylint: disable=cell-var-from-loop,g-long-lambda
      def update(cov_matrix, data_vector):
        return update_a_and_b_with_forgetting(
            cov_matrix, data_vector, rewards_for_arm, observations_for_arm,
            self._gamma, self._use_eigendecomp)
      a_new, b_new, eig_vals, eig_matrix = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0,
          lambda: update(self._cov_matrix_list[k], self._data_vector_list[k]),
          lambda: (self._cov_matrix_list[k], self._data_vector_list[k],
                   self._eig_vals_list[k], self._eig_matrix_list[k]))

      tf.compat.v1.assign(self._cov_matrix_list[k], a_new)
      tf.compat.v1.assign(self._data_vector_list[k], b_new)
      tf.compat.v1.assign(self._eig_vals_list[k], eig_vals)
      tf.compat.v1.assign(self._eig_matrix_list[k], eig_matrix)

    loss = -1. * tf.reduce_sum(experience.reward)
    self.compute_summaries(loss)

    batch_size = tf.cast(
        tf.compat.dimension_value(tf.shape(reward)[0]), dtype=tf.int64)
    self._train_step_counter.assign_add(batch_size)

    return tf_agent.LossInfo(loss=(loss), extra=())
