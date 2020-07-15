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

"""An agent that uses and trains a greedy reward prediction policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import greedy_reward_prediction_policy as greedy_reward_policy
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.utils import common
from tf_agents.utils import eager_utils


@gin.configurable
class GreedyRewardPredictionAgent(tf_agent.TFAgent):
  """A neural reward network based bandit agent.

  This agent receives a neural network that it trains to predict rewards. The
  action is chosen greedily with respect to the prediction.
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      reward_network,
      optimizer,
      observation_and_action_constraint_splitter=None,
      accepts_per_arm_features=False,
      constraints=(),
      # Params for training.
      error_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gradient_clipping=None,
      # Params for debugging.
      debug_summaries=False,
      summarize_grads_and_vars=False,
      enable_summaries=True,
      emit_policy_info=(),
      train_step_counter=None,
      laplacian_matrix=None,
      laplacian_smoothing_weight=0.001,
      name=None):
    """Creates a Greedy Reward Network Prediction Agent.

     In some use cases, the actions are not independent and they are related to
     each other (e.g., when the actions are ordinal integers). Assuming that
     the relations between arms can be modeled by a graph, we may want to
     enforce that the estimated reward function is smooth over the graph. This
     implies that the estimated rewards `r_i` and `r_j` for two related actions
     `i` and `j`, should be close to each other. To quantify this smoothness
     criterion we use the Laplacian matrix `L` of the graph over the actions.
     When the laplacian smoothing is enabled, the loss is extended to:
     ```
       Loss_new := Loss + lambda r^T * L * r,
     ```
     where `r` is the estimated reward vector for all actions. The second
     term is the laplacian smoothing regularization term and `lambda` is the
     weight that determines how strongly we enforce the regularization.
     For more details, please see:
     "Bandits on graphs and structures", Michal Valko
     https://hal.inria.fr/tel-01359757/document

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      reward_network: A `tf_agents.network.Network` to be used by the agent. The
        network will be called with call(observation, step_type) and it is
        expected to provide a reward prediction for all actions.
      optimizer: The optimizer to use for training.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
      constraints: iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.NeuralConstraint`.
      error_loss_fn: A function for computing the error loss, taking parameters
        labels, predictions, and weights (any function from tf.losses would
        work). The default is `tf.losses.mean_squared_error`.
      gradient_clipping: A float representing the norm length to clip gradients
        (or None for no clipping.)
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      train_step_counter: An optional `tf.Variable` to increment every time the
        train op is run.  Defaults to the `global_step`.
      laplacian_matrix: A float `Tensor` or a numpy array shaped
        `[num_actions, num_actions]`. This holds the Laplacian matrix used to
        regularize the smoothness of the estimated expected reward function.
        This only applies to problems where the actions have a graph structure.
        If `None`, the regularization is not applied.
      laplacian_smoothing_weight: A float that determines the weight of the
        regularization term. Note that this has no effect if `laplacian_matrix`
        above is `None`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or or it is
      not a bounded scalar int32 spec with minimum 0.
      InvalidArgumentError: if the Laplacian provided is not None and not valid.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)
    self._num_actions = bandit_utils.get_num_actions_from_tensor_spec(
        action_spec)
    self._accepts_per_arm_features = accepts_per_arm_features
    self._constraints = constraints

    reward_network.create_variables()
    self._reward_network = reward_network
    self._optimizer = optimizer
    self._error_loss_fn = error_loss_fn
    self._gradient_clipping = gradient_clipping
    self._heteroscedastic = isinstance(
        reward_network, heteroscedastic_q_network.HeteroscedasticQNetwork)
    self._laplacian_matrix = None
    if laplacian_matrix is not None:
      self._laplacian_matrix = tf.convert_to_tensor(
          laplacian_matrix, dtype=tf.float32)
      # Check the validity of the laplacian matrix.
      tf.debugging.assert_near(
          0.0, tf.norm(tf.reduce_sum(self._laplacian_matrix, 1)))
      tf.debugging.assert_near(
          0.0, tf.norm(tf.reduce_sum(self._laplacian_matrix, 0)))
    self._laplacian_smoothing_weight = laplacian_smoothing_weight

    policy = greedy_reward_policy.GreedyRewardPredictionPolicy(
        time_step_spec,
        action_spec,
        reward_network,
        observation_and_action_constraint_splitter,
        constraints=constraints,
        accepts_per_arm_features=accepts_per_arm_features,
        emit_policy_info=emit_policy_info)
    training_data_spec = None
    if accepts_per_arm_features:
      training_data_spec = bandit_spec_utils.drop_arm_observation(
          policy.trajectory_spec, observation_and_action_constraint_splitter)

    super(GreedyRewardPredictionAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy=policy,
        train_sequence_length=None,
        training_data_spec=training_data_spec,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        train_step_counter=train_step_counter)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def _variables_to_train(self):
    variables_to_train = self._reward_network.variables
    for c in self._constraints:
      variables_to_train.extend(c.variables)
    return variables_to_train

  def _train(self, experience, weights):
    (observations, actions,
     rewards) = bandit_utils.process_experience_for_neural_agents(
         experience, self._observation_and_action_constraint_splitter,
         self._accepts_per_arm_features, self.training_data_spec)

    with tf.GradientTape() as tape:
      loss_info = self.loss(observations,
                            actions,
                            rewards,
                            weights=weights,
                            training=True)

    variables_to_train = self._variables_to_train()
    if not variables_to_train:
      logging.info('No variable to train in the agent.')
      return loss_info

    grads = tape.gradient(loss_info.loss, variables_to_train)
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(grads, variables_to_train))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)

    return loss_info

  def reward_loss(self,
                  observations,
                  actions,
                  rewards,
                  weights=None,
                  training=False):
    """Computes loss for reward prediction training.

    Args:
      observations: A batch of observations.
      actions: A batch of actions.
      rewards: A batch of rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `Tensor` containing the loss for the training step.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    with tf.name_scope('loss'):
      sample_weights = weights if weights is not None else 1
      if self._heteroscedastic:
        predictions, _ = self._reward_network(observations,
                                              training=training)
        predicted_values = predictions.q_value_logits
        predicted_log_variance = predictions.log_variance
        action_predicted_log_variance = common.index_with_actions(
            predicted_log_variance, tf.cast(actions, dtype=tf.int32))
        sample_weights = sample_weights * 0.5 * tf.exp(
            -action_predicted_log_variance)

        loss = 0.5 * tf.reduce_mean(action_predicted_log_variance)
        # loss = 1/(2 * var(x)) * (y - f(x))^2 + 1/2 * log var(x)
        # Kendall, Alex, and Yarin Gal. "What Uncertainties Do We Need in
        # Bayesian Deep Learning for Computer Vision?." Advances in Neural
        # Information Processing Systems. 2017. https://arxiv.org/abs/1703.04977
      else:
        predicted_values, _ = self._reward_network(observations,
                                                   training=training)
        loss = tf.constant(0.0)

      action_predicted_values = common.index_with_actions(
          predicted_values,
          tf.cast(actions, dtype=tf.int32))

      # Apply Laplacian smoothing on the estimated rewards, if applicable.
      if self._laplacian_matrix is not None:
        smoothness_batched = tf.matmul(
            predicted_values,
            tf.matmul(self._laplacian_matrix, predicted_values,
                      transpose_b=True))
        loss += (self._laplacian_smoothing_weight * tf.reduce_mean(
            tf.linalg.tensor_diag_part(smoothness_batched) * sample_weights))

      loss += self._error_loss_fn(
          rewards,
          action_predicted_values,
          sample_weights,
          reduction=tf.compat.v1.losses.Reduction.MEAN)

    return loss

  def loss(self,
           observations,
           actions,
           rewards,
           weights=None,
           training=False):
    """Computes loss for training the reward and constraint networks.

    Args:
      observations: A batch of observations.
      actions: A batch of actions.
      rewards: A batch of rewards. In the case we have constraints, we assume
        that reward is a dict of tensors with 'reward' and 'constraint' keys
        defined in 'bandit_spec_utils'. In case of many constraint signals, the
        constraint tensor has many columns; one column per constraint signal.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `LossInfo` containing the loss for the training step.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    if self._constraints:
      rewards_tensor = rewards[bandit_spec_utils.REWARD_SPEC_KEY]
    else:
      rewards_tensor = rewards
    reward_loss = self.reward_loss(
        observations, actions, rewards_tensor, weights, training)

    constraint_loss = tf.constant(0.0)
    for i, c in enumerate(self._constraints, 0):
      if self._time_step_spec.reward[
          bandit_spec_utils.CONSTRAINTS_SPEC_KEY].shape.rank > 1:
        constraint_targets = rewards[
            bandit_spec_utils.CONSTRAINTS_SPEC_KEY][:, i]
      else:
        constraint_targets = rewards[bandit_spec_utils.CONSTRAINTS_SPEC_KEY]
      constraint_loss += c.compute_loss(
          observations, actions, constraint_targets, weights, training)

    self.compute_summaries(reward_loss, constraint_loss=(
        constraint_loss if self._constraints else None))

    total_loss = reward_loss
    if self._constraints:
      total_loss += constraint_loss
    return tf_agent.LossInfo(total_loss, extra=())

  def compute_summaries(self, loss, constraint_loss=None):
    if self.summaries_enabled:
      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='loss', data=loss, step=self.train_step_counter)
        if constraint_loss is not None:
          tf.compat.v2.summary.scalar(
              name='constraint_loss',
              data=constraint_loss,
              step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._variables_to_train():
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)
