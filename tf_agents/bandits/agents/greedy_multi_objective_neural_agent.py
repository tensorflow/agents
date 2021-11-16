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

"""An agent that uses and trains a greedy reward prediction policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Optional, Sequence, Text, Tuple

from absl import logging
import gin
import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.multi_objective import multi_objective_scalarizer
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import greedy_multi_objective_neural_policy as greedy_multi_objective_policy
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks.network import Network
from tf_agents.policies import utils as policy_utilities
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils


@gin.configurable
class GreedyMultiObjectiveNeuralAgent(tf_agent.TFAgent):
  """A neural-network based bandit agent for multi-objective optimization.

  This agent receives multiple neural networks. Each network will be trained by
  the agent to predict a specific objective. The agent also receives a
  Scalarizer, which transforms multiple predicted objectives to a single reward.
  The action is chosen greedily by the policy with respect to the scalarized
  predicted reward.
  """

  def __init__(
      self,
      time_step_spec: Optional[ts.TimeStep],
      action_spec: Optional[types.NestedBoundedTensorSpec],
      scalarizer: multi_objective_scalarizer.Scalarizer,
      objective_network_and_loss_fn_sequence: Sequence[Tuple[Network, Callable[
          ..., tf.Tensor]]],
      optimizer: tf.keras.optimizers.Optimizer,
      observation_and_action_constraint_splitter: types.Splitter = None,
      accepts_per_arm_features: bool = False,
      # Params for training.
      gradient_clipping: Optional[float] = None,
      # Params for debugging.
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      enable_summaries: bool = True,
      emit_policy_info: Tuple[Text, ...] = (),
      train_step_counter: Optional[tf.Variable] = None,
      name: Optional[Text] = None):
    """Creates a Greedy Multi-objective Neural Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      scalarizer: A
       `tf_agents.bandits.multi_objective.multi_objective_scalarizer.Scalarizer`
        object that implements scalarization of multiple objectives into a
        single scalar reward.
      objective_network_and_loss_fn_sequence: A Sequence of Tuples
        (`tf_agents.network.Network`, error loss function) to be used by the
        agent. Each network `net` will be called as
        `net(observation, training=...)` and is expected to output a
        `tf.Tensor` of predicted values for a specific objective for all
        actions, shaped as [batch-size, number-of-actions]. Each network will be
        trained via minimizing the accompanying error loss function, which takes
        parameters labels, predictions, and weights (any function from tf.losses
        would work).
      optimizer: A 'tf.keras.optimizers.Optimizer' object, the optimizer to use
        for training.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask of shape `[batch_size, num_actions]`.
        This function should also work with a `TensorSpec` as input, and should
        output `TensorSpec` objects for the observation and mask.
      accepts_per_arm_features: (bool) Whether the agent accepts per-arm
        features.
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
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.

    Raises:
      ValueError:
        - If the action spec contains more than one action or or it is not a
          bounded scalar int32 spec with minimum 0.
        - If the length of `objective_network_and_loss_fn_sequence` is less than
          two.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)
    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)
    self._accepts_per_arm_features = accepts_per_arm_features

    self._num_objectives = len(objective_network_and_loss_fn_sequence)
    if self._num_objectives < 2:
      raise ValueError(
          'Number of objectives should be at least two, but found to be {}'
          .format(self._num_objectives))
    self._objective_networks, self._error_loss_fns = tuple(
        zip(*objective_network_and_loss_fn_sequence))
    self._optimizer = optimizer
    self._gradient_clipping = gradient_clipping
    self._heteroscedastic = [
        isinstance(network, heteroscedastic_q_network.HeteroscedasticQNetwork)
        for network in self._objective_networks
    ]

    policy = greedy_multi_objective_policy.GreedyMultiObjectiveNeuralPolicy(
        time_step_spec,
        action_spec,
        scalarizer,
        self._objective_networks,
        observation_and_action_constraint_splitter,
        accepts_per_arm_features=accepts_per_arm_features,
        emit_policy_info=emit_policy_info)
    training_data_spec = None
    if accepts_per_arm_features:
      training_data_spec = bandit_spec_utils.drop_arm_observation(
          policy.trajectory_spec)

    super(GreedyMultiObjectiveNeuralAgent, self).__init__(
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
    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def _variables_to_train(self):
    variables_to_train = tf.nest.flatten(
        [net.trainable_variables for net in self._objective_networks])
    return variables_to_train

  def _train(self, experience: types.NestedTensor,
             weights: types.Tensor) -> tf_agent.LossInfo:
    experience = self._as_trajectory(experience)

    with tf.GradientTape() as tape:
      loss_info = self._loss(
          experience,
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

  def _single_objective_loss(self,
                             objective_idx: int,
                             observations: tf.Tensor,
                             actions: tf.Tensor,
                             single_objective_values: tf.Tensor,
                             weights: types.Tensor = None,
                             training: bool = False) -> tf.Tensor:
    """Computes loss for a single objective.

    Args:
      objective_idx: The index into `self._objective_networks` for a specific
        objective network.
      observations: A batch of observations.
      actions: A batch of actions.
      single_objective_values: A batch of objective values shaped as
        [batch_size] for the objective that the network indexed by
        `objective_idx` is predicting.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and the
        final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `Tensor` containing the loss for the training step.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    if objective_idx >= self._num_objectives or objective_idx < 0:
      raise ValueError(
          'objective_idx should be between 0 and {}, but is {}'.format(
              self._num_objectives, objective_idx))
    with tf.name_scope('loss_for_objective_{}'.format(objective_idx)):
      objective_network = self._objective_networks[objective_idx]
      sample_weights = weights if weights is not None else 1
      if self._heteroscedastic[objective_idx]:
        predictions, _ = objective_network(observations, training=training)
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
        predicted_values, _ = objective_network(observations, training=training)
        loss = tf.constant(0.0)

      action_predicted_values = common.index_with_actions(
          predicted_values,
          tf.cast(actions, dtype=tf.int32))

      # Reduction is done outside of the loss function because non-scalar
      # weights with unknown shapes may trigger shape validation that fails
      # XLA compilation.
      loss += tf.reduce_mean(
          tf.multiply(
              self._error_loss_fns[objective_idx](
                  single_objective_values,
                  action_predicted_values,
                  reduction=tf.compat.v1.losses.Reduction.NONE),
              sample_weights))

    return loss

  def _loss(self,
            experience: types.NestedTensor,
            weights: types.Tensor = None,
            training: bool = False) -> tf_agent.LossInfo:
    """Computes loss for training the objective networks.

    Args:
      experience: A batch of experience data in the form of a `Trajectory` or
        `Transition`.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and the
        final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `LossInfo` containing the loss for the training step.
    Raises:
      ValueError:
        - If the number of actions is greater than 1.
        - If `objectives` is not rank-2.
        - If the number of columns in `objectives` does not equal
          `self._num_objectives`.
    """
    (observations, actions,
     objective_values) = bandit_utils.process_experience_for_neural_agents(
         experience, self._accepts_per_arm_features, self.training_data_spec)
    if self._observation_and_action_constraint_splitter is not None:
      observations, _ = self._observation_and_action_constraint_splitter(
          observations)
    if objective_values.shape.rank != 2:
      raise ValueError(
          'The objectives tensor should be rank-2 [batch_size, num_objectives],'
          ' but found to be rank-{}'.format(objective_values.shape.rank))
    if objective_values.shape[1] != self._num_objectives:
      raise ValueError(
          'The number of objectives in the objective_values tensor: {} '
          'is different from the number of objective networks: {}.'.format(
              objective_values.shape[1], self._num_objectives))

    objective_losses = []
    for idx in range(self._num_objectives):
      single_objective_values = objective_values[:, idx]
      objective_losses.append(
          self._single_objective_loss(idx, observations, actions,
                                      single_objective_values, weights,
                                      training))

    self.compute_summaries(objective_losses)
    total_loss = tf.reduce_sum(objective_losses)
    return tf_agent.LossInfo(total_loss, extra=())

  def compute_summaries(self, losses: Sequence[tf.Tensor]):
    if self._num_objectives != len(losses):
      raise ValueError('The number of losses: {} does not equal the number '
                       'of objectives: {}'.format(
                           len(losses), self._num_objectives))
    if self.summaries_enabled:
      with tf.name_scope('Losses/'):
        for idx in range(self._num_objectives):
          name_of_loss = self._objective_networks[idx].name
          if not name_of_loss:
            name_of_loss = 'loss_{}'.format(idx)
          tf.compat.v2.summary.scalar(
              name=name_of_loss, data=losses[idx], step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._variables_to_train():
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)
