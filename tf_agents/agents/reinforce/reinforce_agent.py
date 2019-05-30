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

"""A REINFORCE Agent.

Implements the REINFORCE algorithm from (Williams, 1992):
http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import value_ops


def _standard_normalize(values, axes=(0,)):
  """Standard normalizes values `values`.

  Args:
    values: Tensor with values to be standardized.
    axes: Axes used to compute mean and variances.

  Returns:
    Standardized values (values - mean(values[axes])) / std(values[axes]).
  """
  values_mean, values_var = tf.nn.moments(x=values, axes=axes, keepdims=True)
  epsilon = np.finfo(values.dtype.as_numpy_dtype).eps
  normalized_values = (
      (values - values_mean) / (tf.sqrt(values_var) + epsilon))
  return normalized_values


def _entropy_loss(distributions, spec, weights=None):
  """Computes entropy loss.

  Args:
    distributions: A possibly batched tuple of distributions.
    spec: A nested tuple representing the action spec.
    weights: Optional scalar or element-wise (per-batch-entry) importance
      weights.  Includes a mask for invalid timesteps.

  Returns:
    A Tensor representing the entropy loss.
  """
  with tf.name_scope('entropy_regularization'):
    entropy = -tf.cast(common.entropy(distributions, spec), tf.float32)
    if weights is not None:
      entropy *= weights
    return tf.reduce_mean(input_tensor=entropy)


def _get_initial_policy_state(policy, time_steps):
  """Gets the initial state of a policy."""
  batch_size = (tf.compat.dimension_at_index(time_steps.discount.shape, 0) or
                tf.shape(time_steps.discount)[0])
  return policy.get_initial_state(batch_size=batch_size)


@gin.configurable
class ReinforceAgent(tf_agent.TFAgent):
  """A REINFORCE Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               optimizer,
               value_network=None,
               value_estimation_loss_coef=0.2,
               gamma=1.0,
               normalize_returns=True,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               entropy_regularization=None,
               train_step_counter=None,
               name=None):
    """Creates a REINFORCE Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type).
      optimizer: Optimizer for the actor network.
      value_network: (Optional) A `tf_agents.network.Network` to be used by the
        agent. The network will be called with call(observation, step_type) and
        returns a floating point value tensor.
      value_estimation_loss_coef: (Optional) Multiplier for value prediction
        loss to balance with policy gradient loss.
      gamma: A discount factor for future rewards.
      normalize_returns: Whether to normalize returns across episodes when
        computing the loss.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      entropy_regularization: Coefficient for entropy regularization loss term.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)

    self._actor_network = actor_network
    self._value_network = value_network

    collect_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        clip=True)

    policy = greedy_policy.GreedyPolicy(collect_policy)

    self._optimizer = optimizer
    self._gamma = gamma
    self._normalize_returns = normalize_returns
    self._gradient_clipping = gradient_clipping
    self._entropy_regularization = entropy_regularization
    self._value_estimation_loss_coef = value_estimation_loss_coef
    self._baseline = self._value_network is not None

    super(ReinforceAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _initialize(self):
    pass

  def _train(self, experience, weights=None):
    # Add a mask to ensure we reset the return calculation at episode
    # boundaries. This is needed in cases where episodes are truncated before
    # reaching a terminal state.
    non_last_mask = tf.cast(
        tf.math.not_equal(experience.next_step_type, ts.StepType.LAST),
        tf.float32)
    discounts = non_last_mask * experience.discount * self._gamma
    returns = value_ops.discounted_return(
        experience.reward, discounts, time_major=False)

    if self._debug_summaries:
      tf.compat.v2.summary.histogram(
          name='rewards', data=experience.reward, step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='discounts',
          data=experience.discount,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='returns', data=returns, step=self.train_step_counter)

    time_step = ts.TimeStep(experience.step_type,
                            tf.zeros_like(experience.reward),
                            tf.zeros_like(experience.discount),
                            experience.observation)

    with tf.GradientTape() as tape:
      loss_info = self.total_loss(time_step,
                                  experience.action,
                                  tf.stop_gradient(returns),
                                  weights=weights)
      tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
    variables_to_train = self._actor_network.trainable_weights
    if self._baseline:
      variables_to_train += self._value_network.trainable_weights
    grads = tape.gradient(loss_info.loss, variables_to_train)

    grads_and_vars = zip(grads, variables_to_train)
    if self._gradient_clipping:
      grads_and_vars = eager_utils.clip_gradient_norms(
          grads_and_vars, self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(
          grads_and_vars, self.train_step_counter)
      eager_utils.add_gradients_summaries(
          grads_and_vars, self.train_step_counter)

    self._optimizer.apply_gradients(
        grads_and_vars, global_step=self.train_step_counter)

    return tf.nest.map_structure(tf.identity, loss_info)

  def total_loss(self, time_steps, actions, returns, weights):
    # Ensure we see at least one full episode.
    is_last = time_steps.is_last()
    num_episodes = tf.reduce_sum(tf.cast(is_last, tf.float32))
    tf.debugging.assert_greater(
        num_episodes, 0.0,
        message='No complete episode found. REINFORCE requires full episodes '
        'to compute losses.')

    # Mask out partial episodes at the end of each batch of time_steps.
    valid_mask = tf.cast(is_last, dtype=tf.float32)
    valid_mask = tf.math.cumsum(valid_mask, axis=1, reverse=True)
    valid_mask = tf.cast(valid_mask > 0, dtype=tf.float32)
    if weights is not None:
      weights *= valid_mask
    else:
      weights = valid_mask

    advantages = returns
    if self._baseline:
      value_preds, _ = self._value_network(
          time_steps.observation, time_steps.step_type)
      advantages = returns - value_preds
      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='value_preds', data=value_preds, step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='advantages', data=advantages, step=self.train_step_counter)

    # TODO(b/126592060): replace with tensor normalizer.
    if self._normalize_returns:
      advantages = _standard_normalize(advantages, axes=(0, 1))
      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='normalized_%s'%'advantages' if self._baseline else 'returns',
            data=advantages,
            step=self.train_step_counter)

    tf.nest.assert_same_structure(time_steps, self.time_step_spec)
    policy_state = _get_initial_policy_state(self.collect_policy, time_steps)
    actions_distribution = self.collect_policy.distribution(
        time_steps, policy_state=policy_state).action

    policy_gradient_loss = self.policy_gradient_loss(actions_distribution,
                                                     actions,
                                                     is_last,
                                                     advantages,
                                                     num_episodes,
                                                     weights)
    entropy_regularization_loss = self.entropy_regularization_loss(
        actions_distribution, weights)

    total_loss = policy_gradient_loss + entropy_regularization_loss

    if self._baseline:
      value_estimation_loss = self.value_estimation_loss(
          value_preds, returns, num_episodes, weights)
      total_loss += value_estimation_loss

    with tf.name_scope('Losses/'):
      tf.compat.v2.summary.scalar(
          name='policy_gradient_loss',
          data=policy_gradient_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='entropy_regularization_loss',
          data=entropy_regularization_loss,
          step=self.train_step_counter)
      if self._baseline:
        tf.compat.v2.summary.scalar(
            name='value_estimation_loss',
            data=value_estimation_loss,
            step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='total_loss', data=total_loss, step=self.train_step_counter)

    return tf_agent.LossInfo(total_loss, ())

  def policy_gradient_loss(self, actions_distribution, actions, is_last,
                           returns, num_episodes, weights=None):
    """Computes the policy gradient loss.

    Args:
      actions_distribution: A possibly batched tuple of action distributions.
      actions: Tensor with a batch of actions.
      is_last: Tensor of booleans that indicate if the end of the trajectory
        has been reached.
      returns: Tensor with a return from each timestep, aligned on index. Works
        better when returns are normalized.
      num_episodes: Number of episodes contained in the training data.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  May include a mask for invalid timesteps.

    Returns:
      policy_gradient_loss: A tensor that will contain policy gradient loss for
        the on-policy experience.
    """
    # TODO(b/126594799): Add class IndependentNested(tfd.Distribution) to handle
    # nests of independent distributions like this.
    action_log_prob = common.log_probability(actions_distribution, actions,
                                             self.action_spec)

    # Filter out transitions between end state of previous episode and start
    # state of next episode.
    valid_mask = tf.cast(~is_last, tf.float32)
    action_log_prob *= valid_mask

    action_log_prob_times_return = action_log_prob * returns

    if weights is not None:
      action_log_prob_times_return *= weights

    if self._debug_summaries:
      tf.compat.v2.summary.histogram(
          name='action_log_prob',
          data=action_log_prob,
          step=self.train_step_counter)
      tf.compat.v2.summary.histogram(
          name='action_log_prob_times_return',
          data=action_log_prob_times_return,
          step=self.train_step_counter)

    # Policy gradient loss is defined as the sum, over timesteps, of action
    #   log-probability times the cumulative return from that timestep onward.
    #   For more information, see (Williams, 1992).
    policy_gradient_loss = -tf.reduce_sum(
        input_tensor=action_log_prob_times_return)

    # We take the mean over episodes by dividing by num_episodes.
    policy_gradient_loss = policy_gradient_loss / num_episodes

    return policy_gradient_loss

  def entropy_regularization_loss(self, actions_distribution, weights=None):
    """Computes the optional entropy regularization loss.

    Extending REINFORCE by entropy regularization was originally proposed in
    "Function optimization using connectionist reinforcement learning
    algorithms." (Williams and Peng, 1991).

    Args:
      actions_distribution: A possibly batched tuple of action distributions.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  May include a mask for invalid timesteps.

    Returns:
      entropy_regularization_loss: A tensor with the entropy regularization
      loss.
    """
    if self._entropy_regularization:
      loss = _entropy_loss(
          actions_distribution, self.action_spec, weights)
      loss *= self._entropy_regularization
    else:
      loss = tf.constant(0.0, dtype=tf.float32)

    return loss

  def value_estimation_loss(
      self, value_preds, returns, num_episodes, weights=None):
    """Computes the value estimation loss.

    Args:
      value_preds: Per-timestep estimated values.
      returns: Per-timestep returns for value function to predict.
      num_episodes: Number of episodes contained in the training data.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  May include a mask for invalid timesteps.

    Returns:
      value_estimation_loss: A scalar value_estimation_loss loss.
    """
    value_estimation_error = tf.math.squared_difference(returns, value_preds)
    if weights is not None:
      value_estimation_error *= weights

    value_estimation_loss = (
        tf.reduce_sum(input_tensor=value_estimation_error) *
        self._value_estimation_loss_coef)

    # We take the mean over episodes by dividing by num_episodes.
    value_estimation_loss = value_estimation_loss / num_episodes

    return value_estimation_loss
