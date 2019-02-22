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

"""A Soft Actor-Critic Agent.

Implements the Soft Actor-Critic (SAC) algorithm from
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Soft Actor" by Haarnoja et al (2017).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.distributions import tanh_bijector_stable
from tf_agents.environments import trajectory
from tf_agents.policies import actor_policy
from tf_agents.utils import common
from tf_agents.utils import eager_utils

import gin.tf


@gin.configurable
def std_clip_transform(stddevs):
  stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 2),
                                  stddevs)
  return tf.exp(stddevs)


@gin.configurable
class SacAgent(tf_agent.TFAgent):
  """A SAC Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               critic_network,
               actor_network,
               actor_optimizer,
               critic_optimizer,
               alpha_optimizer,
               actor_policy_ctor=actor_policy.ActorPolicy,
               squash_actions=True,
               target_update_tau=1.0,
               target_update_period=1,
               td_errors_loss_fn=tf.math.squared_difference,
               gamma=1.0,
               reward_scale_factor=1.0,
               initial_log_alpha=0.0,
               target_entropy=None,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               name=None):
    """Creates a SAC Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      critic_network: A function critic_network((observations, actions)) that
        returns the q_values for each observation and action.
      actor_network: A function actor_network(observation, action_spec) that
       returns action distribution.
      actor_optimizer: The optimizer to use for the actor network.
      critic_optimizer: The default optimizer to use for the critic network.
      alpha_optimizer: The default optimizer to use for the alpha variable.
      actor_policy_ctor: The policy class to use.
      squash_actions: Whether or not to use tanh to squash actions between
        -1 and 1.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      td_errors_loss_fn:  A function for computing the elementwise TD errors
        loss.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      initial_log_alpha: Initial value for log_alpha.
      target_entropy: The target average policy entropy, for updating alpha.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)

    self._critic_network1 = critic_network
    self._critic_network2 = critic_network.copy(
        name='CriticNetwork2')
    self._target_critic_network1 = critic_network.copy(
        name='TargetCriticNetwork1')
    self._target_critic_network2 = critic_network.copy(
        name='TargetCriticNetwork2')
    self._actor_network = actor_network

    policy = actor_policy_ctor(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network)

    self._log_alpha = common.create_variable(
        'initial_log_alpha',
        initial_value=initial_log_alpha,
        dtype=tf.float32,
        trainable=True)

    # If target_entropy was not passed, set it to negative of the total number
    # of action dimensions.
    if target_entropy is None:
      flat_action_spec = tf.nest.flatten(action_spec)
      target_entropy = -np.sum([np.product(single_spec.shape.as_list())
                                for single_spec in flat_action_spec])

    self._squash_actions = squash_actions
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer
    self._alpha_optimizer = alpha_optimizer
    self._td_errors_loss_fn = td_errors_loss_fn
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_entropy = target_entropy
    self._gradient_clipping = gradient_clipping
    self._debug_summaries = debug_summaries
    self._summarize_grads_and_vars = summarize_grads_and_vars

    super(SacAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=2,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars)

  def _initialize(self):
    """Returns an op to initialize the agent.

    Copies weights from the Q networks to the target Q network.

    Returns:
      An op to initialize the agent.
    """
    return self._update_targets(1.0, 1)

  def _experience_to_transitions(self, experience):
    transitions = trajectory.to_transition(experience)
    time_steps, policy_steps, next_time_steps = transitions
    actions = policy_steps.action
    # TODO(eholly): Figure out how to properly deal with time dimension.
    time_steps, actions, next_time_steps = tf.nest.map_structure(
        lambda t: tf.squeeze(t, axis=1), (time_steps, actions, next_time_steps))
    return time_steps, actions, next_time_steps

  def _train(self, experience, weights, train_step_counter):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      train_step_counter: An optional counter to increment every time the train
        op is run. Typically the global_step.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    time_steps, actions, next_time_steps = self._experience_to_transitions(
        experience)

    critic_loss = self.critic_loss(
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn=self._td_errors_loss_fn,
        gamma=self._gamma,
        reward_scale_factor=self._reward_scale_factor,
        weights=weights)

    actor_loss = self.actor_loss(time_steps, weights=weights)

    alpha_loss = self.alpha_loss(time_steps, weights=weights)

    def clip_and_summarize_gradients(grads_and_vars):
      """Clips gradients, and summarizes gradients and variables."""
      if self._gradient_clipping is not None:
        grads_and_vars = eager_utils.clip_gradient_norms_fn(
            self._gradient_clipping)(
                grads_and_vars)

      if self._summarize_grads_and_vars:
        # TODO(kbanoop): Move gradient summaries to train_op after we switch to
        # eager train op, and move variable summaries to critic_loss.
        for grad, var in grads_and_vars:
          with tf.name_scope('Gradients/'):
            if grad is not None:
              tf.contrib.summary.histogram(grad.op.name, grad)
          with tf.name_scope('Variables/'):
            if var is not None:
              tf.contrib.summary.histogram(var.op.name, var)
      return grads_and_vars

    with tf.name_scope('Losses'):
      tf.contrib.summary.scalar('critic_loss', critic_loss)
      tf.contrib.summary.scalar('actor_loss', actor_loss)
      tf.contrib.summary.scalar('alpha_loss', alpha_loss)

    critic_train_op = eager_utils.create_train_op(
        critic_loss,
        self._critic_optimizer,
        global_step=train_step_counter,
        transform_grads_fn=clip_and_summarize_gradients,
        variables_to_train=(self._critic_network1.trainable_weights +
                            self._critic_network2.trainable_weights),
    )

    actor_train_op = eager_utils.create_train_op(
        actor_loss,
        self._actor_optimizer,
        global_step=None,
        transform_grads_fn=clip_and_summarize_gradients,
        variables_to_train=self._actor_network.trainable_weights,
    )

    alpha_train_op = eager_utils.create_train_op(
        alpha_loss,
        self._alpha_optimizer,
        global_step=None,
        transform_grads_fn=clip_and_summarize_gradients,
        variables_to_train=[self._log_alpha],
    )

    with tf.control_dependencies([critic_train_op,
                                  actor_train_op,
                                  alpha_train_op]):
      update_targets_op = self._update_targets(
          tau=self._target_update_tau, period=self._target_update_period)

    with tf.control_dependencies([update_targets_op]):
      train_op = (
          critic_train_op + actor_train_op + alpha_train_op)

    return tf_agent.LossInfo(loss=train_op, extra=tf.no_op())

  def _update_targets(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.
    Returns:
      An operation that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_target'):
      def update():
        """Update target network."""
        critic_update_1 = common.soft_variables_update(
            self._critic_network1.variables,
            self._target_critic_network1.variables, tau)
        critic_update_2 = common.soft_variables_update(
            self._critic_network2.variables,
            self._target_critic_network2.variables, tau)
        return tf.group(critic_update_1, critic_update_2)

      return common.periodically(update, period, 'update_targets')

  def _action_spec_means_magnitudes(self):
    """Get the center and magnitude of the ranges in action spec."""
    action_spec = self.action_spec
    action_means = tf.nest.map_structure(
        lambda spec: (spec.maximum + spec.minimum) / 2.0, action_spec)
    action_magnitudes = tf.nest.map_structure(
        lambda spec: (spec.maximum - spec.minimum) / 2.0, action_spec)
    return tf.cast(
        action_means, dtype=tf.float32), tf.cast(
            action_magnitudes, dtype=tf.float32)

  def _actions_and_log_probs(self, time_steps):
    """Get actions and corresponding log probabilities from policy."""
    # Get raw action distribution from policy, and initialize bijectors list.
    action_distribution = self.policy.distribution(time_steps).action

    if self._squash_actions:
      bijectors = []

      # Bijector to rescale actions to ranges in action spec.
      action_means, action_magnitudes = self._action_spec_means_magnitudes()
      bijectors.append(tfp.bijectors.AffineScalar(
          shift=action_means, scale=action_magnitudes))

      # Bijector to squash actions to range (-1.0, +1.0).
      bijectors.append(tanh_bijector_stable.Tanh())

      # Chain applies bijectors in reverse order, so squash will happen before
      # rescaling to action spec.
      bijector_chain = tfp.bijectors.Chain(bijectors)
      action_distribution = tfp.distributions.TransformedDistribution(
          distribution=action_distribution, bijector=bijector_chain)

    # Sample actions and log_pis from transformed distribution.
    actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
    log_pi = common.log_probability(action_distribution, actions,
                                    self.action_spec)

    return actions, log_pi

  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  td_errors_loss_fn,
                  gamma=1.0,
                  reward_scale_factor=1.0,
                  weights=None):
    """Computes the critic loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute
        elementwise (per-batch-entry) loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      tf.nest.assert_same_structure(actions, self.action_spec)
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)
      tf.nest.assert_same_structure(next_time_steps, self.time_step_spec)

      next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
      target_input_1 = (next_time_steps.observation, next_actions)
      target_q_values1, unused_network_state1 = self._target_critic_network1(
          target_input_1, next_time_steps.step_type)
      target_input_2 = (next_time_steps.observation, next_actions)
      target_q_values2, unused_network_state2 = self._target_critic_network2(
          target_input_2, next_time_steps.step_type)
      target_q_values = (tf.minimum(target_q_values1, target_q_values2) -
                         tf.exp(self._log_alpha) * next_log_pis)

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_input_1 = (time_steps.observation, actions)
      pred_td_targets1, unused_network_state1 = self._critic_network1(
          pred_input_1, time_steps.step_type)
      pred_input_2 = (time_steps.observation, actions)
      pred_td_targets2, unused_network_state2 = self._critic_network2(
          pred_input_2, time_steps.step_type)
      critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
      critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
      critic_loss = critic_loss1 + critic_loss2

      if weights is not None:
        critic_loss *= weights

      # Take the mean across the batch.
      critic_loss = tf.reduce_mean(input_tensor=critic_loss)

      if self._debug_summaries:
        td_errors1 = td_targets - pred_td_targets1
        td_errors2 = td_targets - pred_td_targets2
        td_errors = tf.concat([td_errors1, td_errors2], axis=0)
        common.generate_tensor_summaries('td_errors', td_errors)
        common.generate_tensor_summaries('td_targets', td_targets)
        common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1)
        common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2)

      return critic_loss

  def actor_loss(self, time_steps, weights=None):
    """Computes the actor_loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)

      actions, log_pi = self._actions_and_log_probs(time_steps)
      target_input_1 = (time_steps.observation, actions)
      target_q_values1, unused_network_state1 = self._critic_network1(
          target_input_1, time_steps.step_type)
      target_input_2 = (time_steps.observation, actions)
      target_q_values2, unused_network_state2 = self._critic_network2(
          target_input_2, time_steps.step_type)
      target_q_values = tf.minimum(target_q_values1, target_q_values2)
      actor_loss = tf.exp(self._log_alpha) * log_pi - target_q_values
      if weights is not None:
        actor_loss *= weights
      actor_loss = tf.reduce_mean(input_tensor=actor_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('actor_loss', actor_loss)
        common.generate_tensor_summaries('actions', actions)
        common.generate_tensor_summaries('log_pi', log_pi)
        tf.contrib.summary.scalar('entropy_avg',
                                  -tf.reduce_mean(input_tensor=log_pi))
        common.generate_tensor_summaries('target_q_values', target_q_values)
        action_distribution = self.policy.distribution(time_steps).action
        common.generate_tensor_summaries('act_mean', action_distribution.loc)
        common.generate_tensor_summaries('act_stddev',
                                         action_distribution.scale)
        common.generate_tensor_summaries('entropy_raw_action',
                                         action_distribution.entropy())

      return actor_loss

  def alpha_loss(self, time_steps, weights=None):
    """Computes the alpha_loss for EC-SAC training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      alpha_loss: A scalar alpha loss.
    """
    with tf.name_scope('alpha_loss'):
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)

      unused_actions, log_pi = self._actions_and_log_probs(time_steps)
      alpha_loss = (
          self._log_alpha *
          tf.stop_gradient(-log_pi - self._target_entropy))

      if weights is not None:
        alpha_loss *= weights

      alpha_loss = tf.reduce_mean(input_tensor=alpha_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('alpha_loss', alpha_loss)

      return alpha_loss
