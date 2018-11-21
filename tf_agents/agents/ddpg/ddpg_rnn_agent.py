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

"""A DDPG agent that supports RNN networks.

Implements the Deep Deterministic Policy Gradient (DDPG) algorithm from
"Continuous control with deep reinforcement learning" - Lilicrap et al.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.ddpg import rnn_networks
from tf_agents.environments import time_step as ts
from tf_agents.policies import actor_rnn_policy
from tf_agents.policies import ou_noise_policy
import tf_agents.utils.common as common_utils
import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class DdpgRnnAgent(ddpg_agent.DdpgAgent):
  """A DDPG RNN Agent."""

  ACTOR_NET_SCOPE = 'actor_net'
  TARGET_ACTOR_NET_SCOPE = 'target_actor_net'
  CRITIC_NET_SCOPE = 'critic_net'
  TARGET_CRITIC_NET_SCOPE = 'target_critic_net'

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_net=rnn_networks.actor_network,
               critic_net=rnn_networks.critic_network,
               replay_buffer_ctor=None,
               ou_stddev=1.0,
               ou_damping=1.0,
               target_update_tau=1.0,
               target_update_period=1,
               actor_optimizer=None,
               critic_optimizer=None,
               train_batch_size=32,
               train_sequence_length=32,
               dqda_clipping=None,
               td_errors_loss_fn=None,
               gamma=1.0,
               reward_scale_factor=1.0,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False):
    """Creates a DDPG Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_net: A function actor_net(observation, action_spec) that returns the
        actions for each observation.
      critic_net: A function critic_net(observations, actions) that returns the
        q_values for each observation and action.
      replay_buffer_ctor: A function to construct the replay buffer. Default is
        None, in which case we only allow training from an external replay
        buffer. In most cases, this argument has to be provided.
      ou_stddev: Standard deviation for the Ornstein-Uhlenbeck (OU) noise added
        in the default collect policy.
      ou_damping: Damping factor for the OU noise added in the default collect
        policy.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      actor_optimizer: The default optimizer to use for the actor network.
      critic_optimizer: The default optimizer to use for the critic network.
      train_batch_size: An integer batch_size for training.
      train_sequence_length: An integer indicating how long of a sequence to
        sample from the replay buffer for training.
      dqda_clipping: when computing the actor loss, clips the gradient dqda
        element-wise between [-dqda_clipping, dqda_clipping]. Does not perform
        clipping if dqda_clipping == 0.
      td_errors_loss_fn:  A function for computing the TD errors loss. If None,
        a default value of tf.losses.huber_loss is used.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
    """
    super(DdpgRnnAgent, self).__init__(
        time_step_spec,
        action_spec,
        actor_net=actor_net,
        critic_net=critic_net,
        replay_buffer_ctor=replay_buffer_ctor,
        ou_stddev=ou_stddev,
        ou_damping=ou_damping,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        train_batch_size=train_batch_size,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars)
    self._train_sequence_length = train_sequence_length
    self._policy_state_spec = rnn_networks.get_state_spec()
    self._policy = actor_rnn_policy.ActorRnnPolicy(
        time_step_spec=self.time_step_spec(),
        action_spec=self.action_spec(),
        policy_state_spec=self._policy_state_spec,
        actor_network=self._actor_net,
        clip=True)

  def collect_policy(self, policy_wrapper=None):
    """Returns a policy for collecting experience in the environment.

    Args:
      policy_wrapper: A wrapper(policy) that modifies the agent's actor policy.
        By default, we use a wrapper that adds Ornstein-Uhlenbeck (OU) noise
        to the unclipped actions from the agent's actor network.

    Returns:
      A subclass of tf_policy.Base for collecting data.
    """
    wrapped_policy = actor_rnn_policy.ActorRnnPolicy(
        time_step_spec=self.time_step_spec(),
        action_spec=self.action_spec(),
        policy_state_spec=self._policy_state_spec,
        actor_network=self._actor_net,
        clip=False)
    policy_wrapper = policy_wrapper or functools.partial(
        ou_noise_policy.OUNoisePolicy,
        ou_stddev=self._ou_stddev,
        ou_damping=self._ou_damping,
        clip=True)
    return policy_wrapper(wrapped_policy)

  def train(self,
            replay_buffer=None,
            train_step_counter=None,
            actor_optimizer=None,
            critic_optimizer=None):
    """Returns a train op to update the agent's networks.

    Args:
      replay_buffer: An optional replay buffer containing [time_steps, actions,
        next_time_steps]. If None, the agent's replay buffer is used.
      train_step_counter: An optional counter to increment every time the train
        op is run. Typically the global_step.
      actor_optimizer: An optimizer to use for training the actor_network. If
        None, the optimizer provided in the constructor is used.
      critic_optimizer: An optimizer to use for training the critic network. If
        None, the optimizer provided in the constructor is used.

    Returns:
      A train_op.
    Raises:
      ValueError: If optimizer or replay_buffer are None, and a default value
        was not provided in the constructor.

    """
    replay_buffer = replay_buffer or self.replay_buffer
    actor_optimizer = actor_optimizer or self._actor_optimizer
    critic_optimizer = critic_optimizer or self._critic_optimizer

    if replay_buffer is None:
      raise ValueError('`replay_buffer` cannot be None.')
    if actor_optimizer is None:
      raise ValueError('`actor_optimizer` cannot be None.')
    if critic_optimizer is None:
      raise ValueError('`critic_optimizer` cannot be None.')

    batch, _ = replay_buffer.get_next(
        sample_batch_size=self._train_batch_size,
        num_steps=self._train_sequence_length + 1,
        time_stacked=True)

    # TODO(kbanoop):  use trajectory.to_transition(*batched, time_staked=True)
    actions = batch.action[:, :-1]
    time_steps = ts.TimeStep(
        batch.step_type[:, :-1],
        tf.zeros_like(batch.reward[:, :-1]),  # ignored
        tf.zeros_like(batch.discount[:, :-1]),  # ignored
        batch.observation[:, :-1])
    next_time_steps = ts.TimeStep(batch.next_step_type[:, :-1],
                                  batch.reward[:, :-1], batch.discount[:, :-1],
                                  batch.observation[:, 1:])

    experience = (time_steps, actions, next_time_steps)

    train_op = self.train_from_experience(
        experience,
        actor_optimizer,
        critic_optimizer,
        train_step_counter=train_step_counter)
    return train_op

  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  td_errors_loss_fn=tf.losses.huber_loss,
                  gamma=1.0,
                  reward_scale_factor=1.0):
    """Computes the critic loss for DDPG training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      nest.assert_same_structure(actions, self.action_spec())
      nest.assert_same_structure(time_steps, self.time_step_spec())
      nest.assert_same_structure(next_time_steps, self.time_step_spec())

      target_actions, _ = self._target_actor_net(next_time_steps,
                                                 self.action_spec())

      target_q_values, _ = self._target_critic_net(next_time_steps,
                                                   target_actions)
      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      q_values, _ = self._critic_net(time_steps, actions)
      critic_loss = td_errors_loss_fn(td_targets, q_values)
      with tf.name_scope('Losses/'):
        tf.contrib.summary.scalar('critic_loss', critic_loss)

      if self._debug_summaries:
        td_errors = td_targets - q_values
        common_utils.generate_tensor_summaries('td_errors', td_errors)
        common_utils.generate_tensor_summaries('td_targets', td_targets)
        common_utils.generate_tensor_summaries('q_values', q_values)

      return critic_loss

  def actor_loss(self, time_steps, dqda_clipping=None):
    """Computes the actor_loss for DDPG training.

    Args:
      time_steps: A batch of timesteps.
      dqda_clipping: (float) clips the gradient dqda element-wise between
        [-dqda_clipping, dqda_clipping]. Does not perform clipping if
        dqda_clipping is None.
      # TODO(kbanoop): Add an action norm regularizer.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      nest.assert_same_structure(time_steps, self.time_step_spec())
      actions, _ = self._actor_net(time_steps, self.action_spec())

      q_values, _ = self._critic_net(time_steps, actions)
      actions = nest.flatten(actions)
      dqda = tf.gradients([q_values], actions)
      actor_losses = []
      for dqda, action in zip(dqda, actions):
        if dqda_clipping is not None:
          dqda = tf.clip_by_value(dqda, -dqda_clipping, dqda_clipping)  # pylint: disable=invalid-unary-operand-type
        actor_losses.append(
            tf.losses.mean_squared_error(
                tf.stop_gradient(dqda + action), action))
      actor_loss = tf.add_n(actor_losses)
      with tf.name_scope('Losses/'):
        tf.contrib.summary.scalar('actor_loss', actor_loss)

    return actor_loss
