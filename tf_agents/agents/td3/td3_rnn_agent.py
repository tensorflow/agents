# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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

"""Twin Delayed Deep Deterministic policy gradient (TD3) agent with RNN support.

"Addressing Function Approximation Error in Actor-Critic Methods"
by Fujimoto et al.

For the full paper, see https://arxiv.org/abs/1802.09477.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from tf_agents.agents.ddpg import rnn_networks
from tf_agents.agents.td3 import td3_agent
from tf_agents.environments import time_step as ts
from tf_agents.policies import actor_rnn_policy
from tf_agents.policies import ou_noise_policy
import gin

nest = tf.contrib.framework.nest


@gin.configurable
class Td3RnnAgent(td3_agent.Td3Agent):
  """A TD3 Agent."""

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
               td_errors_loss_fn=tf.losses.huber_loss,
               gamma=1.0,
               reward_scale_factor=1.0,
               target_policy_noise=0.2,
               target_policy_noise_clip=0.5,
               debug_summaries=False):
    """Creates a Td3Agent Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_net: A function actor_net(observation, action_spec) that returns
        the actions for each observation
      critic_net: A function critic_net(observations, actions) that returns
        the q_values for each observation and action.
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
      dqda_clipping: A scalar or float clips the gradient dqda element-wise
        between [-dqda_clipping, dqda_clipping]. Default is None representing no
        clippiing.
      td_errors_loss_fn:  A function for computing the TD errors loss. If None,
        a default value of tf.losses.huber_loss is used.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      target_policy_noise: Scale factor on target action noise
      target_policy_noise_clip: Value to clip noise.
      debug_summaries: A bool to gather debug summaries.
    """
    super(Td3RnnAgent, self).__init__(
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
        target_policy_noise=target_policy_noise,
        target_policy_noise_clip=target_policy_noise_clip,
        debug_summaries=debug_summaries)

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
      replay_buffer: An optional replay buffer containing
        [time_steps, actions, next_time_steps]. If None, the agent's replay
        buffer is used.
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

    batch, _, _ = replay_buffer.get_next(
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
    train_op = self.train_from_experience(
        (time_steps, actions, next_time_steps),
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
                  target_policy_noise=0.2,
                  target_policy_noise_clip=0.5,
                  reward_scale_factor=1.0):
    """Computes the critic loss for TD3 training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute loss.
      gamma: Discount for future rewards.
      target_policy_noise: Scale factor on target action noise
      target_policy_noise_clip: Value to clip noise.
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

      # Add gaussian noise to each action before computing target q values
      def add_noise_to_action(action):  # pylint: disable=missing-docstring
        dist = tf.distributions.Normal(loc=tf.zeros_like(action),
                                       scale=target_policy_noise * \
                                       tf.ones_like(action))
        noise = dist.sample()
        noise = tf.clip_by_value(noise, -target_policy_noise_clip,
                                 target_policy_noise_clip)
        return action + noise

      noisy_target_actions = nest.map_structure(add_noise_to_action,
                                                target_actions)

      # Target q-values are the min of the two networks
      target_q_values_1, _ = self._target_critic_net_1(next_time_steps,
                                                       noisy_target_actions)
      target_q_values_2, _ = self._target_critic_net_2(next_time_steps,
                                                       noisy_target_actions)
      target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_td_targets_1, _ = self._critic_net_1(time_steps, actions)
      pred_td_targets_2, _ = self._critic_net_2(time_steps, actions)
      pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]

      if self._debug_summaries:
        tf.contrib.summary.histogram('td_targets', td_targets)
        with tf.name_scope('td_targets'):
          tf.contrib.summary.scalar('mean', tf.reduce_mean(td_targets))
          tf.contrib.summary.scalar('max', tf.reduce_max(td_targets))
          tf.contrib.summary.scalar('min', tf.reduce_min(td_targets))

        for td_target_idx in range(2):
          pred_td_targets = pred_td_targets_all[td_target_idx]
          td_errors = td_targets - pred_td_targets
          with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
            tf.contrib.summary.histogram('td_errors', td_errors)
            tf.contrib.summary.histogram('pred_td_targets', pred_td_targets)
            with tf.name_scope('td_errors'):
              tf.contrib.summary.scalar('mean', tf.reduce_mean(td_errors))
              tf.contrib.summary.scalar('mean_abs',
                                        tf.reduce_mean(tf.abs(td_errors)))
              tf.contrib.summary.scalar('max', tf.reduce_max(td_errors))
              tf.contrib.summary.scalar('min', tf.reduce_min(td_errors))
            with tf.name_scope('pred_td_targets'):
              tf.contrib.summary.scalar('mean', tf.reduce_mean(pred_td_targets))
              tf.contrib.summary.scalar('max', tf.reduce_max(pred_td_targets))
              tf.contrib.summary.scalar('min', tf.reduce_min(pred_td_targets))

      return td_errors_loss_fn(td_targets, pred_td_targets_1) + \
             td_errors_loss_fn(td_targets, pred_td_targets_2)

  def actor_loss(self, time_steps, dqda_clipping=None):
    """Computes the actor_loss for TD3 training.

    Args:
      time_steps: A batch of timesteps.
      dqda_clipping: A scalar float clips the gradient dqda element-wise
        between [-dqda_clipping, dqda_clipping]. Default is None representing no
        clipping. # TODO(kbanoop): Add an action norm regularizer.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      nest.assert_same_structure(time_steps, self.time_step_spec())

      actions, _ = self._actor_net(time_steps, self.action_spec())
      q_values, _ = self._critic_net_1(time_steps, actions)

      actions = nest.flatten(actions)
      dqda = tf.gradients([q_values], actions)
      actor_losses = []
      for dqda, action in zip(dqda, actions):
        if dqda_clipping is not None:
          # pylint: disable=invalid-unary-operand-type
          dqda = tf.clip_by_value(dqda, -dqda_clipping, dqda_clipping)
        actor_losses.append(
            tf.losses.mean_squared_error(
                tf.stop_gradient(dqda + action), action))
      return tf.add_n(actor_losses)
