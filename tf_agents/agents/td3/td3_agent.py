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

"""Twin Delayed Deep Deterministic policy gradient (TD3) agent.

"Addressing Function Approximation Error in Actor-Critic Methods"
by Fujimoto et al.

For the full paper, see https://arxiv.org/abs/1802.09477.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.agents.ddpg import networks
from tf_agents.environments import trajectory
from tf_agents.policies import actor_policy
from tf_agents.policies import ou_noise_policy
from tf_agents.policies import policy_step
import tf_agents.utils.common as common_utils
import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class Td3Agent(tf_agent.Base):
  """A TD3 Agent."""

  ACTOR_NET_SCOPE = 'actor_net'
  TARGET_ACTOR_NET_SCOPE = 'target_actor_net'
  CRITIC_NET_SCOPE = 'critic_net'
  TARGET_CRITIC_NET_SCOPE = 'target_critic_net'

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_net=networks.actor_network,
               critic_net=networks.critic_network,
               replay_buffer_ctor=None,
               ou_stddev=1.0,
               ou_damping=1.0,
               target_update_tau=1.0,
               target_update_period=1,
               actor_optimizer=None,
               critic_optimizer=None,
               train_batch_size=32,
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

    super(Td3Agent, self).__init__(time_step_spec, action_spec)

    self._critic_net_1 = tf.make_template(
        self.CRITIC_NET_SCOPE + '_1', critic_net, create_scope_now_=True)
    self._target_critic_net_1 = tf.make_template(
        self.TARGET_CRITIC_NET_SCOPE + '_1', critic_net, create_scope_now_=True)

    self._critic_net_2 = tf.make_template(
        self.CRITIC_NET_SCOPE + '_2', critic_net, create_scope_now_=True)
    self._target_critic_net_2 = tf.make_template(
        self.TARGET_CRITIC_NET_SCOPE + '_2', critic_net, create_scope_now_=True)

    self._actor_net = tf.make_template(
        self.ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
    self._target_actor_net = tf.make_template(
        self.TARGET_ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
    self._policy = actor_policy.ActorPolicy(
        time_step_spec=self.time_step_spec(), action_spec=self.action_spec(),
        actor_network=self._actor_net, clip=True)

    if replay_buffer_ctor is not None:
      # TODO(kewa): Get action_step_spec from policy.
      action_step_spec = policy_step.PolicyStep(self._action_spec)
      trajectory_spec = trajectory.from_transition(
          time_step_spec, action_step_spec, time_step_spec)
      self._replay_buffer = tf.contrib.checkpoint.NoDependency(
          replay_buffer_ctor(trajectory_spec))

    # TODO(kewa): better variable names.
    self._ou_stddev = ou_stddev
    self._ou_damping = ou_damping
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer
    self._train_batch_size = train_batch_size
    self._dqda_clipping = dqda_clipping
    self._td_errors_loss_fn = td_errors_loss_fn
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_policy_noise = target_policy_noise
    self._target_policy_noise_clip = target_policy_noise_clip
    self._debug_summaries = debug_summaries

  def initialize(self):
    """Returns an op to initialize the agent.

    Copies weights from the actor and critic networks to the respective
    target actor and critic networks.

    Returns:
      An op to initialize the agent.
    """
    return self.update_targets(tau=1.0, period=1)

  def policy(self):
    """Return the current policy held by the agent.

    Returns:
      A subclass of tf_policy.Base.
    """
    return self._policy

  def collect_policy(self, policy_wrapper=None):
    """Returns a policy for collecting experience in the environment.

    Args:
      policy_wrapper: A wrapper(policy) that modifies the agent's actor policy.
        By default, we use a wrapper that adds Ornstein-Uhlenbeck (OU) noise
        to the unclipped actions from the agent's actor network.

    Returns:
      A subclass of tf_policy.Base for collecting data.
    """
    wrapped_policy = actor_policy.ActorPolicy(
        time_step_spec=self.time_step_spec(), action_spec=self.action_spec(),
        actor_network=self._actor_net, clip=False)
    policy_wrapper = policy_wrapper or functools.partial(
        ou_noise_policy.OUNoisePolicy,
        ou_stddev=self._ou_stddev,
        ou_damping=self._ou_damping,
        clip=True)
    return policy_wrapper(wrapped_policy)

  @property
  def replay_buffer(self):
    return self._replay_buffer

  def observers(self):
    if self.replay_buffer:
      return [self.replay_buffer.add_batch]
    return []

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
        sample_batch_size=self._train_batch_size, num_steps=2)
    (time_steps, policy_steps, next_time_steps) = (
        trajectory.to_transition(*batch))
    train_op = self.train_from_experience(
        (time_steps, policy_steps.action, next_time_steps),
        actor_optimizer,
        critic_optimizer,
        train_step_counter=train_step_counter)
    return train_op

  def train_from_experience(self,
                            experience,
                            actor_optimizer=None,
                            critic_optimizer=None,
                            train_step_counter=None,
                            summarize_gradients=False):
    """Update the agent estimates given a batch of experience.

    Args:
      experience: A list/tuple containing batches of [time_steps, actions,
        next_time_steps].
      actor_optimizer: Optimizer for the actor.
      critic_optimizer: Optimizer for the critic.
      train_step_counter: A optional scalar to increment for each train step.
        Typically representing the global_step.
      summarize_gradients: If true, gradient summaries will be written.

    Returns:
      A train_op to train the actor and critic networks.
    """
    time_steps, actions, next_time_steps = experience

    critic_loss = self.critic_loss(
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn=self._td_errors_loss_fn,
        gamma=self._gamma,
        target_policy_noise=self._target_policy_noise,
        target_policy_noise_clip=self._target_policy_noise_clip,
        reward_scale_factor=self._reward_scale_factor)

    actor_loss = self.actor_loss(time_steps, self._dqda_clipping)
    with tf.name_scope('Losses'):
      tf.contrib.summary.scalar('critic_loss', critic_loss)
      tf.contrib.summary.scalar('actor_loss', actor_loss)

    critic_train_op = tf.contrib.training.create_train_op(
        critic_loss,
        critic_optimizer,
        global_step=train_step_counter,
        summarize_gradients=summarize_gradients,
        variables_to_train=self._critic_net_1.trainable_variables +
        self._critic_net_2.trainable_variables,
    )

    actor_train_op = tf.contrib.training.create_train_op(
        actor_loss,
        actor_optimizer,
        global_step=None,
        summarize_gradients=summarize_gradients,
        variables_to_train=self._actor_net.trainable_variables,
    )

    with tf.control_dependencies([critic_train_op, actor_train_op]):
      update_targets_op = self.update_targets(self._target_update_tau,
                                              self._target_update_period)

    with tf.control_dependencies([update_targets_op]):
      train_op = critic_train_op + actor_train_op

    return train_op

  def update_targets(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target networks are updated.
    Returns:
      An operation that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets'):

      def update():  # pylint: disable=missing-docstring
        critic_update_1 = common_utils.soft_variables_update(
            self._critic_net_1.global_variables,
            self._target_critic_net_1.global_variables, tau)
        critic_update_2 = common_utils.soft_variables_update(
            self._critic_net_2.global_variables,
            self._target_critic_net_2.global_variables, tau)
        actor_update = common_utils.soft_variables_update(
            self._actor_net.global_variables,
            self._target_actor_net.global_variables, tau)
        return tf.group(critic_update_1, critic_update_2, actor_update)

      return common_utils.periodically(update, period, 'update_targets')

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

      target_actions = self._target_actor_net(next_time_steps,
                                              self.action_spec())

      # Add gaussian noise to each action before computing target q values
      def add_noise_to_action(action):  # pylint: disable=missing-docstring
        dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                        scale=target_policy_noise * \
                                        tf.ones_like(action))
        noise = dist.sample()
        noise = tf.clip_by_value(noise, -target_policy_noise_clip,
                                 target_policy_noise_clip)
        return action + noise

      noisy_target_actions = nest.map_structure(add_noise_to_action,
                                                target_actions)

      # Target q-values are the min of the two networks
      target_q_values_1 = self._target_critic_net_1(next_time_steps,
                                                    noisy_target_actions)
      target_q_values_2 = self._target_critic_net_2(next_time_steps,
                                                    noisy_target_actions)
      target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_td_targets_1 = self._critic_net_1(time_steps, actions)
      pred_td_targets_2 = self._critic_net_2(time_steps, actions)
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

      actions = self._actor_net(time_steps, self.action_spec())
      q_values = self._critic_net_1(time_steps, actions)

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
