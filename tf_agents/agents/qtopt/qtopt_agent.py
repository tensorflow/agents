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

"""A Qtopt Agent.

Implements the Qt-Opt algorithm from

"QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic"
  "Manipulation"

 Dmitry Kalashnikov et al., 2018
 https://arxiv.org/abs/1806.10293

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import utils as network_utils
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import qtopt_cem_policy
from tf_agents.specs import tensor_spec
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils


def compute_td_targets(next_q_values: types.Tensor,
                       rewards: types.Tensor,
                       discounts: types.Tensor) -> types.Tensor:
  return tf.stop_gradient(rewards + discounts * next_q_values)


class QtOptLossInfo(typing.NamedTuple):
  """QtOptLossInfo is stored in the `extras` field of the LossInfo instance.

  Both `td_loss` and `td_error` have a validity mask applied to ensure that
  no loss or error is calculated for episode boundaries.

  td_loss: The **weighted** TD loss (depends on choice of loss metric and
    any weights passed to the QtOpt loss function.
  td_error: The **unweighted** TD errors, which are just calculated as:

    ```
    td_error = td_targets - q_values
    ```

    These can be used to update Prioritized Replay Buffer priorities.

    Note that, unlike `td_loss`, `td_error` may contain a time dimension when
    training with RNN mode.  For `td_loss`, this axis is averaged out.
  """
  td_loss: types.Tensor
  td_error: types.Tensor


@gin.configurable
class QtOptAgent(tf_agent.TFAgent):
  """A Qtopt Agent.

  Implements the Qt-Opt algorithm from

  "QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic "
    "Manipulation"

   Dmitry Kalashnikov et al., 2018
   https://arxiv.org/abs/1806.10293

  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      q_network,
      optimizer,
      actions_sampler,
      epsilon_greedy=0.1,
      n_step_update=1,
      emit_log_probability=False,
      in_graph_bellman_update=True,
      # Params for cem
      init_mean_cem=None,
      init_var_cem=None,
      num_samples_cem=32,
      num_elites_cem=4,
      num_iter_cem=3,
      # Params for target network updates
      target_q_network=None,
      target_update_tau=1.0,
      target_update_period=1,
      enable_td3=True,
      target_q_network_delayed=None,
      target_q_network_delayed_2=None,
      delayed_target_update_period=5,
      # Params for training.
      td_errors_loss_fn=None,
      auxiliary_loss_fns=None,
      gamma=1.0,
      reward_scale_factor=1.0,
      gradient_clipping=None,
      # Params for debugging
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=None,
      info_spec=None,
      name=None):
    """Creates a Qtopt Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call((observation, action), step_type). The
        q_network is different from the one used in DQN where the input is state
        and the output has multiple dimension representing Q values for
        different actions. The input of this q_network is a tuple of state and
        action. The output is one dimension representing Q value for that
        specific action. DDPG critic network can be used directly here.
      optimizer: The optimizer to use for training.
      actions_sampler: A tf_agents.policies.sampler.ActionsSampler to be used to
        sample actions in CEM.
      epsilon_greedy: probability of choosing a random action in the default
        epsilon-greedy collect policy (used only if a wrapper is not provided to
        the collect_policy method).
      n_step_update: Currently, only n_step_update == 1 is supported.
      emit_log_probability: Whether policies emit log probabilities or not.
      in_graph_bellman_update: If False, configures the agent to expect
        experience containing computed q_values in the policy_step's info field.
        This allows simplifies splitting the loss calculation across several
        jobs.
      init_mean_cem: Initial mean value of the Gaussian distribution to sample
        actions for CEM.
      init_var_cem: Initial variance value of the Gaussian distribution to
        sample actions for CEM.
      num_samples_cem: Number of samples to sample for each iteration in CEM.
      num_elites_cem: Number of elites to select for each iteration in CEM.
      num_iter_cem: Number of iterations in CEM.
      target_q_network: (Optional.)  A `tf_agents.network.Network`
        to be used as the target network during Q learning.  Every
        `target_update_period` train steps, the weights from
        `q_network` are copied (possibly with smoothing via
        `target_update_tau`) to `target_q_network`.

        If `target_q_network` is not provided, it is created by
        making a copy of `q_network`, which initializes a new
        network with the same structure and its own layers and weights.

        Network copying is performed via the `Network.copy` superclass method,
        with the same arguments used during the original network's construction
        and may inadvertently lead to weights being shared between networks.
        This can happen if, for example, the original
        network accepted a pre-built Keras layer in its `__init__`, or
        accepted a Keras layer that wasn't built, but neglected to create
        a new copy.

        In these cases, it is up to you to provide a target Network having
        weights that are not shared with the original `q_network`.
        If you provide a `target_q_network` that shares any
        weights with `q_network`, an exception is thrown.

      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      enable_td3: Whether or not to enable using a delayed target network to
        calculate q value and assign min(q_delayed, q_delayed_2) as
        q_next_state.
      target_q_network_delayed: (Optional.) Similar network as
        target_q_network but lags behind even more. See documentation
        for target_q_network. Will only be used if 'enable_td3' is True.
      target_q_network_delayed_2: (Optional.) Similar network as
        target_q_network_delayed but lags behind even more. See documentation
        for target_q_network. Will only be used if 'enable_td3' is True.
      delayed_target_update_period: Used when enable_td3 is true. Period for
        soft update of the delayed target networks.
      td_errors_loss_fn: A function for computing the TD errors loss. If None, a
        default value of element_wise_huber_loss is used. This function takes as
        input the target and the estimated Q values and returns the loss for
        each element of the batch.
      auxiliary_loss_fns: An optional list of functions for computing auxiliary
        losses. Each auxiliary_loss_fn expects network and transition as
        input and should output auxiliary_loss and auxiliary_reg_loss.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      info_spec: If not None, the policy info spec is set to this spec.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or action
        spec minimum is not equal to 0.
      NotImplementedError: If `q_network` has non-empty `state_spec` (i.e., an
        RNN is provided) and `n_step_update > 1`.
    """
    tf.Module.__init__(self, name=name)

    self._sampler = actions_sampler
    self._init_mean_cem = init_mean_cem
    self._init_var_cem = init_var_cem
    self._num_samples_cem = num_samples_cem
    self._num_elites_cem = num_elites_cem
    self._num_iter_cem = num_iter_cem
    self._in_graph_bellman_update = in_graph_bellman_update
    if not in_graph_bellman_update:
      if info_spec is not None:
        self._info_spec = info_spec
      else:
        self._info_spec = {
            'target_q': tensor_spec.TensorSpec((), tf.float32),
        }
    else:
      self._info_spec = ()

    self._q_network = q_network
    net_observation_spec = (time_step_spec.observation, action_spec)

    q_network.create_variables(net_observation_spec)

    if target_q_network:
      target_q_network.create_variables(net_observation_spec)

    self._target_q_network = common.maybe_copy_target_network_with_checks(
        self._q_network, target_q_network, input_spec=net_observation_spec,
        name='TargetQNetwork')

    self._target_updater = self._get_target_updater(target_update_tau,
                                                    target_update_period)

    self._enable_td3 = enable_td3

    if (not self._enable_td3 and
        (target_q_network_delayed or target_q_network_delayed_2)):
      raise ValueError('enable_td3 is set to False but target_q_network_delayed'
                       ' or target_q_network_delayed_2 is passed.')

    if self._enable_td3:
      if target_q_network_delayed:
        target_q_network_delayed.create_variables()
      self._target_q_network_delayed = (
          common.maybe_copy_target_network_with_checks(
              self._q_network, target_q_network_delayed,
              'TargetQNetworkDelayed'))
      self._target_updater_delayed = self._get_target_updater_delayed(
          1.0, delayed_target_update_period)

      if target_q_network_delayed_2:
        target_q_network_delayed_2.create_variables()
      self._target_q_network_delayed_2 = (
          common.maybe_copy_target_network_with_checks(
              self._q_network, target_q_network_delayed_2,
              'TargetQNetworkDelayed2'))
      self._target_updater_delayed_2 = self._get_target_updater_delayed_2(
          1.0, delayed_target_update_period)

      self._update_target = self._update_both
    else:
      self._update_target = self._target_updater
      self._target_q_network_delayed = None
      self._target_q_network_delayed_2 = None

    self._check_network_output(self._q_network, 'q_network')
    self._check_network_output(self._target_q_network, 'target_q_network')

    self._epsilon_greedy = epsilon_greedy
    self._n_step_update = n_step_update
    self._optimizer = optimizer
    self._td_errors_loss_fn = (
        td_errors_loss_fn or common.element_wise_huber_loss)
    self._auxiliary_loss_fns = auxiliary_loss_fns
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._gradient_clipping = gradient_clipping

    policy, collect_policy = self._setup_policy(time_step_spec, action_spec,
                                                emit_log_probability)

    if q_network.state_spec and n_step_update != 1:
      raise NotImplementedError(
          'QtOptAgent does not currently support n-step updates with stateful '
          'networks (i.e., RNNs), but n_step_update = {}'.format(n_step_update))

    # Bypass the train_sequence_length check when RNN is used.
    train_sequence_length = (
        n_step_update + 1 if not q_network.state_spec else None)

    super(QtOptAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=train_sequence_length,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
    )

    self._setup_data_converter(q_network, gamma, n_step_update)

  @property
  def policy_q_network(self):
    return self._target_q_network

  @property
  def enable_td3(self):
    return self._enable_td3

  def _setup_data_converter(self, q_network, gamma, n_step_update):
    if q_network.state_spec:
      if not self._in_graph_bellman_update:
        self._data_context = data_converter.DataContext(
            time_step_spec=self._time_step_spec,
            action_spec=self._action_spec,
            info_spec=self._collect_policy.info_spec,
            policy_state_spec=self._q_network.state_spec,
            use_half_transition=True)
        self._as_transition = data_converter.AsHalfTransition(
            self.data_context, squeeze_time_dim=False)
      else:
        self._data_context = data_converter.DataContext(
            time_step_spec=self._time_step_spec,
            action_spec=self._action_spec,
            info_spec=self._collect_policy.info_spec,
            policy_state_spec=self._q_network.state_spec,
            use_half_transition=False)
        self._as_transition = data_converter.AsTransition(
            self.data_context, squeeze_time_dim=False,
            prepend_t0_to_next_time_step=True)
    else:
      if not self._in_graph_bellman_update:
        self._data_context = data_converter.DataContext(
            time_step_spec=self._time_step_spec,
            action_spec=self._action_spec,
            info_spec=self._collect_policy.info_spec,
            policy_state_spec=self._q_network.state_spec,
            use_half_transition=True)

        self._as_transition = data_converter.AsHalfTransition(
            self.data_context, squeeze_time_dim=True)
      else:
        # This reduces the n-step return and removes the extra time dimension,
        # allowing the rest of the computations to be independent of the
        # n-step parameter.
        self._as_transition = data_converter.AsNStepTransition(
            self.data_context, gamma=gamma, n=n_step_update)

  def _setup_policy(self, time_step_spec, action_spec, emit_log_probability):
    policy = qtopt_cem_policy.CEMPolicy(
        time_step_spec,
        action_spec,
        q_network=self._target_q_network,
        sampler=self._sampler,
        init_mean=self._init_mean_cem,
        init_var=self._init_var_cem,
        info_spec=self._info_spec,
        num_samples=self._num_samples_cem,
        num_elites=self._num_elites_cem,
        num_iterations=self._num_iter_cem,
        emit_log_probability=emit_log_probability,
        training=False)

    collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        policy, epsilon=self._epsilon_greedy)

    return policy, collect_policy

  def _check_network_output(self, net, label):
    network_utils.check_single_floating_network_output(
        net.create_variables(), expected_output_shape=(), label=label)

  def _initialize(self):
    common.soft_variables_update(
        self._q_network.variables, self._target_q_network.variables, tau=1.0)
    if self._enable_td3:
      common.soft_variables_update(
          self._q_network.variables,
          self._target_q_network_delayed.variables, tau=1.0)
      common.soft_variables_update(
          self._q_network.variables,
          self._target_q_network_delayed_2.variables, tau=1.0)

  def _update_both(self):
    self._target_updater_delayed_2()
    self._target_updater_delayed()
    self._target_updater()

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network.

    For each weight w_s in the q network, and its corresponding
    weight w_t in the target_q_network, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update. Used
        for target network.
      period: Step interval at which the target network is updated. Used for
        target network.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets'):

      def update():
        return common.soft_variables_update(
            self._q_network.variables,
            self._target_q_network.variables,
            tau,
            tau_non_trainable=1.0)

      return common.Periodically(update, period, 'periodic_update_targets')

  def _get_target_updater_delayed(self, tau_delayed=1.0, period_delayed=1):
    """Performs a soft update of the delayed target network.

    For each weight w_s in the q network, and its corresponding
    weight w_t in the target_q_network, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
      tau_delayed: A float scalar in [0, 1]. Default `tau=1.0` means hard
        update. Used for delayed target network.
      period_delayed: Step interval at which the target network is updated. Used
        for delayed target network.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets_delayed'):

      def update_delayed():
        return common.soft_variables_update(
            self._target_q_network.variables,
            self._target_q_network_delayed.variables,
            tau_delayed,
            tau_non_trainable=1.0)

      return common.Periodically(update_delayed, period_delayed,
                                 'periodic_update_targets_delayed')

  def _get_target_updater_delayed_2(self, tau_delayed=1.0, period_delayed=1):
    """Performs a soft update of the delayed target network.

    For each weight w_s in the q network, and its corresponding
    weight w_t in the target_q_network, a soft update is:
    w_t = (1 - tau) * w_t + tau * w_s

    Args:
      tau_delayed: A float scalar in [0, 1]. Default `tau=1.0` means hard
        update. Used for delayed target network.
      period_delayed: Step interval at which the target network is updated. Used
        for delayed target network.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets_delayed'):

      def update_delayed():
        return common.soft_variables_update(
            self._target_q_network_delayed.variables,
            self._target_q_network_delayed_2.variables,
            tau_delayed,
            tau_non_trainable=1.0)

      return common.Periodically(update_delayed, period_delayed,
                                 'periodic_update_targets_delayed')

  # Use @common.function in graph mode or for speeding up.
  def _train(self, experience, weights):
    with tf.GradientTape() as tape:
      loss_info = self._loss(
          experience,
          weights=weights,
          training=True)

    tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
    variables_to_train = self._q_network.trainable_weights

    non_trainable_weights = self._q_network.non_trainable_weights
    assert list(variables_to_train), "No variables in the agent's q_network."
    grads = tape.gradient(loss_info.loss, variables_to_train)
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = list(zip(grads, variables_to_train))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      grads_and_vars_with_non_trainable = (
          grads_and_vars + [(None, v) for v in non_trainable_weights])
      eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)
    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)

    self._update_target()

    return loss_info

  def _add_auxiliary_losses(self, transition, weights, losses_dict):
    """Computes auxiliary losses, updating losses_dict in place."""
    total_auxiliary_loss = 0
    if self._auxiliary_loss_fns is not None:
      for auxiliary_loss_fn in self._auxiliary_loss_fns:
        auxiliary_loss, auxiliary_reg_loss = auxiliary_loss_fn(
            network=self._q_network, transition=transition)
        agg_auxiliary_loss = common.aggregate_losses(
            per_example_loss=auxiliary_loss,
            sample_weight=weights,
            regularization_loss=auxiliary_reg_loss)
        total_auxiliary_loss += agg_auxiliary_loss.total_loss
        losses_dict.update(
            {'auxiliary_loss_{}'.format(
                auxiliary_loss_fn.__name__
                ): agg_auxiliary_loss.weighted,
             'auxiliary_reg_loss_{}'.format(
                 auxiliary_loss_fn.__name__
                 ): agg_auxiliary_loss.regularization,
             })
    return total_auxiliary_loss

  def _loss(self,
            experience,
            weights=None,
            training=False):
    """Computes loss for QtOpt training.

    Args:
      experience: A batch of experience data in the form of a `Trajectory` or
        `Transition`. The structure of `experience` must match that of
        `self.collect_policy.step_spec`.

        If a `Trajectory`, all tensors in `experience` must be shaped
        `[B, T, ...]` where `T` must be equal to `self.train_sequence_length`
        if that property is not `None`.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output td_loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.
      training: Whether this loss is being used for training.

    Returns:
      loss: An instance of `QtOptLossInfo`.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    transition = self._as_transition(experience)
    time_steps, policy_steps, next_time_steps = transition
    actions = policy_steps.action

    with tf.name_scope('loss'):
      q_values = self._compute_q_values(
          time_steps, actions, policy_steps.state, training=training)

      next_q_values = self._compute_next_q_values(
          next_time_steps, policy_steps.info, policy_steps.state)

      # This applies to any value of n_step_update and also in RNN-QtOpt.
      # In the RNN-QtOpt case, inputs and outputs contain a time dimension.
      td_targets = compute_td_targets(
          next_q_values,
          rewards=self._reward_scale_factor * next_time_steps.reward,
          discounts=self._gamma * next_time_steps.discount)

      valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
      td_error = valid_mask * (td_targets - q_values)

      td_loss = valid_mask * self._td_errors_loss_fn(td_targets, q_values)

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):

        # Do a sum over the time dimension.
        td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

      # Aggregate across the elements of the batch and add regularization loss.
      # Note: We use an element wise loss above to ensure each element is always
      #   weighted by 1/N where N is the batch size, even when some of the
      #   weights are zero due to boundary transitions. Weighting by 1/K where K
      #   is the actual number of non-zero weight would artificially increase
      #   their contribution in the loss. Think about what would happen as
      #   the number of boundary samples increases.

      agg_loss = common.aggregate_losses(
          per_example_loss=td_loss,
          sample_weight=weights,
          regularization_loss=self._q_network.losses)
      total_loss = agg_loss.total_loss

      losses_dict = {'td_loss': agg_loss.weighted,
                     'reg_loss': agg_loss.regularization}
      total_auxiliary_loss = self._add_auxiliary_losses(
          transition, weights, losses_dict)
      total_loss += total_auxiliary_loss

      losses_dict['total_loss'] = total_loss

      common.summarize_scalar_dict(losses_dict,
                                   step=self.train_step_counter,
                                   name_scope='Losses/')

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._q_network.trainable_weights:
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)
      if self._debug_summaries:
        diff_q_values = q_values - next_q_values
        common.generate_tensor_summaries('td_error', td_error,
                                         self.train_step_counter)
        common.generate_tensor_summaries('q_values', q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('next_q_values', next_q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('reward', next_time_steps.reward,
                                         self.train_step_counter)

      return tf_agent.LossInfo(total_loss, QtOptLossInfo(td_loss=td_loss,
                                                         td_error=td_error))

  def _compute_q_values(
      self, time_steps, actions, network_state=(), training=False):
    q_values, _ = self._q_network((time_steps.observation, actions),
                                  step_type=time_steps.step_type,
                                  network_state=network_state,
                                  training=training)

    return q_values

  def _compute_next_q_values(self, next_time_steps, info, network_state=()):
    if not self._in_graph_bellman_update:
      return info['target_q']

    next_action_policy_step = self._policy.action(
        next_time_steps, network_state)

    if self._enable_td3:
      q_values_target_delayed, _ = self._target_q_network_delayed(
          (next_time_steps.observation, next_action_policy_step.action),
          step_type=next_time_steps.step_type,
          network_state=network_state,
          training=False)

      q_values_target_delayed_2, _ = self._target_q_network_delayed_2(
          (next_time_steps.observation, next_action_policy_step.action),
          step_type=next_time_steps.step_type,
          network_state=network_state,
          training=False)

      q_next_state = tf.minimum(q_values_target_delayed_2,
                                q_values_target_delayed)
    else:
      q_next_state, _ = self._target_q_network(
          (next_time_steps.observation, next_action_policy_step.action),
          step_type=next_time_steps.step_type,
          network_state=network_state,
          training=False)

    if self._q_network.state_spec:
      q_next_state = q_next_state[:, 1:]
    return q_next_state
