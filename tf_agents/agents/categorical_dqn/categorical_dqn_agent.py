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

"""A Categorical DQN Agent.

Implements the Categorical DQN agent from

"A Distributional Perspective on Reinforcement Learning"
  Bellemare et al., 2017
  https://arxiv.org/abs/1707.06887
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents import tf_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import boltzmann_policy
from tf_agents.policies import categorical_q_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops


@gin.configurable
class CategoricalDqnAgent(dqn_agent.DqnAgent):
  """A Categorical DQN Agent based on the DQN Agent."""

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      categorical_q_network: network.Network,
      optimizer: types.Optimizer,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      min_q_value: types.Float = -10.0,
      max_q_value: types.Float = 10.0,
      epsilon_greedy: types.Float = 0.1,
      n_step_update: int = 1,
      boltzmann_temperature: Optional[types.Float] = None,
      # Params for target network updates
      target_categorical_q_network: Optional[network.Network] = None,
      target_update_tau: types.Float = 1.0,
      target_update_period: types.Int = 1,
      # Params for training.
      td_errors_loss_fn: Optional[types.LossFn] = None,
      gamma: types.Float = 1.0,
      reward_scale_factor: types.Float = 1.0,
      gradient_clipping: Optional[types.Float] = None,
      # Params for debugging
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      train_step_counter: Optional[tf.Variable] = None,
      name: Optional[Text] = None):
    """Creates a Categorical DQN Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A `BoundedTensorSpec` representing the actions.
      categorical_q_network: A categorical_q_network.CategoricalQNetwork that
        returns the q_distribution for each action.
      optimizer: The optimizer to use for training.
      observation_and_action_constraint_splitter: A function used to process
        observations with action constraints. These constraints can indicate,
        for example, a mask of valid/invalid actions for a given state of the
        environment.
        The function takes in a full observation and returns a tuple consisting
        of 1) the part of the observation intended as input to the network and
        2) the constraint. An example
        `observation_and_action_constraint_splitter` could be as simple as:
        ```
        def observation_and_action_constraint_splitter(observation):
          return observation['network_input'], observation['constraint']
        ```
        *Note*: when using `observation_and_action_constraint_splitter`, make
        sure the provided `q_network` is compatible with the network-specific
        half of the output of the `observation_and_action_constraint_splitter`.
        In particular, `observation_and_action_constraint_splitter` will be
        called on the observation before passing to the network.
        If `observation_and_action_constraint_splitter` is None, action
        constraints are not applied.
      min_q_value: A float specifying the minimum Q-value, used for setting up
        the support.
      max_q_value: A float specifying the maximum Q-value, used for setting up
        the support.
      epsilon_greedy: probability of choosing a random action in the default
        epsilon-greedy collect policy (used only if a wrapper is not provided to
        the collect_policy method).
      n_step_update: The number of steps to consider when computing TD error and
        TD loss. Defaults to single-step updates. Note that this requires the
        user to call train on Trajectory objects with a time dimension of
        `n_step_update + 1`. However, note that we do not yet support
        `n_step_update > 1` in the case of RNNs (i.e., non-empty
        `q_network.state_spec`).
      boltzmann_temperature: Temperature value to use for Boltzmann sampling of
        the actions during data collection. The closer to 0.0, the higher the
        probability of choosing the best action.
      target_categorical_q_network: (Optional.)  A `tf_agents.network.Network`
        to be used as the target network during Q learning.  Every
        `target_update_period` train steps, the weights from
        `categorical_q_network` are copied (possibly with smoothing via
        `target_update_tau`) to `target_categorical_q_network`.

        If `target_categorical_q_network` is not provided, it is created by
        making a copy of `categorical_q_network`, which initializes a new
        network with the same structure and its own layers and weights.

        Network copying is performed via the `Network.copy` superclass method,
        and may inadvertently lead to the resulting network to share weights
        with the original.  This can happen if, for example, the original
        network accepted a pre-built Keras layer in its `__init__`, or
        accepted a Keras layer that wasn't built, but neglected to create
        a new copy.

        In these cases, it is up to you to provide a target Network having
        weights that are not shared with the original `categorical_q_network`.
        If you provide a `target_categorical_q_network` that shares any
        weights with `categorical_q_network`, a warning will be logged but
        no exception is thrown.

        Note; shallow copies of Keras layers may be built via the code:

        ```python
        new_layer = type(layer).from_config(layer.get_config())
        ```
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      td_errors_loss_fn: A function for computing the TD errors loss. If None, a
        default value of huber_loss is used. This function takes as input the
        target and the estimated Q values and returns the loss for each element
        of the batch.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      TypeError: If the action spec contains more than one action.
      TypeError: If the q network(s) lack a `num_atoms` property.
    """
    def check_atoms(net, label):
      try:
        num_atoms = net.num_atoms
      except AttributeError:
        raise TypeError('Expected {} to have property `num_atoms`, but it '
                        'doesn\'t. (Note: you likely want to use a '
                        'CategoricalQNetwork.) Network is: {}'.format(
                            label, net))
      return num_atoms

    self._num_atoms = check_atoms(
        categorical_q_network, 'categorical_q_network')

    if target_categorical_q_network is not None:
      target_num_atoms = check_atoms(
          target_categorical_q_network, 'target_categorical_q_network')
      if self._num_atoms != target_num_atoms:
        raise ValueError(
            'categorical_q_network and target_categorical_q_network have '
            'different numbers of atoms: {} vs. {}'.format(
                self._num_atoms, target_num_atoms))

    self._min_q_value = min_q_value
    self._max_q_value = max_q_value
    min_q_value = tf.convert_to_tensor(min_q_value, dtype_hint=tf.float32)
    max_q_value = tf.convert_to_tensor(max_q_value, dtype_hint=tf.float32)
    self._support = tf.linspace(min_q_value, max_q_value, self._num_atoms)

    super(CategoricalDqnAgent, self).__init__(
        time_step_spec,
        action_spec,
        categorical_q_network,
        optimizer,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        boltzmann_temperature=boltzmann_temperature,
        target_q_network=target_categorical_q_network,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name)

  def _setup_policy(self, time_step_spec, action_spec,
                    boltzmann_temperature, emit_log_probability):
    policy = categorical_q_policy.CategoricalQPolicy(
        time_step_spec,
        action_spec,
        self._q_network,
        self._min_q_value,
        self._max_q_value,
        observation_and_action_constraint_splitter=(
            self._observation_and_action_constraint_splitter))

    if boltzmann_temperature is not None:
      collect_policy = boltzmann_policy.BoltzmannPolicy(
          policy, temperature=boltzmann_temperature)
    else:
      collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
          policy, epsilon=self._epsilon_greedy)
    policy = greedy_policy.GreedyPolicy(policy)

    target_policy = categorical_q_policy.CategoricalQPolicy(
        time_step_spec,
        action_spec,
        self._target_q_network,
        self._min_q_value,
        self._max_q_value,
        observation_and_action_constraint_splitter=(
            self._observation_and_action_constraint_splitter))
    self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)

    return policy, collect_policy

  def _check_network_output(self, net, label):
    network_utils.check_single_floating_network_output(
        net.create_variables(),
        expected_output_shape=(self._num_actions, self._num_atoms),
        label=label)

  def _loss(self,
            experience,
            td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
            gamma=1.0,
            reward_scale_factor=1.0,
            weights=None,
            training=False):
    """Computes critic loss for CategoricalDQN training.

    See Algorithm 1 and the discussion immediately preceding it in page 6 of
    "A Distributional Perspective on Reinforcement Learning"
      Bellemare et al., 2017
      https://arxiv.org/abs/1707.06887

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.policy.step_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.required_experience_time_steps` if that
        property is not `None`.
      td_errors_loss_fn: A function(td_targets, predictions) to compute loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional weights used for importance sampling.
      training: Whether the loss is being used for training.
    Returns:
      critic_loss: A scalar critic loss.
    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    # Check that `experience` includes two outer dimensions [B, T, ...]. This
    # method requires a time dimension to compute the loss properly.
    self._check_trajectory_dimensions(experience)

    squeeze_time_dim = not self._q_network.state_spec
    if self._n_step_update == 1:
      time_steps, policy_steps, next_time_steps = (
          trajectory.experience_to_transitions(experience, squeeze_time_dim))
      actions = policy_steps.action
    else:
      # To compute n-step returns, we need the first time steps, the first
      # actions, and the last time steps. Therefore we extract the first and
      # last transitions from our Trajectory.
      first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
      last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
      time_steps, policy_steps, _ = (
          trajectory.experience_to_transitions(
              first_two_steps, squeeze_time_dim))
      actions = policy_steps.action
      _, _, next_time_steps = (
          trajectory.experience_to_transitions(
              last_two_steps, squeeze_time_dim))

    with tf.name_scope('critic_loss'):
      nest_utils.assert_same_structure(actions, self.action_spec)
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)
      nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

      rank = nest_utils.get_outer_rank(time_steps.observation,
                                       self._time_step_spec.observation)

      # If inputs have a time dimension and the q_network is stateful,
      # combine the batch and time dimension.
      batch_squash = (None
                      if rank <= 1 or self._q_network.state_spec in ((), None)
                      else network_utils.BatchSquash(rank))

      network_observation = time_steps.observation

      if self._observation_and_action_constraint_splitter is not None:
        network_observation, _ = (
            self._observation_and_action_constraint_splitter(
                network_observation))

      # q_logits contains the Q-value logits for all actions.
      q_logits, _ = self._q_network(network_observation,
                                    step_type=time_steps.step_type,
                                    training=training)

      if batch_squash is not None:
        # Squash outer dimensions to a single dimensions for facilitation
        # computing the loss the following. Required for supporting temporal
        # inputs, for example.
        q_logits = batch_squash.flatten(q_logits)
        actions = batch_squash.flatten(actions)
        next_time_steps = tf.nest.map_structure(batch_squash.flatten,
                                                next_time_steps)

      next_q_distribution = self._next_q_distribution(next_time_steps)

      if actions.shape.rank > 1:
        actions = tf.squeeze(actions, list(range(1, actions.shape.rank)))

      # Project the sample Bellman update \hat{T}Z_{\theta} onto the original
      # support of Z_{\theta} (see Figure 1 in paper).
      batch_size = q_logits.shape[0] or tf.shape(q_logits)[0]
      tiled_support = tf.tile(self._support, [batch_size])
      tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

      if self._n_step_update == 1:
        discount = next_time_steps.discount
        if discount.shape.rank == 1:
          # We expect discount to have a shape of [batch_size], while
          # tiled_support will have a shape of [batch_size, num_atoms]. To
          # multiply these, we add a second dimension of 1 to the discount.
          discount = tf.expand_dims(discount, -1)
        next_value_term = tf.multiply(discount,
                                      tiled_support,
                                      name='next_value_term')

        reward = next_time_steps.reward
        if reward.shape.rank == 1:
          # See the explanation above.
          reward = tf.expand_dims(reward, -1)
        reward_term = tf.multiply(reward_scale_factor,
                                  reward,
                                  name='reward_term')

        target_support = tf.add(reward_term, gamma * next_value_term,
                                name='target_support')
      else:
        # When computing discounted return, we need to throw out the last time
        # index of both reward and discount, which are filled with dummy values
        # to match the dimensions of the observation.
        rewards = reward_scale_factor * experience.reward[:, :-1]
        discounts = gamma * experience.discount[:, :-1]

        # TODO(b/134618876): Properly handle Trajectories that include episode
        # boundaries with nonzero discount.

        discounted_returns = value_ops.discounted_return(
            rewards=rewards,
            discounts=discounts,
            final_value=tf.zeros([batch_size], dtype=discounts.dtype),
            time_major=False,
            provide_all_returns=False)

        # Convert discounted_returns from [batch_size] to [batch_size, 1]
        discounted_returns = tf.expand_dims(discounted_returns, -1)

        final_value_discount = tf.reduce_prod(discounts, axis=1)
        final_value_discount = tf.expand_dims(final_value_discount, -1)

        # Save the values of discounted_returns and final_value_discount in
        # order to check them in unit tests.
        self._discounted_returns = discounted_returns
        self._final_value_discount = final_value_discount

        target_support = tf.add(discounted_returns,
                                final_value_discount * tiled_support,
                                name='target_support')

      target_distribution = tf.stop_gradient(project_distribution(
          target_support, next_q_distribution, self._support))

      # Obtain the current Q-value logits for the selected actions.
      indices = tf.range(batch_size)
      indices = tf.cast(indices, actions.dtype)
      reshaped_actions = tf.stack([indices, actions], axis=-1)
      chosen_action_logits = tf.gather_nd(q_logits, reshaped_actions)

      # Compute the cross-entropy loss between the logits. If inputs have
      # a time dimension, compute the sum over the time dimension before
      # computing the mean over the batch dimension.
      if batch_squash is not None:
        target_distribution = batch_squash.unflatten(target_distribution)
        chosen_action_logits = batch_squash.unflatten(chosen_action_logits)
        critic_loss = tf.reduce_sum(
            tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                labels=target_distribution,
                logits=chosen_action_logits),
            axis=1)
      else:
        critic_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
            labels=target_distribution,
            logits=chosen_action_logits)

      agg_loss = common.aggregate_losses(
          per_example_loss=critic_loss,
          regularization_loss=self._q_network.losses)
      total_loss = agg_loss.total_loss

      dict_losses = {'critic_loss': agg_loss.weighted,
                     'reg_loss': agg_loss.regularization,
                     'total_loss': total_loss}

      common.summarize_scalar_dict(dict_losses,
                                   step=self.train_step_counter,
                                   name_scope='Losses/')

      if self._debug_summaries:
        distribution_errors = target_distribution - chosen_action_logits
        with tf.name_scope('distribution_errors'):
          common.generate_tensor_summaries(
              'distribution_errors', distribution_errors,
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'mean', tf.reduce_mean(distribution_errors),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'mean_abs', tf.reduce_mean(tf.abs(distribution_errors)),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'max', tf.reduce_max(distribution_errors),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              'min', tf.reduce_min(distribution_errors),
              step=self.train_step_counter)
        with tf.name_scope('target_distribution'):
          common.generate_tensor_summaries(
              'target_distribution', target_distribution,
              step=self.train_step_counter)

      # TODO(b/127318640): Give appropriate values for td_loss and td_error for
      # prioritized replay.
      return tf_agent.LossInfo(total_loss, dqn_agent.DqnLossInfo(td_loss=(),
                                                                 td_error=()))

  def _next_q_distribution(self, next_time_steps):
    """Compute the q distribution of the next state for TD error computation.

    Args:
      next_time_steps: A batch of next timesteps

    Returns:
      A [batch_size, num_atoms] tensor representing the Q-distribution for the
      next state.
    """
    network_observation = next_time_steps.observation

    if self._observation_and_action_constraint_splitter is not None:
      network_observation, _ = self._observation_and_action_constraint_splitter(
          network_observation)

    next_target_logits, _ = self._target_q_network(
        network_observation,
        step_type=next_time_steps.step_type,
        training=False)
    batch_size = next_target_logits.shape[0] or tf.shape(next_target_logits)[0]
    next_target_probabilities = tf.nn.softmax(next_target_logits)
    next_target_q_values = tf.reduce_sum(
        self._support * next_target_probabilities, axis=-1)
    dummy_state = self._target_greedy_policy.get_initial_state(batch_size)
    # Find the greedy actions using our target greedy policy. This ensures that
    # action constraints are respected and helps centralize the greedy logic.
    greedy_actions = self._target_greedy_policy.action(
        next_time_steps, dummy_state).action
    next_qt_argmax = tf.cast(greedy_actions, tf.int32)[:, None]
    batch_indices = tf.range(
        tf.cast(tf.shape(next_target_q_values)[0], tf.int32))[:, None]
    next_qt_argmax = tf.concat([batch_indices, next_qt_argmax], axis=-1)
    return tf.gather_nd(next_target_probabilities, next_qt_argmax)


# The following method is copied from the Dopamine codebase with permission
# (https://github.com/google/dopamine). Thanks to Marc Bellemare and also to
# Pablo Castro, who wrote the original version of this method.
def project_distribution(supports: types.Tensor,
                         weights: types.Tensor,
                         target_support: types.Tensor,
                         validate_args: bool = False) -> types.Tensor:
  """Projects a batch of (support, weights) onto target_support.

  Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
  In the rest of the comments we will refer to this equation simply as Eq7.

  This code is not easy to digest, so we will use a running example to clarify
  what is going on, with the following sample inputs:

    * supports =       [[0, 2, 4, 6, 8],
                        [1, 3, 4, 5, 6]]
    * weights =        [[0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.2, 0.5, 0.1, 0.1]]
    * target_support = [4, 5, 6, 7, 8]

  In the code below, comments preceded with 'Ex:' will be referencing the above
  values.

  Args:
    supports: Tensor of shape (batch_size, num_dims) defining supports for the
      distribution.
    weights: Tensor of shape (batch_size, num_dims) defining weights on the
      original support points. Although for the CategoricalDQN agent these
      weights are probabilities, it is not required that they are.
    target_support: Tensor of shape (num_dims) defining support of the projected
      distribution. The values must be monotonically increasing. Vmin and Vmax
      will be inferred from the first and last elements of this tensor,
      respectively. The values in this tensor must be equally spaced.
    validate_args: Whether we will verify the contents of the
      target_support parameter.

  Returns:
    A Tensor of shape (batch_size, num_dims) with the projection of a batch of
    (support, weights) onto target_support.

  Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
  """
  target_support_deltas = target_support[1:] - target_support[:-1]
  # delta_z = `\Delta z` in Eq7.
  delta_z = target_support_deltas[0]
  validate_deps = []
  supports.shape.assert_is_compatible_with(weights.shape)
  supports[0].shape.assert_is_compatible_with(target_support.shape)
  target_support.shape.assert_has_rank(1)
  if validate_args:
    # Assert that supports and weights have the same shapes.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(supports), tf.shape(weights))),
            [supports, weights]))
    # Assert that elements of supports and target_support have the same shape.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(
                tf.equal(tf.shape(supports)[1], tf.shape(target_support))),
            [supports, target_support]))
    # Assert that target_support has a single dimension.
    validate_deps.append(
        tf.Assert(
            tf.equal(tf.size(tf.shape(target_support)), 1), [target_support]))
    # Assert that the target_support is monotonically increasing.
    validate_deps.append(
        tf.Assert(tf.reduce_all(target_support_deltas > 0), [target_support]))
    # Assert that the values in target_support are equally spaced.
    validate_deps.append(
        tf.Assert(
            tf.reduce_all(tf.equal(target_support_deltas, delta_z)),
            [target_support]))

  with tf.control_dependencies(validate_deps):
    # Ex: `v_min, v_max = 4, 8`.
    v_min, v_max = target_support[0], target_support[-1]
    # Ex: `batch_size = 2`.
    batch_size = tf.shape(supports)[0]
    # `N` in Eq7.
    # Ex: `num_dims = 5`.
    num_dims = tf.shape(target_support)[0]
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    # Ex: `clipped_support = [[[ 4.  4.  4.  6.  8.]]
    #                         [[ 4.  4.  4.  5.  6.]]]`.
    clipped_support = tf.clip_by_value(supports, v_min, v_max)[:, None, :]
    # Ex: `tiled_support = [[[[ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]
    #                         [ 4.  4.  4.  6.  8.]]
    #                        [[ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]
    #                         [ 4.  4.  4.  5.  6.]]]]`.
    tiled_support = tf.tile([clipped_support], [1, 1, num_dims, 1])
    # Ex: `reshaped_target_support = [[[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]
    #                                 [[ 4.]
    #                                  [ 5.]
    #                                  [ 6.]
    #                                  [ 7.]
    #                                  [ 8.]]]`.
    reshaped_target_support = tf.tile(target_support[:, None], [batch_size, 1])
    reshaped_target_support = tf.reshape(reshaped_target_support,
                                         [batch_size, num_dims, 1])
    # numerator = `|clipped_support - z_i|` in Eq7.
    # Ex: `numerator = [[[[ 0.  0.  0.  2.  4.]
    #                     [ 1.  1.  1.  1.  3.]
    #                     [ 2.  2.  2.  0.  2.]
    #                     [ 3.  3.  3.  1.  1.]
    #                     [ 4.  4.  4.  2.  0.]]
    #                    [[ 0.  0.  0.  1.  2.]
    #                     [ 1.  1.  1.  0.  1.]
    #                     [ 2.  2.  2.  1.  0.]
    #                     [ 3.  3.  3.  2.  1.]
    #                     [ 4.  4.  4.  3.  2.]]]]`.
    numerator = tf.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    # Ex: `clipped_quotient = [[[[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  1.]]
    #                           [[ 1.  1.  1.  0.  0.]
    #                            [ 0.  0.  0.  1.  0.]
    #                            [ 0.  0.  0.  0.  1.]
    #                            [ 0.  0.  0.  0.  0.]
    #                            [ 0.  0.  0.  0.  0.]]]]`.
    clipped_quotient = tf.clip_by_value(quotient, 0, 1)
    # Ex: `weights = [[ 0.1  0.6  0.1  0.1  0.1]
    #                 [ 0.1  0.2  0.5  0.1  0.1]]`.
    weights = weights[:, None, :]
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))`
    # in Eq7.
    # Ex: `inner_prod = [[[[ 0.1  0.6  0.1  0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0.1]]
    #                     [[ 0.1  0.2  0.5  0.  0. ]
    #                      [ 0.   0.   0.   0.1 0. ]
    #                      [ 0.   0.   0.   0.  0.1]
    #                      [ 0.   0.   0.   0.  0. ]
    #                      [ 0.   0.   0.   0.  0. ]]]]`.
    inner_prod = clipped_quotient * weights
    # Ex: `projection = [[ 0.8 0.0 0.1 0.0 0.1]
    #                    [ 0.8 0.1 0.1 0.0 0.0]]`.
    projection = tf.reduce_sum(inner_prod, 3)
    projection = tf.reshape(projection, [batch_size, num_dims])
    return projection
