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

"""A REINFORCE Agent.

Implements the REINFORCE algorithm from (Williams, 1992):
https://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Callable, Optional, Text

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory as traj
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops

# A function `advantage(returns, value_predictions) -> advantages.
AdvantageFnType = Callable[[types.Tensor, types.Tensor], types.Tensor]


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
  normalized_values = ((values - values_mean) / (tf.sqrt(values_var) + epsilon))
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
  batch_size = (
      tf.compat.dimension_at_index(time_steps.discount.shape, 0) or
      tf.shape(time_steps.discount)[0])
  return policy.get_initial_state(batch_size=batch_size)


class ReinforceAgentLossInfo(
    collections.namedtuple(
        'ReinforceAgentLossInfo',
        ('policy_gradient_loss', 'policy_network_regularization_loss',
         'entropy_regularization_loss', 'value_estimation_loss',
         'value_network_regularization_loss'))):
  """ReinforceAgentLossInfo is stored in the `extras` field of the LossInfo.

  All losses, except for `policy_network_regularization_loss` have a validity
  mask applied to ensure no loss or error is calculated for episode boundaries.

  policy_gradient_loss: The weighted policy_gradient loss.
  policy_network_regularization_loss: The regularization loss terms from the
    policy network used to generate the `policy_gradient_loss`.
  entropy_regularization_loss: The entropy regularization loss.
  value_estimation_loss: If value estimation network is being used, the loss
    associated with that network.

  """
  pass


@gin.configurable
class ReinforceAgent(tf_agent.TFAgent):
  """A REINFORCE Agent.

  Implements:

  REINFORCE algorithm from

  "Simple statistical gradient-following algorithms for connectionist
  reinforcement learning"
  Williams, R.J., 1992.
  https://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

  REINFORCE with state-value baseline, where state-values are estimated with
  function approximation, from

  "Reinforcement learning: An introduction" (Sec. 13.4)
  Sutton, R.S. and Barto, A.G., 2018.
  http://incompleteideas.net/book/the-book-2nd.html

  The REINFORCE agent can be optionally provided with:
  - value_network: A `tf_agents.network.Network` which parameterizes state-value
    estimation as a neural network. The network will be called with
    call(observation, step_type) and returns a floating point state-values
    tensor.
  - value_estimation_loss_coef: Weight on the value prediction loss.

  If value_network and value_estimation_loss_coef are provided, advantages are
  computed as
    `advantages = (discounted accumulated rewards) - (estimated state-values)`
  and the overall learning objective becomes:
    `(total loss) =
      (policy gradient loss) +
      value_estimation_loss_coef * (squared error of estimated state-values)`

  """

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.TensorSpec,
               actor_network: network.Network,
               optimizer: types.Optimizer,
               value_network: Optional[network.Network] = None,
               value_estimation_loss_coef: types.Float = 0.2,
               advantage_fn: Optional[AdvantageFnType] = None,
               use_advantage_loss: bool = True,
               gamma: types.Float = 1.0,
               normalize_returns: bool = True,
               gradient_clipping: Optional[types.Float] = None,
               debug_summaries: bool = False,
               summarize_grads_and_vars: bool = False,
               entropy_regularization: Optional[types.Float] = None,
               train_step_counter: Optional[tf.Variable] = None,
               name: Optional[Text] = None):
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
      advantage_fn: A function `A(returns, value_preds)` that takes returns and
        value function predictions as input and returns advantages. The default
        is `A(returns, value_preds) = returns - value_preds` if a value network
        is specified and `use_advantage_loss=True`, otherwise `A(returns,
        value_preds) = returns`.
      use_advantage_loss: Whether to use value function predictions for
        computing returns. `use_advantage_loss=False` is equivalent to setting
        `advantage_fn=lambda returns, value_preds: returns`.
      gamma: A discount factor for future rewards.
      normalize_returns: Whether to normalize returns across episodes when
        computing the loss.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      entropy_regularization: Coefficient for entropy regularization loss term.
      train_step_counter: An optional counter to increment every time the train
        op is run. Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)

    actor_network.create_variables()
    self._actor_network = actor_network
    if value_network:
      value_network.create_variables()
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
    self._advantage_fn = advantage_fn
    if self._advantage_fn is None:
      if use_advantage_loss and self._baseline:
        self._advantage_fn = lambda returns, value_preds: returns - value_preds
      else:
        self._advantage_fn = lambda returns, _: returns

    super(ReinforceAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)
    self._as_trajectory = data_converter.AsTrajectory(self.data_context)

  def _initialize(self):
    pass

  def _train(self, experience, weights=None):
    experience = self._as_trajectory(experience)

    # Add a mask to ensure we reset the return calculation at episode
    # boundaries. This is needed in cases where episodes are truncated before
    # reaching a terminal state. Note experience is a batch of trajectories
    # where reward=next_step.reward so the mask may look shifted at first.
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

    with tf.GradientTape() as tape:
      loss_info = self.total_loss(
          experience, tf.stop_gradient(returns), weights=weights,
          training=True)
      tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
    variables_to_train = self._actor_network.trainable_weights
    if self._baseline:
      variables_to_train += self._value_network.trainable_weights
    grads = tape.gradient(loss_info.loss, variables_to_train)

    grads_and_vars = list(zip(grads, variables_to_train))
    if self._gradient_clipping:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)

    return tf.nest.map_structure(tf.identity, loss_info)

  def total_loss(self,
                 experience: traj.Trajectory,
                 returns: types.Tensor,
                 weights: types.Tensor,
                 training: bool = False) -> tf_agent.LossInfo:
    # Ensure we see at least one full episode.
    time_steps = ts.TimeStep(experience.step_type,
                             tf.zeros_like(experience.reward),
                             tf.zeros_like(experience.discount),
                             experience.observation)
    is_last = experience.is_last()
    num_episodes = tf.reduce_sum(tf.cast(is_last, tf.float32))
    tf.debugging.assert_greater(
        num_episodes,
        0.0,
        message='No complete episode found. REINFORCE requires full episodes '
        'to compute losses.')

    # Mask out partial episodes at the end of each batch of time_steps.
    # NOTE: We use is_last rather than is_boundary because the last transition
    # is the transition with the last valid reward.  In other words, the
    # reward on the boundary transitions do not have valid rewards.  Since
    # REINFORCE is calculating a loss w.r.t. the returns (and not bootstrapping)
    # keeping the boundary transitions is irrelevant.
    valid_mask = tf.cast(experience.is_last(), dtype=tf.float32)
    valid_mask = tf.math.cumsum(valid_mask, axis=1, reverse=True)
    valid_mask = tf.cast(valid_mask > 0, dtype=tf.float32)
    if weights is not None:
      weights *= valid_mask
    else:
      weights = valid_mask

    advantages = returns
    value_preds = None

    if self._baseline:
      value_preds, _ = self._value_network(time_steps.observation,
                                           time_steps.step_type,
                                           training=True)
      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='value_preds', data=value_preds, step=self.train_step_counter)

    advantages = self._advantage_fn(returns, value_preds)
    if self._debug_summaries:
      tf.compat.v2.summary.histogram(
          name='advantages', data=advantages, step=self.train_step_counter)

    # TODO(b/126592060): replace with tensor normalizer.
    if self._normalize_returns:
      advantages = _standard_normalize(advantages, axes=(0, 1))
      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='normalized_%s' %
            ('advantages' if self._baseline else 'returns'),
            data=advantages,
            step=self.train_step_counter)

    nest_utils.assert_same_structure(time_steps, self.time_step_spec)
    policy_state = _get_initial_policy_state(self.collect_policy, time_steps)
    actions_distribution = self.collect_policy.distribution(
        time_steps, policy_state=policy_state).action

    policy_gradient_loss = self.policy_gradient_loss(
        actions_distribution,
        experience.action,
        experience.is_boundary(),
        advantages,
        num_episodes,
        weights,
    )

    entropy_regularization_loss = self.entropy_regularization_loss(
        actions_distribution, weights)

    network_regularization_loss = tf.nn.scale_regularization_loss(
        self._actor_network.losses)

    total_loss = (policy_gradient_loss +
                  network_regularization_loss +
                  entropy_regularization_loss)

    losses_dict = {
        'policy_gradient_loss': policy_gradient_loss,
        'policy_network_regularization_loss': network_regularization_loss,
        'entropy_regularization_loss': entropy_regularization_loss,
        'value_estimation_loss': 0.0,
        'value_network_regularization_loss': 0.0,
    }

    value_estimation_loss = None
    if self._baseline:
      value_estimation_loss = self.value_estimation_loss(
          value_preds, returns, num_episodes, weights)
      value_network_regularization_loss = tf.nn.scale_regularization_loss(
          self._value_network.losses)
      total_loss += value_estimation_loss + value_network_regularization_loss
      losses_dict['value_estimation_loss'] = value_estimation_loss
      losses_dict['value_network_regularization_loss'] = (
          value_network_regularization_loss)

    loss_info_extra = ReinforceAgentLossInfo(**losses_dict)

    losses_dict['total_loss'] = total_loss  # Total loss not in loss_info_extra.

    common.summarize_scalar_dict(losses_dict,
                                 self.train_step_counter,
                                 name_scope='Losses/')

    return tf_agent.LossInfo(total_loss, loss_info_extra)

  def policy_gradient_loss(
      self,
      actions_distribution: types.NestedDistribution,
      actions: types.NestedTensor,
      is_boundary: types.Tensor,
      returns: types.Tensor,
      num_episodes: types.Int,
      weights: Optional[types.Tensor] = None) -> types.Tensor:
    """Computes the policy gradient loss.

    Args:
      actions_distribution: A possibly batched tuple of action distributions.
      actions: Tensor with a batch of actions.
      is_boundary: Tensor of booleans that indicate if the corresponding action
        was in a boundary trajectory and should be ignored.
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
    valid_mask = tf.cast(~is_boundary, tf.float32)
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

  def entropy_regularization_loss(
      self,
      actions_distribution: types.NestedDistribution,
      weights: Optional[types.Tensor] = None) -> types.Tensor:
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
      loss = _entropy_loss(actions_distribution, self.action_spec, weights)
      loss *= self._entropy_regularization
    else:
      loss = tf.constant(0.0, dtype=tf.float32)

    return loss

  def value_estimation_loss(
      self,
      value_preds: types.Tensor,
      returns: types.Tensor,
      num_episodes: types.Int,
      weights: Optional[types.Tensor] = None) -> types.Tensor:
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
