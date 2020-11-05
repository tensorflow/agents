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

# Lint as: python2, python3
"""A PPO Agent.

Implements the PPO algorithm from (Schulman, 2017):
https://arxiv.org/abs/1707.06347

If you do not rely on using a combination of KL penalty and importance ratio
clipping, use `PPOClipAgent` or `PPOKLPenaltyAgent` instead.

PPO is a simplification of the TRPO algorithm, both of which add stability to
policy gradient RL, while allowing multiple updates per batch of on-policy data,
by limiting the KL divergence between the policy that sampled the data and the
updated policy.

TRPO enforces a hard optimization constraint, but is a complex algorithm, which
often makes it harder to use in practice. PPO approximates the effect of TRPO
by using a soft constraint. There are two methods presented in the paper for
implementing the soft constraint: an adaptive KL loss penalty, and
limiting the objective value based on a clipped version of the policy importance
ratio. This code implements both, and allows the user to use either method or
both by modifying hyperparameters.

The importance ratio clipping is described in eq (7) and the adaptive KL penatly
is described in eq (8) of https://arxiv.org/pdf/1707.06347.pdf
- To disable IR clipping, set the importance_ratio_clipping parameter to 0.0
- To disable the adaptive KL penalty, set the initial_adaptive_kl_beta parameter
  to 0.0
- To disable the fixed KL cutoff penalty, set the kl_cutoff_factor parameter
  to 0.0

In order to compute KL divergence, the replay buffer must store action
distribution parameters from data collection. For now, it is assumed that
continuous actions are represented by a Normal distribution with mean & stddev,
and discrete actions are represented by a Categorical distribution with logits.

Note that the objective function chooses the lower value of the clipped and
unclipped objectives. Thus, if the importance ratio exceeds the clipped bounds,
then the optimizer will still not be incentivized to pass the bounds, as it is
only optimizing the minimum.

Advantage is computed using Generalized Advantage Estimation (GAE):
https://arxiv.org/abs/1506.02438
"""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import Optional, Text, Tuple

from absl import logging

import gin
from six.moves import range
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_policy
from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity
from tf_agents.utils import tensor_normalizer
from tf_agents.utils import value_ops


PPOLossInfo = collections.namedtuple('PPOLossInfo', (
    'policy_gradient_loss',
    'value_estimation_loss',
    'l2_regularization_loss',
    'entropy_regularization_loss',
    'kl_penalty_loss',
))


@gin.configurable
class PPOAgent(tf_agent.TFAgent):
  """A PPO Agent."""

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      optimizer: Optional[types.Optimizer] = None,
      actor_net: Optional[network.Network] = None,
      value_net: Optional[network.Network] = None,
      importance_ratio_clipping: types.Float = 0.0,
      lambda_value: types.Float = 0.95,
      discount_factor: types.Float = 0.99,
      entropy_regularization: types.Float = 0.0,
      policy_l2_reg: types.Float = 0.0,
      value_function_l2_reg: types.Float = 0.0,
      shared_vars_l2_reg: types.Float = 0.0,
      value_pred_loss_coef: types.Float = 0.5,
      num_epochs: int = 25,
      use_gae: bool = False,
      use_td_lambda_return: bool = False,
      normalize_rewards: bool = True,
      reward_norm_clipping: types.Float = 10.0,
      normalize_observations: bool = True,
      log_prob_clipping: types.Float = 0.0,
      kl_cutoff_factor: types.Float = 2.0,
      kl_cutoff_coef: types.Float = 1000.0,
      initial_adaptive_kl_beta: types.Float = 1.0,
      adaptive_kl_target: types.Float = 0.01,
      adaptive_kl_tolerance: types.Float = 0.3,
      gradient_clipping: Optional[types.Float] = None,
      value_clipping: Optional[types.Float] = None,
      check_numerics: bool = False,
      # TODO(b/150244758): Change the default to False once we move
      # clients onto Reverb.
      compute_value_and_advantage_in_train: bool = True,
      update_normalizers_in_train: bool = True,
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      train_step_counter: Optional[tf.Variable] = None,
      name: Optional[Text] = None):
    """Creates a PPO Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      optimizer: Optimizer to use for the agent, default to using
        `tf.compat.v1.train.AdamOptimizer`.
      actor_net: A `network.DistributionNetwork` which maps observations to
        action distributions. Commonly, it is set to
        `actor_distribution_network.ActorDistributionNetwork`.
      value_net: A `Network` which returns the value prediction for input
        states, with `call(observation, step_type, network_state)`. Commonly, it
        is set to `value_network.ValueNetwork`.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation. Default to `0.99`
        which is the value used for all environments from (Schulman, 2017).
      entropy_regularization: Coefficient for entropy regularization loss term.
        Default to `0.0` because no entropy bonus was used in (Schulman, 2017).
      policy_l2_reg: Coefficient for L2 regularization of unshared actor_net
        weights. Default to `0.0` because no L2 regularization was applied on
        the policy network weights in (Schulman, 2017).
      value_function_l2_reg: Coefficient for l2 regularization of unshared value
        function weights. Default to `0.0` because no L2 regularization was
        applied on the policy network weights in (Schulman, 2017).
      shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
        between actor_net and value_net. Default to `0.0` because no L2
        regularization was applied on the policy network or value network
        weights in (Schulman, 2017).
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss. Default to `0.5`, which was used for all
        environments in the OpenAI baseline implementation. This parameters is
        irrelevant unless you are sharing part of actor_net and value_net. In
        that case, you would want to tune this coeeficient, whose value depends
        on the network architecture of your choice.
      num_epochs: Number of epochs for computing policy updates. (Schulman,2017)
        sets this to 10 for Mujoco, 15 for Roboschool and 3 for Atari.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function; here:
        `td_lambda_return = gae_advantage + value_predictions`.
        `use_gae` must be set to `True` as well to enable TD -lambda returns. If
        `use_td_lambda_return` is set to True while `use_gae` is False, the
        empirical return will be used and a warning will be logged.
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards. While not mentioned directly in (Schulman,
        2017), reward normalization was implemented in OpenAI baselines and
        (Ilyas et al., 2018) pointed out that it largely improves performance.
        You may refer to Figure 1 of https://arxiv.org/pdf/1811.02553.pdf for a
        comparison with and without reward scaling.
      reward_norm_clipping: Value above and below to clip normalized reward.
        Additional optimization proposed in (Ilyas et al., 2018) set to
        `5` or `10`.
      normalize_observations: If `True`, keeps moving mean and
        variance of observations and normalizes incoming observations.
        Additional optimization proposed in (Ilyas et al., 2018). If true, and
        the observation spec is not tf.float32 (such as Atari), please manually
        convert the observation spec received from the environment to tf.float32
        before creating the networks. Otherwise, the normalized input to the
        network (float32) will have a different dtype as what the network
        expects, resulting in a mismatch error.

        Example usage:
          ```python
          observation_tensor_spec, action_spec, time_step_tensor_spec = (
            spec_utils.get_tensor_specs(env))
          normalized_observation_tensor_spec = tf.nest.map_structure(
            lambda s: tf.TensorSpec(
              dtype=tf.float32, shape=s.shape, name=s.name
            ),
            observation_tensor_spec
          )

          actor_net = actor_distribution_network.ActorDistributionNetwork(
            normalized_observation_tensor_spec, ...)
          value_net = value_network.ValueNetwork(
            normalized_observation_tensor_spec, ...)
          # Note that the agent still uses the original time_step_tensor_spec
          # from the environment.
          agent = ppo_clip_agent.PPOClipAgent(
            time_step_tensor_spec, action_spec, actor_net, value_net, ...)
          ```
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      kl_cutoff_factor: Only meaningful when `kl_cutoff_coef > 0.0`. A multipler
        used for calculating the KL cutoff ( =
        `kl_cutoff_factor * adaptive_kl_target`). If policy KL averaged across
        the batch changes more than the cutoff, a squared cutoff loss would
        be added to the loss function.
      kl_cutoff_coef: kl_cutoff_coef and kl_cutoff_factor are additional params
        if one wants to use a KL cutoff loss term in addition to the adaptive KL
        loss term. Default to 0.0 to disable the KL cutoff loss term as this was
        not used in the paper.  kl_cutoff_coef is the coefficient to mulitply by
        the KL cutoff loss term, before adding to the total loss function.
      initial_adaptive_kl_beta: Initial value for beta coefficient of adaptive
        KL penalty. This initial value is not important in practice because the
        algorithm quickly adjusts to it. A common default is 1.0.
      adaptive_kl_target: Desired KL target for policy updates. If actual KL is
        far from this target, adaptive_kl_beta will be updated. You should tune
        this for your environment. 0.01 was found to perform well for Mujoco.
      adaptive_kl_tolerance: A tolerance for adaptive_kl_beta. Mean KL above
        `(1 + tol) * adaptive_kl_target`, or below
        `(1 - tol) * adaptive_kl_target`,
        will cause `adaptive_kl_beta` to be updated. `0.5` was chosen
        heuristically in the paper, but the algorithm is not very
        sensitive to it.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      value_clipping: Difference between new and old value predictions are
        clipped to this threshold. Value clipping could be helpful when training
        very deep networks. Default: no clipping.
      check_numerics: If true, adds `tf.debugging.check_numerics` to help find
        NaN / Inf values. For debugging only.
      compute_value_and_advantage_in_train: A bool to indicate where value
        prediction and advantage calculation happen.  If True, both happen in
        agent.train(). If False, value prediction is computed during data
        collection. This argument must be set to `False` if mini batch learning
        is enabled.
      update_normalizers_in_train: A bool to indicate whether normalizers are
        updated as parts of the `train` method. Set to `False` if mini batch
        learning is enabled, or if `train` is called on multiple iterations of
        the same trajectories. In that case, you would need to use `PPOLearner`
        (which updates all the normalizers outside of the agent). This ensures
        that normalizers are updated in the same way as (Schulman, 2017).
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      TypeError: if `actor_net` or `value_net` is not of type
        `tf_agents.networks.Network`.
    """
    if not isinstance(actor_net, network.Network):
      raise TypeError(
          'actor_net must be an instance of a network.Network.')
    if not isinstance(value_net, network.Network):
      raise TypeError('value_net must be an instance of a network.Network.')

    # PPOPolicy validates these, so we skip validation here.
    actor_net.create_variables(time_step_spec.observation)
    value_net.create_variables(time_step_spec.observation)

    tf.Module.__init__(self, name=name)

    self._optimizer = optimizer
    self._actor_net = actor_net
    self._value_net = value_net
    self._importance_ratio_clipping = importance_ratio_clipping
    self._lambda = lambda_value
    self._discount_factor = discount_factor
    self._entropy_regularization = entropy_regularization
    self._policy_l2_reg = policy_l2_reg
    self._value_function_l2_reg = value_function_l2_reg
    self._shared_vars_l2_reg = shared_vars_l2_reg
    self._value_pred_loss_coef = value_pred_loss_coef
    self._num_epochs = num_epochs
    self._use_gae = use_gae
    self._use_td_lambda_return = use_td_lambda_return
    self._reward_norm_clipping = reward_norm_clipping
    self._log_prob_clipping = log_prob_clipping
    self._kl_cutoff_factor = kl_cutoff_factor
    self._kl_cutoff_coef = kl_cutoff_coef
    self._adaptive_kl_target = adaptive_kl_target
    self._adaptive_kl_tolerance = adaptive_kl_tolerance
    self._gradient_clipping = gradient_clipping or 0.0
    self._value_clipping = value_clipping or 0.0
    self._check_numerics = check_numerics
    self._compute_value_and_advantage_in_train = (
        compute_value_and_advantage_in_train)
    self.update_normalizers_in_train = update_normalizers_in_train
    if not isinstance(self._optimizer, tf.keras.optimizers.Optimizer):
      logging.warning(
          'Only tf.keras.optimizers.Optimiers are well supported, got a '
          'non-TF2 optimizer: %s', self._optimizer)

    self._initial_adaptive_kl_beta = initial_adaptive_kl_beta
    if initial_adaptive_kl_beta > 0.0:
      # TODO(kbanoop): Rename create_variable.
      self._adaptive_kl_beta = common.create_variable(
          'adaptive_kl_beta', initial_adaptive_kl_beta, dtype=tf.float32)
    else:
      self._adaptive_kl_beta = None

    self._reward_normalizer = None
    if normalize_rewards:
      self._reward_normalizer = tensor_normalizer.StreamingTensorNormalizer(
          tensor_spec.TensorSpec([], tf.float32), scope='normalize_reward')

    self._observation_normalizer = None
    if normalize_observations:
      self._observation_normalizer = (
          tensor_normalizer.StreamingTensorNormalizer(
              time_step_spec.observation, scope='normalize_observations'))

    self._advantage_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec.TensorSpec([], tf.float32), scope='normalize_advantages')

    policy = greedy_policy.GreedyPolicy(
        ppo_policy.PPOPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
            observation_normalizer=self._observation_normalizer,
            clip=False,
            collect=False))

    collect_policy = ppo_policy.PPOPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        value_network=value_net,
        observation_normalizer=self._observation_normalizer,
        clip=False,
        collect=True,
        compute_value_and_advantage_in_train=(
            self._compute_value_and_advantage_in_train),
    )

    if isinstance(self._actor_net, network.DistributionNetwork):
      # Legacy behavior
      self._action_distribution_spec = self._actor_net.output_spec
    else:
      self._action_distribution_spec = self._actor_net.create_variables(
          time_step_spec.observation)

    # Set training_data_spec to collect_data_spec with augmented policy info,
    # iff return and normalized advantage are saved in preprocess_sequence.
    if self._compute_value_and_advantage_in_train:
      training_data_spec = None
    else:
      training_policy_info = collect_policy.trajectory_spec.policy_info.copy()
      training_policy_info.update({
          'value_prediction':
              collect_policy.trajectory_spec.policy_info['value_prediction'],
          'return':
              tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
          'advantage':
              tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
      })
      training_data_spec = collect_policy.trajectory_spec.replace(
          policy_info=training_policy_info)

    super(PPOAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        training_data_spec=training_data_spec,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        validate_args=False)

    # This must be built after super() which sets up self.data_context.
    self._collected_as_transition = data_converter.AsTransition(
        self.collect_data_context, squeeze_time_dim=False)

    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)

  @property
  def actor_net(self) -> network.Network:
    """Returns actor_net TensorFlow template function."""
    return self._actor_net

  def _initialize(self):
    pass

  def compute_advantages(self,
                         rewards: types.NestedTensor,
                         returns: types.Tensor,
                         discounts: types.Tensor,
                         value_preds: types.Tensor) -> types.Tensor:
    """Compute advantages, optionally using GAE.

    Based on baselines ppo1 implementation. Removes final timestep, as it needs
    to use this timestep for next-step value prediction for TD error
    computation.

    Args:
      rewards: Tensor of per-timestep rewards.
      returns: Tensor of per-timestep returns.
      discounts: Tensor of per-timestep discounts. Zero for terminal timesteps.
      value_preds: Cached value estimates from the data-collection policy.

    Returns:
      advantages: Tensor of length (len(rewards) - 1), because the final
        timestep is just used for next-step value prediction.
    """
    # Arg value_preds was appended with final next_step value. Make tensors
    #   next_value_preds by stripping first and last elements respectively.
    final_value_pred = value_preds[:, -1]
    value_preds = value_preds[:, :-1]

    if not self._use_gae:
      with tf.name_scope('empirical_advantage'):
        advantages = returns - value_preds
    else:
      advantages = value_ops.generalized_advantage_estimation(
          values=value_preds,
          final_value=final_value_pred,
          rewards=rewards,
          discounts=discounts,
          td_lambda=self._lambda,
          time_major=False)

    return advantages

  def get_loss(self,
               time_steps: ts.TimeStep,
               actions: types.NestedTensorSpec,
               act_log_probs: types.Tensor,
               returns: types.Tensor,
               normalized_advantages: types.Tensor,
               action_distribution_parameters: types.NestedTensor,
               weights: types.Tensor,
               train_step: tf.Variable,
               debug_summaries: bool,
               old_value_predictions: Optional[types.Tensor] = None,
               training: bool = False) -> tf_agent.LossInfo:
    """Compute the loss and create optimization op for one training epoch.

    All tensors should have a single batch dimension.

    Args:
      time_steps: A minibatch of TimeStep tuples.
      actions: A minibatch of actions.
      act_log_probs: A minibatch of action probabilities (probability under the
        sampling policy).
      returns: A minibatch of per-timestep returns.
      normalized_advantages: A minibatch of normalized per-timestep advantages.
      action_distribution_parameters: Parameters of data-collecting action
        distribution. Needed for KL computation.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      train_step: A train_step variable to increment for each train step.
        Typically the global_step.
      debug_summaries: True if debug summaries should be created.
      old_value_predictions: (Optional) The saved value predictions, used
        for calculating the value estimation loss when value clipping is
        performed.
      training: Whether this loss is being used for training.

    Returns:
      A tf_agent.LossInfo named tuple with the total_loss and all intermediate
        losses in the extra field contained in a PPOLossInfo named tuple.
    """
    # Evaluate the current policy on timesteps.

    # batch_size from time_steps
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._collect_policy.get_initial_state(batch_size)
    # We must use _distribution because the distribution API doesn't pass down
    # the training= kwarg.
    distribution_step = self._collect_policy._distribution(  # pylint: disable=protected-access
        time_steps,
        policy_state,
        training=training)
    # TODO(eholly): Rename policy distributions to something clear and uniform.
    current_policy_distribution = distribution_step.action

    # Call all loss functions and add all loss values.
    value_estimation_loss = self.value_estimation_loss(
        time_steps=time_steps,
        returns=returns,
        old_value_predictions=old_value_predictions,
        weights=weights,
        debug_summaries=debug_summaries,
        training=training)
    policy_gradient_loss = self.policy_gradient_loss(
        time_steps,
        actions,
        tf.stop_gradient(act_log_probs),
        tf.stop_gradient(normalized_advantages),
        current_policy_distribution,
        weights,
        debug_summaries=debug_summaries)

    if (self._policy_l2_reg > 0.0 or self._value_function_l2_reg > 0.0 or
        self._shared_vars_l2_reg > 0.0):
      l2_regularization_loss = self.l2_regularization_loss(debug_summaries)
    else:
      l2_regularization_loss = tf.zeros_like(policy_gradient_loss)

    if self._entropy_regularization > 0.0:
      entropy_regularization_loss = self.entropy_regularization_loss(
          time_steps, current_policy_distribution, weights, debug_summaries)
    else:
      entropy_regularization_loss = tf.zeros_like(policy_gradient_loss)

    # TODO(b/1613650790: Move this logic to PPOKLPenaltyAgent.
    if self._initial_adaptive_kl_beta == 0:
      kl_penalty_loss = tf.zeros_like(policy_gradient_loss)
    else:
      kl_penalty_loss = self.kl_penalty_loss(time_steps,
                                             action_distribution_parameters,
                                             current_policy_distribution,
                                             weights, debug_summaries)

    total_loss = (
        policy_gradient_loss + value_estimation_loss + l2_regularization_loss +
        entropy_regularization_loss + kl_penalty_loss)

    return tf_agent.LossInfo(
        total_loss,
        PPOLossInfo(
            policy_gradient_loss=policy_gradient_loss,
            value_estimation_loss=value_estimation_loss,
            l2_regularization_loss=l2_regularization_loss,
            entropy_regularization_loss=entropy_regularization_loss,
            kl_penalty_loss=kl_penalty_loss,
        ))

  def compute_return_and_advantage(
      self,
      next_time_steps: ts.TimeStep,
      value_preds: types.Tensor) -> Tuple[types.Tensor, types.Tensor]:
    """Compute the Monte Carlo return and advantage.

    Args:
      next_time_steps: batched tensor of TimeStep tuples after action is taken.
      value_preds: Batched value prediction tensor. Should have one more entry
        in time index than time_steps, with the final value corresponding to the
        value prediction of the final state.

    Returns:
      tuple of (return, advantage), both are batched tensors.
    """
    discounts = next_time_steps.discount * tf.constant(
        self._discount_factor, dtype=tf.float32)

    rewards = next_time_steps.reward
    if self._debug_summaries:
      # Summarize rewards before they get normalized below.
      # TODO(b/171573175): remove the condition once histograms are
      # supported on TPUs.
      if not tf.config.list_logical_devices('TPU'):
        tf.compat.v2.summary.histogram(
            name='rewards', data=rewards, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='rewards_mean',
          data=tf.reduce_mean(rewards),
          step=self.train_step_counter)

    # Normalize rewards if self._reward_normalizer is defined.
    if self._reward_normalizer:
      rewards = self._reward_normalizer.normalize(
          rewards, center_mean=False, clip_value=self._reward_norm_clipping)
      if self._debug_summaries:
        # TODO(b/171573175): remove the condition once histograms are
        # supported on TPUs.
        if not tf.config.list_logical_devices('TPU'):
          tf.compat.v2.summary.histogram(
              name='rewards_normalized',
              data=rewards,
              step=self.train_step_counter)
        tf.compat.v2.summary.scalar(
            name='rewards_normalized_mean',
            data=tf.reduce_mean(rewards),
            step=self.train_step_counter)

    # Make discount 0.0 at end of each episode to restart cumulative sum
    #   end of each episode.
    episode_mask = common.get_episode_mask(next_time_steps)
    discounts *= episode_mask

    # Compute Monte Carlo returns. Data from incomplete trajectories, not
    #   containing the end of an episode will also be used, with a bootstrapped
    #   estimation from the last value.
    # Note that when a trajectory driver is used, then the final step is
    #   terminal, the bootstrapped estimation will not be used, as it will be
    #   multiplied by zero (the discount on the last step).
    final_value_bootstrapped = value_preds[:, -1]
    returns = value_ops.discounted_return(
        rewards,
        discounts,
        time_major=False,
        final_value=final_value_bootstrapped)
    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not tf.config.list_logical_devices('TPU'):
      tf.compat.v2.summary.histogram(
          name='returns', data=returns, step=self.train_step_counter)

    # Compute advantages.
    advantages = self.compute_advantages(rewards, returns, discounts,
                                         value_preds)

    # TODO(b/171573175): remove the condition once historgrams are
    # supported on TPUs.
    if self._debug_summaries and not tf.config.list_logical_devices('TPU'):
      tf.compat.v2.summary.histogram(
          name='advantages', data=advantages, step=self.train_step_counter)

    # Return TD-Lambda returns if both use_td_lambda_return and use_gae.
    if self._use_td_lambda_return:
      if not self._use_gae:
        logging.warning('use_td_lambda_return was True, but use_gae was '
                        'False. Using Monte Carlo return.')
      else:
        returns = tf.add(
            advantages, value_preds[:, :-1], name='td_lambda_returns')

    return returns, advantages

  def _preprocess(self, experience):
    """Performs advantage calculation for the collected experience.

    Args:
      experience: A (batch of) experience in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.collect_data_spec`.
        All tensors in `experience` must be shaped `[batch, time + 1, ...]` or
        [time + 1, ...]. The "+1" is needed as the last action from the set of
        trajectories cannot be used for training, as its advantage and returns
        are unknown.

    Returns:
      The processed experience which has normalized_advantages and returns
      filled in its policy info. The advantages and returns for the last
      transition is filled with 0s as they cannot be calculated.
    """
    # Try to be agnostic about the input type of experience before we call
    # to_transition() below.
    outer_rank = nest_utils.get_outer_rank(
        _get_discount(experience), self.collect_data_spec.discount)

    # Add 1 as the batch dimension for inputs that just have the time dimension,
    # as all utility functions below require the batch dimension.
    if outer_rank == 1:
      batched_experience = nest_utils.batch_nested_tensors(experience)
    else:
      batched_experience = experience

    # Get individual tensors from experience.
    num_steps = _get_discount(batched_experience).shape[1]
    if num_steps and num_steps <= 1:
      raise ValueError(
          'Experience used for advantage calculation must have >1 num_steps.')

    transition = self._collected_as_transition(batched_experience)
    time_steps, _, next_time_steps = transition

    # TODO(b/170680358): Decide if we will require Trajectory for preprocess,
    # or if we want to handle transitions here too; if so, what do we return?
    # Below we use batched_experience assuming it's a Trajectory, and return
    # a trajectory.

    # Compute the value predictions for states using the current value function.
    # To be used for return & advantage computation.
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    if self._compute_value_and_advantage_in_train:
      value_state = self._collect_policy.get_initial_value_state(batch_size)
      value_preds, _ = self._collect_policy.apply_value_network(
          batched_experience.observation,
          batched_experience.step_type,
          value_state=value_state,
          training=False)
      value_preds = tf.stop_gradient(value_preds)
    else:
      value_preds = batched_experience.policy_info['value_prediction']

    new_policy_info = {
        'dist_params': batched_experience.policy_info['dist_params'],
        'value_prediction': value_preds,
    }

    # Add the calculated advantage and return into the input experience.
    returns, advantages = self.compute_return_and_advantage(
        next_time_steps, value_preds)

    # Pad returns and normalized_advantages in the time dimension so that the
    # time dimensions are aligned with the input experience's time dimension.
    # When the output trajectory gets sliced by trajectory.to_transition during
    # training, the padded last timesteps will be automatically dropped.
    last_transition_padding = tf.zeros((batch_size, 1), dtype=tf.float32)
    new_policy_info['return'] = tf.concat([returns, last_transition_padding],
                                          axis=1)
    new_policy_info['advantage'] = tf.concat(
        [advantages, last_transition_padding], axis=1)

    # Remove the batch dimension iff the input experience does not have it.
    if outer_rank == 1:
      new_policy_info = nest_utils.unbatch_nested_tensors(new_policy_info)
    # The input experience with its policy info filled with the calculated
    # advantages and returns for each action.
    return experience.replace(policy_info=new_policy_info)

  def _preprocess_sequence(self, experience):
    """Performs advantage calculation for the collected experience.

    This function is a no-op if self._compute_value_and_advantage_in_train is
    True, which means advantage calculation happens as part of agent.train().

    Args:
      experience: A (batch of) experience in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.collect_data_spec`.
        All tensors in `experience` must be shaped `[batch, time + 1, ...]` or
        [time + 1, ...]. The "+1" is needed as the last action from the set of
        trajectories cannot be used for training, as its advantage and returns
        are unknown.

    Returns:
      A post processed `Trajectory` with the same shape as the input, with
        `return` and `normalized_advantage` stored inside of the policy info
        dictionary. The advantages and returns for the last transition is filled
        with 0s as they cannot be calculated.
    """
    if self._compute_value_and_advantage_in_train:
      return experience

    return self._preprocess(experience)

  def _train(self, experience, weights):
    experience = self._as_trajectory(experience)

    if self._compute_value_and_advantage_in_train:
      processed_experience = self._preprocess(experience)
    else:
      processed_experience = experience

    # Mask trajectories that cannot be used for training.
    valid_mask = ppo_utils.make_trajectory_mask(processed_experience)
    if weights is None:
      masked_weights = valid_mask
    else:
      masked_weights = weights * valid_mask

    # Reconstruct per-timestep policy distribution from stored distribution
    #   parameters.
    old_action_distribution_parameters = processed_experience.policy_info[
        'dist_params']

    old_actions_distribution = (
        ppo_utils.distribution_from_spec(
            self._action_distribution_spec,
            old_action_distribution_parameters,
            legacy_distribution_network=isinstance(
                self._actor_net, network.DistributionNetwork)))

    # Compute log probability of actions taken during data collection, using the
    #   collect policy distribution.
    old_act_log_probs = common.log_probability(old_actions_distribution,
                                               processed_experience.action,
                                               self._action_spec)

    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not tf.config.list_logical_devices('TPU'):
      actions_list = tf.nest.flatten(processed_experience.action)
      show_action_index = len(actions_list) != 1
      for i, single_action in enumerate(actions_list):
        action_name = ('actions_{}'.format(i)
                       if show_action_index else 'actions')
        tf.compat.v2.summary.histogram(
            name=action_name, data=single_action, step=self.train_step_counter)

    time_steps = ts.TimeStep(
        step_type=processed_experience.step_type,
        reward=processed_experience.reward,
        discount=processed_experience.discount,
        observation=processed_experience.observation)
    actions = processed_experience.action
    returns = processed_experience.policy_info['return']

    if self.update_normalizers_in_train:
      self._reset_advantage_normalizer()
      self._update_advantage_normalizer(
          processed_experience.policy_info['advantage'])
    normalized_advantages = self._advantage_normalizer.normalize(
        processed_experience.policy_info['advantage'],
        clip_value=0,
        center_mean=True,
        variance_epsilon=1e-8)
    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not tf.config.list_logical_devices('TPU'):
      tf.compat.v2.summary.histogram(
          name='advantages_normalized',
          data=normalized_advantages,
          step=self.train_step_counter)
    old_value_predictions = processed_experience.policy_info['value_prediction']

    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    # Loss tensors across batches will be aggregated for summaries.
    policy_gradient_losses = []
    value_estimation_losses = []
    l2_regularization_losses = []
    entropy_regularization_losses = []
    kl_penalty_losses = []

    loss_info = None  # TODO(b/123627451): Remove.
    variables_to_train = list(
        object_identity.ObjectIdentitySet(self._actor_net.trainable_weights +
                                          self._value_net.trainable_weights))
    # Sort to ensure tensors on different processes end up in same order.
    variables_to_train = sorted(variables_to_train, key=lambda x: x.name)

    for i_epoch in range(self._num_epochs):
      with tf.name_scope('epoch_%d' % i_epoch):
        # Only save debug summaries for first and last epochs.
        debug_summaries = (
            self._debug_summaries and
            (i_epoch == 0 or i_epoch == self._num_epochs - 1))

        with tf.GradientTape() as tape:
          loss_info = self.get_loss(
              time_steps,
              actions,
              old_act_log_probs,
              returns,
              normalized_advantages,
              old_action_distribution_parameters,
              masked_weights,
              self.train_step_counter,
              debug_summaries,
              old_value_predictions=old_value_predictions,
              training=True)

        grads = tape.gradient(loss_info.loss, variables_to_train)
        if self._gradient_clipping > 0:
          grads, _ = tf.clip_by_global_norm(grads, self._gradient_clipping)

        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = tuple(zip(grads, variables_to_train))

        # If summarize_gradients, create functions for summarizing both
        # gradients and variables.
        if self._summarize_grads_and_vars and debug_summaries:
          eager_utils.add_gradients_summaries(grads_and_vars,
                                              self.train_step_counter)
          eager_utils.add_variables_summaries(grads_and_vars,
                                              self.train_step_counter)

        self._optimizer.apply_gradients(grads_and_vars)
        self.train_step_counter.assign_add(1)

        policy_gradient_losses.append(loss_info.extra.policy_gradient_loss)
        value_estimation_losses.append(loss_info.extra.value_estimation_loss)
        l2_regularization_losses.append(loss_info.extra.l2_regularization_loss)
        entropy_regularization_losses.append(
            loss_info.extra.entropy_regularization_loss)
        kl_penalty_losses.append(loss_info.extra.kl_penalty_loss)

    # TODO(b/1613650790: Move this logic to PPOKLPenaltyAgent.
    if self._initial_adaptive_kl_beta > 0:
      # After update epochs, update adaptive kl beta, then update observation
      #   normalizer and reward normalizer.
      policy_state = self._collect_policy.get_initial_state(batch_size)
      # Compute the mean kl from previous action distribution.
      kl_divergence = self._kl_divergence(
          time_steps, old_action_distribution_parameters,
          self._collect_policy.distribution(time_steps, policy_state).action)
      self.update_adaptive_kl_beta(kl_divergence)

    if self.update_normalizers_in_train:
      self.update_observation_normalizer(time_steps.observation)
      self.update_reward_normalizer(processed_experience.reward)

    loss_info = tf.nest.map_structure(tf.identity, loss_info)

    # Make summaries for total loss averaged across all epochs.
    # The *_losses lists will have been populated by
    #   calls to self.get_loss. Assumes all the losses have same length.
    with tf.name_scope('Losses/'):
      num_epochs = len(policy_gradient_losses)
      total_policy_gradient_loss = tf.add_n(policy_gradient_losses) / num_epochs
      total_value_estimation_loss = tf.add_n(
          value_estimation_losses) / num_epochs
      total_l2_regularization_loss = tf.add_n(
          l2_regularization_losses) / num_epochs
      total_entropy_regularization_loss = tf.add_n(
          entropy_regularization_losses) / num_epochs
      total_kl_penalty_loss = tf.add_n(kl_penalty_losses) / num_epochs
      tf.compat.v2.summary.scalar(
          name='policy_gradient_loss',
          data=total_policy_gradient_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=total_value_estimation_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='l2_regularization_loss',
          data=total_l2_regularization_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='entropy_regularization_loss',
          data=total_entropy_regularization_loss,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='kl_penalty_loss',
          data=total_kl_penalty_loss,
          step=self.train_step_counter)

      total_abs_loss = (
          tf.abs(total_policy_gradient_loss) +
          tf.abs(total_value_estimation_loss) +
          tf.abs(total_entropy_regularization_loss) +
          tf.abs(total_l2_regularization_loss) + tf.abs(total_kl_penalty_loss))

      tf.compat.v2.summary.scalar(
          name='total_abs_loss',
          data=total_abs_loss,
          step=self.train_step_counter)

    with tf.name_scope('LearningRate/'):
      learning_rate = ppo_utils.get_learning_rate(self._optimizer)
      tf.compat.v2.summary.scalar(
          name='learning_rate',
          data=learning_rate,
          step=self.train_step_counter)

    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._summarize_grads_and_vars and not tf.config.list_logical_devices(
        'TPU'):
      with tf.name_scope('Variables/'):
        all_vars = (
            self._actor_net.trainable_weights +
            self._value_net.trainable_weights)
        for var in all_vars:
          tf.compat.v2.summary.histogram(
              name=var.name.replace(':', '_'),
              data=var,
              step=self.train_step_counter)

    return loss_info

  def update_observation_normalizer(self, batched_observations):
    if self._observation_normalizer:
      self._observation_normalizer.update(
          batched_observations, outer_dims=[0, 1])

  def update_reward_normalizer(self, batched_rewards):
    if self._reward_normalizer:
      self._reward_normalizer.update(batched_rewards, outer_dims=[0, 1])

  def _update_advantage_normalizer(self, batched_advantages):
    if self._advantage_normalizer:
      self._advantage_normalizer.update(batched_advantages, outer_dims=[0, 1])

  def _reset_advantage_normalizer(self):
    self._advantage_normalizer.reset()

  def l2_regularization_loss(self,
                             debug_summaries: bool = False) -> types.Tensor:
    if (self._policy_l2_reg > 0 or self._value_function_l2_reg > 0 or
        self._shared_vars_l2_reg > 0):
      with tf.name_scope('l2_regularization'):
        # Regularize policy weights.
        policy_vars_to_regularize = (
            v for v in self._actor_net.trainable_weights if 'kernel' in v.name)
        vf_vars_to_regularize = (
            v for v in self._value_net.trainable_weights if 'kernel' in v.name)

        (unshared_policy_vars_to_regularize, unshared_vf_vars_to_regularize,
         shared_vars_to_regularize) = common.extract_shared_variables(
             policy_vars_to_regularize, vf_vars_to_regularize)

        # Regularize policy weights.
        policy_l2_losses = [
            common.aggregate_losses(
                regularization_loss=tf.square(v)).regularization *
            self._policy_l2_reg for v in unshared_policy_vars_to_regularize
        ]

        # Regularize value function weights.
        vf_l2_losses = [
            common.aggregate_losses(
                regularization_loss=tf.square(v)).regularization *
            self._value_function_l2_reg for v in unshared_vf_vars_to_regularize
        ]

        # Regularize shared weights
        shared_l2_losses = [
            common.aggregate_losses(
                regularization_loss=tf.square(v)).regularization *
            self._shared_vars_l2_reg for v in shared_vars_to_regularize
        ]

        l2_losses = policy_l2_losses + vf_l2_losses + shared_l2_losses
        total_l2_loss = tf.add_n(l2_losses, name='l2_loss')

        if self._check_numerics:
          total_l2_loss = tf.debugging.check_numerics(total_l2_loss,
                                                      'total_l2_loss')

        # TODO(b/171573175): remove the condition once histograms are
        # supported on TPUs.
        if debug_summaries and not tf.config.list_logical_devices('TPU'):
          tf.compat.v2.summary.histogram(
              name='l2_loss', data=total_l2_loss, step=self.train_step_counter)
    else:
      total_l2_loss = tf.constant(0.0, dtype=tf.float32, name='zero_l2_loss')

    return total_l2_loss

  def entropy_regularization_loss(
      self,
      time_steps: ts.TimeStep,
      current_policy_distribution: types.NestedDistribution,
      weights: types.Tensor,
      debug_summaries: bool = False) -> types.Tensor:
    """Create regularization loss tensor based on agent parameters."""
    if self._entropy_regularization > 0:
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)
      with tf.name_scope('entropy_regularization'):
        entropy = tf.cast(
            common.entropy(current_policy_distribution, self.action_spec),
            tf.float32)

        entropy_reg_loss = common.aggregate_losses(
            per_example_loss=-entropy,
            sample_weight=weights).total_loss * self._entropy_regularization

        if self._check_numerics:
          entropy_reg_loss = tf.debugging.check_numerics(
              entropy_reg_loss, 'entropy_reg_loss')

        # TODO(b/171573175): remove the condition once histograms are supported
        # on TPUs.
        if debug_summaries and not tf.config.list_logical_devices('TPU'):
          tf.compat.v2.summary.histogram(
              name='entropy_reg_loss',
              data=entropy_reg_loss,
              step=self.train_step_counter)
    else:
      entropy_reg_loss = tf.constant(
          0.0, dtype=tf.float32, name='zero_entropy_reg_loss')

    return entropy_reg_loss

  def value_estimation_loss(
      self,
      time_steps: ts.TimeStep,
      returns: types.Tensor,
      weights: types.Tensor,
      old_value_predictions: Optional[types.Tensor] = None,
      debug_summaries: bool = False,
      training: bool = False) -> types.Tensor:
    """Computes the value estimation loss for actor-critic training.

    All tensors should have a single batch dimension.

    Args:
      time_steps: A batch of timesteps.
      returns: Per-timestep returns for value function to predict. (Should come
        from TD-lambda computation.)
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      old_value_predictions: (Optional) The saved value predictions from
        policy_info, required when self._value_clipping > 0.
      debug_summaries: True if debug summaries should be created.
      training: Whether this loss is going to be used for training.

    Returns:
      value_estimation_loss: A scalar value_estimation_loss loss.

    Raises:
      ValueError: If old_value_predictions was not passed in, but value clipping
        was performed.
    """

    observation = time_steps.observation
    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if debug_summaries and not tf.config.list_logical_devices('TPU'):
      observation_list = tf.nest.flatten(observation)
      show_observation_index = len(observation_list) != 1
      for i, single_observation in enumerate(observation_list):
        observation_name = ('observations_{}'.format(i)
                            if show_observation_index else 'observations')
        tf.compat.v2.summary.histogram(
            name=observation_name,
            data=single_observation,
            step=self.train_step_counter)

    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    value_state = self._collect_policy.get_initial_value_state(batch_size)

    value_preds, _ = self._collect_policy.apply_value_network(
        time_steps.observation,
        time_steps.step_type,
        value_state=value_state,
        training=training)
    value_estimation_error = tf.math.squared_difference(returns, value_preds)

    if self._value_clipping > 0:
      if old_value_predictions is None:
        raise ValueError(
            'old_value_predictions is None but needed for value clipping.')
      clipped_value_preds = old_value_predictions + tf.clip_by_value(
          value_preds - old_value_predictions, -self._value_clipping,
          self._value_clipping)
      clipped_value_estimation_error = tf.math.squared_difference(
          returns, clipped_value_preds)
      value_estimation_error = tf.maximum(value_estimation_error,
                                          clipped_value_estimation_error)

    value_estimation_loss = (
        common.aggregate_losses(
            per_example_loss=value_estimation_error,
            sample_weight=weights).total_loss * self._value_pred_loss_coef)
    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='value_pred_avg',
          data=tf.reduce_mean(input_tensor=value_preds),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_actual_avg',
          data=tf.reduce_mean(input_tensor=returns),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=value_estimation_loss,
          step=self.train_step_counter)
      # TODO(b/171573175): remove the condition once histograms are supported
      # on TPUs.
      if not tf.config.list_logical_devices('TPU'):
        tf.compat.v2.summary.histogram(
            name='value_preds', data=value_preds, step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='value_estimation_error',
            data=value_estimation_error,
            step=self.train_step_counter)

    if self._check_numerics:
      value_estimation_loss = tf.debugging.check_numerics(
          value_estimation_loss, 'value_estimation_loss')

    return value_estimation_loss

  def policy_gradient_loss(
      self,
      time_steps: ts.TimeStep,
      actions: types.NestedTensor,
      sample_action_log_probs: types.Tensor,
      advantages: types.Tensor,
      current_policy_distribution: types.NestedDistribution,
      weights: types.Tensor,
      debug_summaries: bool = False) -> types.Tensor:
    """Create tensor for policy gradient loss.

    All tensors should have a single batch dimension.

    Args:
      time_steps: TimeSteps with observations for each timestep.
      actions: Tensor of actions for timesteps, aligned on index.
      sample_action_log_probs: Tensor of sample probability of each action.
      advantages: Tensor of advantage estimate for each timestep, aligned on
        index. Works better when advantage estimates are normalized.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.

    Returns:
      policy_gradient_loss: A tensor that will contain policy gradient loss for
        the on-policy experience.
    """
    nest_utils.assert_same_structure(time_steps, self.time_step_spec)
    action_log_prob = common.log_probability(current_policy_distribution,
                                             actions, self._action_spec)
    action_log_prob = tf.cast(action_log_prob, tf.float32)
    if self._log_prob_clipping > 0.0:
      action_log_prob = tf.clip_by_value(action_log_prob,
                                         -self._log_prob_clipping,
                                         self._log_prob_clipping)
    if self._check_numerics:
      action_log_prob = tf.debugging.check_numerics(action_log_prob,
                                                    'action_log_prob')

    # Prepare both clipped and unclipped importance ratios.
    importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
    importance_ratio_clipped = tf.clip_by_value(
        importance_ratio, 1 - self._importance_ratio_clipping,
        1 + self._importance_ratio_clipping)

    if self._check_numerics:
      importance_ratio = tf.debugging.check_numerics(importance_ratio,
                                                     'importance_ratio')
      if self._importance_ratio_clipping > 0.0:
        importance_ratio_clipped = tf.debugging.check_numerics(
            importance_ratio_clipped, 'importance_ratio_clipped')

    # Pessimistically choose the minimum objective value for clipped and
    #   unclipped importance ratios.
    per_timestep_objective = importance_ratio * advantages
    per_timestep_objective_clipped = importance_ratio_clipped * advantages
    per_timestep_objective_min = tf.minimum(per_timestep_objective,
                                            per_timestep_objective_clipped)

    if self._importance_ratio_clipping > 0.0:
      policy_gradient_loss = -per_timestep_objective_min
    else:
      policy_gradient_loss = -per_timestep_objective

    policy_gradient_loss = common.aggregate_losses(
        per_example_loss=policy_gradient_loss, sample_weight=weights).total_loss

    if debug_summaries:
      if self._importance_ratio_clipping > 0.0:
        clip_fraction = tf.reduce_mean(
            input_tensor=tf.cast(
                tf.greater(
                    tf.abs(importance_ratio -
                           1.0), self._importance_ratio_clipping), tf.float32))
        tf.compat.v2.summary.scalar(
            name='clip_fraction',
            data=clip_fraction,
            step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='importance_ratio_mean',
          data=tf.reduce_mean(input_tensor=importance_ratio),
          step=self.train_step_counter)
      entropy = common.entropy(current_policy_distribution, self.action_spec)
      tf.compat.v2.summary.scalar(
          name='policy_entropy_mean',
          data=tf.reduce_mean(input_tensor=entropy),
          step=self.train_step_counter)
      # TODO(b/171573175): remove the condition once histograms are supported
      # on TPUs.
      if not tf.config.list_logical_devices('TPU'):
        tf.compat.v2.summary.histogram(
            name='action_log_prob',
            data=action_log_prob,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='action_log_prob_sample',
            data=sample_action_log_probs,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='importance_ratio',
            data=importance_ratio,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='importance_ratio_clipped',
            data=importance_ratio_clipped,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='per_timestep_objective',
            data=per_timestep_objective,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='per_timestep_objective_clipped',
            data=per_timestep_objective_clipped,
            step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='per_timestep_objective_min',
            data=per_timestep_objective_min,
            step=self.train_step_counter)

        tf.compat.v2.summary.histogram(
            name='policy_entropy', data=entropy, step=self.train_step_counter)
        for i, (single_action, single_distribution) in enumerate(
            zip(
                tf.nest.flatten(self.action_spec),
                tf.nest.flatten(current_policy_distribution))):
          # Categorical distribution (used for discrete actions) doesn't have a
          # mean.
          distribution_index = '_{}'.format(i) if i > 0 else ''
          if not tensor_spec.is_discrete(single_action):
            tf.compat.v2.summary.histogram(
                name='actions_distribution_mean' + distribution_index,
                data=single_distribution.mean(),
                step=self.train_step_counter)
            tf.compat.v2.summary.histogram(
                name='actions_distribution_stddev' + distribution_index,
                data=single_distribution.stddev(),
                step=self.train_step_counter)
        tf.compat.v2.summary.histogram(
            name='policy_gradient_loss',
            data=policy_gradient_loss,
            step=self.train_step_counter)

    if self._check_numerics:
      policy_gradient_loss = tf.debugging.check_numerics(
          policy_gradient_loss, 'policy_gradient_loss')

    return policy_gradient_loss

  def kl_cutoff_loss(self,
                     kl_divergence: types.Tensor,
                     debug_summaries: bool = False) -> types.Tensor:
    # Squared penalization for mean KL divergence above some threshold.
    if self._kl_cutoff_factor <= 0.0:
      return tf.constant(0.0, dtype=tf.float32, name='zero_kl_cutoff_loss')
    kl_cutoff = self._kl_cutoff_factor * self._adaptive_kl_target
    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    kl_over_cutoff = tf.maximum(mean_kl - kl_cutoff, 0.0)
    kl_cutoff_loss = self._kl_cutoff_coef * tf.square(kl_over_cutoff)

    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='kl_cutoff_count',
          data=tf.reduce_sum(
              input_tensor=tf.cast(kl_divergence > kl_cutoff, dtype=tf.int64)),
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='kl_cutoff_loss',
          data=kl_cutoff_loss,
          step=self.train_step_counter)

    return tf.identity(kl_cutoff_loss, name='kl_cutoff_loss')

  def adaptive_kl_loss(self,
                       kl_divergence: types.Tensor,
                       debug_summaries: bool = False) -> types.Tensor:
    if self._adaptive_kl_beta is None:
      return tf.constant(0.0, dtype=tf.float32, name='zero_adaptive_kl_loss')

    # Define the loss computation, which depends on the update computation.
    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    adaptive_kl_loss = self._adaptive_kl_beta * mean_kl

    if debug_summaries:
      tf.compat.v2.summary.scalar(
          name='adaptive_kl_loss',
          data=adaptive_kl_loss,
          step=self.train_step_counter)

    return adaptive_kl_loss

  def _kl_divergence(self, time_steps, action_distribution_parameters,
                     current_policy_distribution):
    outer_dims = list(
        range(nest_utils.get_outer_rank(time_steps, self.time_step_spec)))

    old_actions_distribution = (
        ppo_utils.distribution_from_spec(
            self._action_distribution_spec, action_distribution_parameters,
            legacy_distribution_network=isinstance(
                self._actor_net, network.DistributionNetwork)))

    kl_divergence = ppo_utils.nested_kl_divergence(
        old_actions_distribution,
        current_policy_distribution,
        outer_dims=outer_dims)
    return kl_divergence

  def kl_penalty_loss(self,
                      time_steps: ts.TimeStep,
                      action_distribution_parameters: types.NestedTensor,
                      current_policy_distribution: types.NestedDistribution,
                      weights: types.Tensor,
                      debug_summaries: bool = False) -> types.Tensor:
    """Compute a loss that penalizes policy steps with high KL.

    Based on KL divergence from old (data-collection) policy to new (updated)
    policy.

    All tensors should have a single batch dimension.

    Args:
      time_steps: TimeStep tuples with observations for each timestep. Used for
        computing new action distributions.
      action_distribution_parameters: Action distribution params of the data
        collection policy, used for reconstruction old action distributions.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Inlcudes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.

    Returns:
      kl_penalty_loss: The sum of a squared penalty for KL over a constant
        threshold, plus an adaptive penalty that encourages updates toward a
        target KL divergence.
    """
    kl_divergence = self._kl_divergence(time_steps,
                                        action_distribution_parameters,
                                        current_policy_distribution)
    kl_divergence *= weights

    # TODO(b/171573175): remove the condition once histograms are supported
    # on TPUs.
    if debug_summaries and not tf.config.list_logical_devices('TPU'):
      tf.compat.v2.summary.histogram(
          name='kl_divergence',
          data=kl_divergence,
          step=self.train_step_counter)

    kl_cutoff_loss = self.kl_cutoff_loss(kl_divergence, debug_summaries)
    adaptive_kl_loss = self.adaptive_kl_loss(kl_divergence, debug_summaries)
    return tf.add(kl_cutoff_loss, adaptive_kl_loss, name='kl_penalty_loss')

  def update_adaptive_kl_beta(
      self, kl_divergence: types.Tensor) -> Optional[tf.Operation]:
    """Create update op for adaptive KL penalty coefficient.

    Args:
      kl_divergence: KL divergence of old policy to new policy for all
        timesteps.

    Returns:
      update_op: An op which runs the update for the adaptive kl penalty term.
    """
    if self._adaptive_kl_beta is None:
      return tf.no_op()

    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)

    # Update the adaptive kl beta after each time it is computed.
    mean_kl_below_bound = (
        mean_kl < self._adaptive_kl_target *
        (1.0 - self._adaptive_kl_tolerance))
    mean_kl_above_bound = (
        mean_kl > self._adaptive_kl_target *
        (1.0 + self._adaptive_kl_tolerance))
    adaptive_kl_update_factor = tf.case(
        [
            (mean_kl_below_bound,
             lambda: tf.constant(1.0 / 1.5, dtype=tf.float32)),
            (mean_kl_above_bound, lambda: tf.constant(1.5, dtype=tf.float32)),
        ],
        default=lambda: tf.constant(1.0, dtype=tf.float32),
        exclusive=True)

    new_adaptive_kl_beta = tf.maximum(
        self._adaptive_kl_beta * adaptive_kl_update_factor, 10e-16)
    tf.compat.v1.assign(self._adaptive_kl_beta, new_adaptive_kl_beta)

    if self._debug_summaries:
      tf.compat.v2.summary.scalar(
          name='adaptive_kl_update_factor',
          data=adaptive_kl_update_factor,
          step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='mean_kl_divergence', data=mean_kl, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='adaptive_kl_beta',
          data=self._adaptive_kl_beta,
          step=self.train_step_counter)

    return self._adaptive_kl_beta


def _get_discount(experience) -> types.Tensor:
  """Try to get the discount entry from `experience`.

  Typically experience is either a Trajectory or a Transition.

  Args:
    experience: Data collected from e.g. a replay buffer.

  Returns:
    discount: The discount tensor stored in `experience`.
  """
  if isinstance(experience, trajectory.Transition):
    return experience.time_step.discount
  else:
    return experience.discount
