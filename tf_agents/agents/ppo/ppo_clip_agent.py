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

"""A PPO Agent implementing the clipped probability ratios.

Please see details of the algorithm in (Schulman,2017):
https://arxiv.org/abs/1707.06347.

Disclaimer: We intend for this class to eventually fully replicate:
https://github.com/openai/baselines/tree/master/baselines/ppo2

Currently, this agent surpasses the paper performance for average returns on
Half-Cheetah when wider networks and higher learning rates are used. However,
some details from this class still differ from the paper implementation.
For example, we do not perform mini-batch learning and learning rate annealing
yet. We are in working progress to reproduce the paper implementation exactly.

PPO is a simplification of the TRPO algorithm, both of which add stability to
policy gradient RL, while allowing multiple updates per batch of on-policy data.

TRPO enforces a hard optimization constraint, but is a complex algorithm, which
often makes it harder to use in practice. PPO approximates the effect of TRPO
by using a soft constraint. There are two methods presented in the paper for
implementing the soft constraint: an adaptive KL loss penalty, and
limiting the objective value based on a clipped version of the policy importance
ratio. This agent implements the clipped version.

The importance ratio clipping is described in eq (7) of
https://arxiv.org/pdf/1707.06347.pdf
- To disable IR clipping, set the importance_ratio_clipping parameter to 0.0.

Note that the objective function chooses the lower value of the clipped and
unclipped objectives. Thus, if the importance ratio exceeds the clipped bounds,
then the optimizer will still not be incentivized to pass the bounds, as it is
only optimizing the minimum.

Advantage is computed using Generalized Advantage Estimation (GAE):
https://arxiv.org/abs/1506.02438
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import network
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


@gin.configurable
class PPOClipAgent(ppo_agent.PPOAgent):
  """A PPO Agent implementing the clipped probability ratios."""

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      optimizer: Optional[types.Optimizer] = None,
      actor_net: Optional[network.Network] = None,
      value_net: Optional[network.Network] = None,
      greedy_eval: bool = True,
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
      gradient_clipping: Optional[types.Float] = None,
      value_clipping: Optional[types.Float] = None,
      check_numerics: bool = False,
      # TODO(b/150244758): Change the default to False once we move
      # clients onto Reverb.
      compute_value_and_advantage_in_train: bool = True,
      update_normalizers_in_train: bool = True,
      aggregate_losses_across_replicas: bool = True,
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      train_step_counter: Optional[tf.Variable] = None,
      name: Optional[Text] = 'PPOClipAgent'):
    """Creates a PPO Agent implementing the clipped probability ratios.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      optimizer: Optimizer to use for the agent.
      actor_net: A function actor_net(observations, action_spec) that returns
        tensor of action distribution params for each observation. Takes nested
        observation and returns nested action.
      value_net: A function value_net(time_steps) that returns value tensor from
        neural net predictions for each observation. Takes nested observation
        and returns batch of value_preds.
      greedy_eval: Whether to use argmax/greedy action selection or sample from
        original action distribution for the evaluation policy. For environments
        such as ProcGen, stochastic is much better than greedy.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation.
      entropy_regularization: Coefficient for entropy regularization loss term.
      policy_l2_reg: Coefficient for l2 regularization of unshared policy
        weights.
      value_function_l2_reg: Coefficient for l2 regularization of unshared value
       function weights.
      shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
        between the policy and value functions.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss.
      num_epochs: Number of epochs for computing policy updates.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function. (td_lambda_return = gae_advantage +
        value_predictions)
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards.
      reward_norm_clipping: Value above and below to clip normalized reward.
      normalize_observations: If true, keeps moving mean and variance of
        observations and normalizes incoming observations. If true, and the
        observation spec is not tf.float32 (such as Atari), please manually
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
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      value_clipping: Difference between new and old value predictions are
        clipped to this threshold. Value clipping could be helpful when training
        very deep networks. Default: no clipping.
      check_numerics: If true, adds tf.debugging.check_numerics to help find NaN
        / Inf values. For debugging only.
      compute_value_and_advantage_in_train: A bool to indicate where value
        prediction and advantage calculation happen.  If True, both happen in
        agent.train(). If False, value prediction is computed during data
        collection. This argument must be set to `False` if mini batch learning
        is enabled.
      update_normalizers_in_train: A bool to indicate whether normalizers are
        updated at the end of the `train` method. Set to `False` if mini batch
        learning is enabled, or if `train` is called on multiple iterations of
        the same trajectories. In that case, you would need to call the
        `update_reward_normalizer` and `update_observation_normalizer` methods
        after all iterations of the same trajectory are done. This ensures that
        normalizers are updated in the same way as (Schulman, 2017).
      aggregate_losses_across_replicas: only applicable to setups using multiple
        relicas. Default to aggregating across multiple cores using tf_agents.
        common.aggregate_losses. If set to `False`, use `reduce_mean` directly,
        which is faster but may impact learning results.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the actor_net is not a DistributionNetwork.
    """
    super(PPOClipAgent, self).__init__(
        time_step_spec,
        action_spec,
        optimizer,
        actor_net,
        value_net,
        greedy_eval,
        importance_ratio_clipping,
        lambda_value,
        discount_factor,
        entropy_regularization,
        policy_l2_reg,
        value_function_l2_reg,
        shared_vars_l2_reg,
        value_pred_loss_coef,
        num_epochs,
        use_gae,
        use_td_lambda_return,
        normalize_rewards,
        reward_norm_clipping,
        normalize_observations,
        gradient_clipping=gradient_clipping,
        value_clipping=value_clipping,
        check_numerics=check_numerics,
        compute_value_and_advantage_in_train=compute_value_and_advantage_in_train,
        update_normalizers_in_train=update_normalizers_in_train,
        aggregate_losses_across_replicas=aggregate_losses_across_replicas,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name,
        # Skips parameters used for the adaptive KL loss penalty version of PPO.
        log_prob_clipping=0.0,
        kl_cutoff_factor=0.0,
        kl_cutoff_coef=0.0,
        initial_adaptive_kl_beta=0.0,
        adaptive_kl_target=0.0,
        adaptive_kl_tolerance=0.0)
