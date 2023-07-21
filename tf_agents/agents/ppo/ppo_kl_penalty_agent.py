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

"""A PPO Agent implementing the KL penalty loss.

Please see details of the algorithm in (Schulman,2017):
https://arxiv.org/abs/1707.06347.

Disclaimer: We intend for this class to eventually fully replicate the KL
penalty version of PPO from:
https://github.com/openai/baselines/tree/master/baselines/ppo1
We are still working on resolving the differences in implementation details,
such as mini batch learning and learning rate annealing.

PPO is a simplification of the TRPO algorithm, both of which add stability to
policy gradient RL, while allowing multiple updates per batch of on-policy data.

TRPO enforces a hard optimization constraint, but is a complex algorithm, which
often makes it harder to use in practice. PPO approximates the effect of TRPO
by using a soft constraint. There are two methods presented in the paper for
implementing the soft constraint: an adaptive KL loss penalty, and
limiting the objective value based on a clipped version of the policy importance
ratio. This agent implements the KL penalty version.

Note that PPOKLPenaltyAgent is known to have worse performance than PPOClipAgent
(Schulman,2017). We included the implementation as it is an important baseline.

Note that PPOKLPenaltyAgent's behavior can be reproduced by the parent
"PPOAgent" if the right set of parameters are set. However, we strongly
encourage future clients to use PPOKLPenaltyAgent instead if you rely on the KL
penalty version of PPO, because PPOKLPenaltyAgent abstracts away the
parameters unrelated to this particular PPO version, making it less error prone.

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
class PPOKLPenaltyAgent(ppo_agent.PPOAgent):
  """A PPO Agent implementing the KL penalty loss."""

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      actor_net: network.Network,
      value_net: network.Network,
      num_epochs: int,
      initial_adaptive_kl_beta: types.Float,
      adaptive_kl_target: types.Float,
      adaptive_kl_tolerance: types.Float,
      optimizer: Optional[types.Optimizer] = None,
      use_gae: bool = True,
      use_td_lambda_return: bool = True,
      lambda_value: types.Float = 0.95,
      discount_factor: types.Float = 0.99,
      value_pred_loss_coef: types.Float = 0.5,
      entropy_regularization: types.Float = 0.0,
      policy_l2_reg: types.Float = 0.0,
      value_function_l2_reg: types.Float = 0.0,
      shared_vars_l2_reg: types.Float = 0.0,
      normalize_observations: bool = False,
      normalize_rewards: bool = True,
      reward_norm_clipping: types.Float = 0.0,
      log_prob_clipping: types.Float = 0.0,
      gradient_clipping: Optional[types.Float] = None,
      value_clipping: Optional[types.Float] = None,
      kl_cutoff_coef: types.Float = 0.0,
      kl_cutoff_factor: Optional[types.Float] = None,
      check_numerics: bool = False,
      debug_summaries: bool = False,
      # TODO(b/150244758): Change the default to False once we move
      # clients onto Reverb.
      compute_value_and_advantage_in_train: bool = True,
      update_normalizers_in_train: bool = True,
      aggregate_losses_across_replicas: bool = True,
      summarize_grads_and_vars: bool = False,
      train_step_counter: Optional[tf.Variable] = None,
      name: Optional[Text] = None):
    """Creates a PPO Agent implementing the KL penalty loss.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      actor_net: A `network.DistributionNetwork` which maps observations to
        action distributions. Commonly, it is set to
        `actor_distribution_network.ActorDistributionNetwork`.
      value_net: A `Network` which returns the value prediction for input
        states, with `call(observation, step_type, network_state)`. Commonly, it
        is set to `value_network.ValueNetwork`.
      num_epochs: Number of epochs for computing policy updates. (Schulman,2017)
        sets this to 10 for Mujoco, 15 for Roboschool and 3 for Atari.
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
      optimizer: Optimizer to use for the agent, default to using
        `tf.compat.v1.train.AdamOptimizer`.
      use_gae: If `True`, uses generalized advantage estimation for computing
        per-timestep advantage. Else, just subtracts value predictions from
        empirical return.
      use_td_lambda_return: If `True`, uses `td_lambda_return` for training
        value function; here:
        `td_lambda_return = gae_advantage + value_predictions`.
        `use_gae` must be set to `True` as well to enable TD -lambda returns. If
        `use_td_lambda_return` is set to True while `use_gae` is False, the
        empirical return will be used and a warning will be logged.
      lambda_value: Lambda parameter for TD-lambda computation. Default to
       `0.95` which is the value used for all environments from the paper.
      discount_factor: Discount factor for return computation. Default to `0.99`
        which is the value used for all environments from the paper.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss. Default to `0.5`, which was used for all
        environments in the OpenAI baseline implementation. This parameters is
        irrelevant unless you are sharing part of actor_net and value_net. In
        that case, you would want to tune this coeeficient, whose value depends
        on the network architecture of your choice
      entropy_regularization: Coefficient for entropy regularization loss term.
        Default to `0.0` because no entropy bonus was applied in the PPO paper.
      policy_l2_reg: Coefficient for L2 regularization of unshared actor_net
        weights. Default to `0.0` because no L2 regularization was applied on
        the policy network weights in the PPO paper.
      value_function_l2_reg: Coefficient for l2 regularization of unshared value
       function weights. Default to `0.0` because no L2 regularization was
       applied on the policy network weights in the PPO paper.
      shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
        between actor_net and value_net. Default to `0.0` because no L2
        regularization was applied on either network in the PPO paper.
      normalize_observations: If `True` (default `False`), keeps moving mean and
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
      normalize_rewards: If `True`, keeps moving variance of rewards and
        normalizes incoming rewards. While not mentioned directly in the PPO
        paper, reward normalization was implemented in OpenAI baselines and
        (Ilyas et al., 2018) pointed out that it largely improves performance.
        You may refer to Figure 1 of https://arxiv.org/pdf/1811.02553.pdf for a
          comparison with and without reward scaling.
      reward_norm_clipping: Value above and below to clip normalized reward.
        Additional optimization proposed in (Ilyas et al., 2018) set to
        `5` or `10`.
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      value_clipping: Difference between new and old value predictions are
        clipped to this threshold. Value clipping could be helpful when training
        very deep networks. Default: no clipping.
      kl_cutoff_coef: kl_cutoff_coef and kl_cutoff_factor are additional params
        if one wants to use a KL cutoff loss term in addition to the adaptive KL
        loss term. Default to 0.0 to disable the KL cutoff loss term as this was
        not used in the paper.  kl_cutoff_coef is the coefficient to mulitply by
        the KL cutoff loss term, before adding to the total loss function.
      kl_cutoff_factor: Only meaningful when `kl_cutoff_coef > 0.0`. A multipler
        used for calculating the KL cutoff ( =
        `kl_cutoff_factor * adaptive_kl_target`). If policy KL averaged across
        the batch changes more than the cutoff, a squared cutoff loss would
        be added to the loss function.
      check_numerics: If true, adds `tf.debugging.check_numerics` to help find
        NaN / Inf values. For debugging only.
      debug_summaries: A bool to gather debug summaries.
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
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the actor_net is not a DistributionNetwork or value_net is
        not a Network.
      ValueError: If kl_cutoff_coef > 0.0 (indicating that a KL cutoff loss term
        will not be added), but kl_cutoff_factor is None.
    """
    if kl_cutoff_coef > 0.0 and kl_cutoff_factor is None:
      raise ValueError(
          'kl_cutoff_factor needs to be set if kl_cutoff_coef is non-zero.')

    super(PPOKLPenaltyAgent, self).__init__(
        time_step_spec,
        action_spec,
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        lambda_value=lambda_value,
        discount_factor=discount_factor,
        entropy_regularization=entropy_regularization,
        policy_l2_reg=policy_l2_reg,
        value_function_l2_reg=value_function_l2_reg,
        shared_vars_l2_reg=shared_vars_l2_reg,
        value_pred_loss_coef=value_pred_loss_coef,
        num_epochs=num_epochs,
        use_gae=use_gae,
        use_td_lambda_return=use_td_lambda_return,
        normalize_rewards=normalize_rewards,
        reward_norm_clipping=reward_norm_clipping,
        normalize_observations=normalize_observations,
        log_prob_clipping=log_prob_clipping,
        kl_cutoff_factor=kl_cutoff_factor,
        kl_cutoff_coef=kl_cutoff_coef,
        initial_adaptive_kl_beta=initial_adaptive_kl_beta,
        adaptive_kl_target=adaptive_kl_target,
        adaptive_kl_tolerance=adaptive_kl_tolerance,
        gradient_clipping=gradient_clipping,
        value_clipping=value_clipping,
        check_numerics=check_numerics,
        debug_summaries=debug_summaries,
        compute_value_and_advantage_in_train=compute_value_and_advantage_in_train,
        update_normalizers_in_train=update_normalizers_in_train,
        aggregate_losses_across_replicas=aggregate_losses_across_replicas,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name,
        # Skips parameters specific to PPOClipAgent.
        importance_ratio_clipping=0.0,
    )
