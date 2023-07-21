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

"""A neural network based agent that implements Falcon exploration.

The policy samples actions with the action distribution proposed in the
FALCON paper: David Simchi-Levi and Yunzong Xu, "Bypassing the Monster: A Faster
and Simpler Optimal Algorithm for Contextual Bandits under Realizability",
Mathematics of Operations Research, 2021. https://arxiv.org/pdf/2003.12699.pdf
"""

from typing import Iterable, Optional, Sequence, Text, Tuple

import gin
import tensorflow as tf
from tf_agents.bandits.agents import greedy_reward_prediction_agent
from tf_agents.bandits.policies import constraints as constr
from tf_agents.bandits.policies import falcon_reward_prediction_policy
from tf_agents.typing import types


@gin.configurable
class NeuralFalconAgent(
    greedy_reward_prediction_agent.GreedyRewardPredictionAgent):
  """A neural network based agent implementing the Falcon sampling strategy.

  This agent receives a neural network that it trains to predict rewards. The
  action is chosen by a stochastic policy that uses the action distribution in:
  David Simchi-Levi and Yunzong Xu, "Bypassing the Monster: A Faster
  and Simpler Optimal Algorithm for Contextual Bandits under Realizability",
  Mathematics of Operations Research, 2021. https://arxiv.org/pdf/2003.12699.pdf
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      reward_network: types.Network,
      optimizer: types.Optimizer,
      num_samples_list: Sequence[tf.Variable],
      exploitation_coefficient: types.FloatOrReturningFloat = 1.0,
      max_exploration_probability_hint: Optional[
          types.FloatOrReturningFloat
      ] = None,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      accepts_per_arm_features: bool = False,
      constraints: Iterable[constr.BaseConstraint] = (),
      # Params for training.
      error_loss_fn: types.LossFn = tf.compat.v1.losses.mean_squared_error,
      gradient_clipping: Optional[float] = None,
      # Params for debugging.
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      enable_summaries: bool = True,
      emit_policy_info: Tuple[Text, ...] = (),
      train_step_counter: Optional[tf.Variable] = None,
      laplacian_matrix: Optional[types.Float] = None,
      laplacian_smoothing_weight: float = 0.001,
      name: Optional[Text] = None):
    """Creates a Neural Falcon Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      reward_network: A `tf_agents.network.Network` to be used by the agent. The
        network will be called with call(observation, step_type) and it is
        expected to provide a reward prediction for all actions. *Note*: when
        using `observation_and_action_constraint_splitter`, make sure the
        `reward_network` is compatible with the network-specific half of the
        output of the `observation_and_action_constraint_splitter`. In
        particular, `observation_and_action_constraint_splitter` will be called
        on the observation before passing to the network.
      optimizer: The optimizer to use for training.
      num_samples_list: list or tuple of tf.Variable's tracking the number of
        training examples for every action.
      exploitation_coefficient: float or callable that returns a float. Its
        value will be internally lower-bounded at 0. It controls how
        exploitative the policy behaves with respect to the predicted rewards: A
        larger value makes the policy sample the greedy action (one with the
        best predicted reward) with a higher probability.
      max_exploration_probability_hint: An optional float, representing a hint
        on the maximum exploration probability, internally clipped to [0, 1].
        When this argument is set, `exploitation_coefficient` is ignored and the
        policy attempts to choose non-greedy actions with at most this
        probability. When such an upper bound cannot be achieved, e.g. due to
        insufficient training data, the policy attempts to minimize the
        probability of choosing non-greedy actions on a best-effort basis. For a
        demonstration of how it affects the policy behavior, see the unit test
        `testTrainedPolicyWithMaxExplorationProbabilityHint` in
        `neural_falcon_agent_test`.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
      constraints: iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.NeuralConstraint`.
      error_loss_fn: A function for computing the error loss, taking parameters
        labels, predictions, and weights (any function from tf.losses would
        work). The default is `tf.losses.mean_squared_error`.
      gradient_clipping: A float representing the norm length to clip gradients
        (or None for no clipping.)
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      train_step_counter: An optional `tf.Variable` to increment every time the
        train op is run.  Defaults to the `global_step`.
      laplacian_matrix: A float `Tensor` or a numpy array shaped `[num_actions,
        num_actions]`. This holds the Laplacian matrix used to regularize the
        smoothness of the estimated expected reward function. This only applies
        to problems where the actions have a graph structure. If `None`, the
        regularization is not applied.
      laplacian_smoothing_weight: A float that determines the weight of the
        regularization term. Note that this has no effect if `laplacian_matrix`
        above is `None`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or or it is
      not a bounded scalar int32 spec with minimum 0.
    """
    super(NeuralFalconAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        reward_network=reward_network,
        optimizer=optimizer,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=accepts_per_arm_features,
        constraints=constraints,
        error_loss_fn=error_loss_fn,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        emit_policy_info=emit_policy_info,
        train_step_counter=train_step_counter,
        num_samples_list=num_samples_list,
        laplacian_smoothing_weight=laplacian_smoothing_weight,
        laplacian_matrix=laplacian_matrix,
        name=name)
    self._policy = falcon_reward_prediction_policy.FalconRewardPredictionPolicy(
        time_step_spec,
        action_spec,
        reward_network,
        exploitation_coefficient=exploitation_coefficient,
        max_exploration_probability_hint=max_exploration_probability_hint,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter
        ),
        constraints=constraints,
        accepts_per_arm_features=accepts_per_arm_features,
        emit_policy_info=emit_policy_info,
        num_samples_list=num_samples_list,
    )

    self._collect_policy = self._policy
