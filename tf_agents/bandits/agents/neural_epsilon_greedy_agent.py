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

"""A neural network based agent that implements epsilon greedy exploration.

Implements an agent based on a neural network that predicts arm rewards.
The policy adds epsilon greedy exploration.

"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Iterable, Optional, Text, Tuple

import gin
import tensorflow as tf

from tf_agents.bandits.agents import greedy_reward_prediction_agent
from tf_agents.bandits.policies import constraints as constr
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.typing import types


@gin.configurable
class NeuralEpsilonGreedyAgent(
    greedy_reward_prediction_agent.GreedyRewardPredictionAgent):
  """A neural network based epsilon greedy agent.

  This agent receives a neural network that it trains to predict rewards. The
  action is chosen greedily with respect to the prediction with probability
  `1 - epsilon`, and uniformly randomly with probability `epsilon`.
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      reward_network: types.Network,
      optimizer: types.Optimizer,
      epsilon: float,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      accepts_per_arm_features: bool = False,
      constraints: Iterable[constr.NeuralConstraint] = (),
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
    """Creates a Neural Epsilon Greedy Agent.

    For more details about the Laplacian smoothing regularization, please see
    the documentation of the `GreedyRewardPredictionAgent`.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      reward_network: A `tf_agents.network.Network` to be used by the agent. The
        network will be called with call(observation, step_type) and it is
        expected to provide a reward prediction for all actions.
        *Note*: when using `observation_and_action_constraint_splitter`, make
        sure the `reward_network` is compatible with the network-specific half
        of the output of the `observation_and_action_constraint_splitter`. In
        particular, `observation_and_action_constraint_splitter` will be called
        on the observation before passing to the network.
      optimizer: The optimizer to use for training.
      epsilon: A float representing the probability of choosing a random action
        instead of the greedy action.
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
      laplacian_matrix: A float `Tensor` shaped `[num_actions, num_actions]`.
        This holds the Laplacian matrix used to regularize the smoothness of the
        estimated expected reward function. This only applies to problems where
        the actions have a graph structure. If `None`, the regularization is not
        applied.
      laplacian_smoothing_weight: A float that determines the weight of the
        regularization term. Note that this has no effect if `laplacian_matrix`
        above is `None`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or or it is
      not a bounded scalar int32 spec with minimum 0.
    """
    super(NeuralEpsilonGreedyAgent, self).__init__(
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
        laplacian_matrix=laplacian_matrix,
        laplacian_smoothing_weight=laplacian_smoothing_weight,
        name=name)
    self._policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        self._policy, epsilon=epsilon)
    self._collect_policy = self._policy
