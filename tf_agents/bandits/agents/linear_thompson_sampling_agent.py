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

"""Implements the Linear Thompson Sampling bandit algorithm.

  Reference:
  "Thompson Sampling for Contextual Bandits with Linear Payoffs",
  Shipra Agrawal, Navin Goyal, ICML 2013. The actual algorithm implemented is
  `Algorithm 3` from the supplementary material of the paper from
  `http://proceedings.mlr.press/v28/agrawal13-supp.pdf`.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Sequence, Text

import gin
import tensorflow as tf

from tf_agents.bandits.agents import linear_bandit_agent as lin_agent
from tf_agents.typing import types


@gin.configurable
class LinearThompsonSamplingAgent(lin_agent.LinearBanditAgent):
  """Linear Thompson Sampling Agent.

  Implements the Linear Thompson Sampling Agent from the following paper:
  "Thompson Sampling for Contextual Bandits with Linear Payoffs",
  Shipra Agrawal, Navin Goyal, ICML 2013. The actual algorithm implemented is
  `Algorithm 3` from the supplementary material of the paper from
  `http://proceedings.mlr.press/v28/agrawal13-supp.pdf`.

  In a nutshell, the agent maintains two parameters `weight_covariances` and
  `parameter_estimators`, and updates them based on experience. The inverse of
  the weight covariance parameters are updated with the outer product of the
  observations using the Woodbury inverse matrix update, while the parameter
  estimators are updated by the reward-weighted observation vectors for every
  action.
  """

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.BoundedTensorSpec,
               variable_collection: Optional[
                   lin_agent.LinearBanditVariableCollection] = None,
               alpha: float = 1.0,
               gamma: float = 1.0,
               use_eigendecomp: bool = False,
               tikhonov_weight: float = 1.0,
               add_bias: bool = False,
               emit_policy_info: Sequence[Text] = (),
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               accepts_per_arm_features: bool = False,
               debug_summaries: bool = False,
               summarize_grads_and_vars: bool = False,
               enable_summaries: bool = True,
               dtype: tf.DType = tf.float32,
               name: Optional[Text] = None):
    """Initialize an instance of `LinearThompsonSamplingAgent`.

    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
      variable_collection: Instance of `LinearBanditVariableCollection`.
        Collection of variables to be updated by the agent. If `None`, a new
        instance of `LinearBanditVariableCollection` will be created.
      alpha: (float) positive scalar. This is the exploration parameter that
        multiplies the confidence intervals.
      gamma: a float forgetting factor in [0.0, 1.0]. When set to
        1.0, the algorithm does not forget.
      use_eigendecomp: whether to use eigen-decomposition or not. The default
        solver is Conjugate Gradient.
      tikhonov_weight: (float) tikhonov regularization term.
      add_bias: If true, a bias term will be added to the linear reward
        estimation.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      accepts_per_arm_features: (bool) Whether the agent accepts per-arm
        features.
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      dtype: The type of the parameters stored and updated by the agent. Should
        be one of `tf.float32` and `tf.float64`. Defaults to `tf.float32`.
      name: a name for this instance of `LinearThompsonSamplingAgent`.

    Raises:
      ValueError if dtype is not one of `tf.float32` or `tf.float64`.
    """
    super(LinearThompsonSamplingAgent, self).__init__(
        exploration_policy=(
            lin_agent.ExplorationPolicy.linear_thompson_sampling_policy),
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        variable_collection=variable_collection,
        alpha=alpha,
        gamma=gamma,
        use_eigendecomp=use_eigendecomp,
        tikhonov_weight=tikhonov_weight,
        add_bias=add_bias,
        emit_policy_info=emit_policy_info,
        emit_log_probability=False,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=accepts_per_arm_features,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        dtype=dtype,
        name=name)
