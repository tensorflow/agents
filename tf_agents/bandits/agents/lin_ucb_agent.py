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

"""Implements the Linear UCB bandit algorithm.

  Reference:
  "A Contextual Bandit Approach to Personalized News Article Recommendation",
  Lihong Li, Wei Chu, John Langford, Robert Schapire, WWW 2010.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.bandits.agents import linear_bandit_agent as lin_agent


@gin.configurable
class LinearUCBAgent(lin_agent.LinearBanditAgent):
  """An agent implementing the Linear UCB bandit algorithm.

  Reference:
  "A Contextual Bandit Approach to Personalized News Article Recommendation",
  Lihong Li, Wei Chu, John Langford, Robert Schapire, WWW 2010.
  """

  def __init__(self,
               time_step_spec,
               action_spec,
               alpha=1.0,
               gamma=1.0,
               use_eigendecomp=False,
               tikhonov_weight=1.0,
               add_bias=False,
               emit_policy_info=(),
               emit_log_probability=False,
               observation_and_action_constraint_splitter=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               enable_summaries=True,
               dtype=tf.float32,
               name=None):
    """Initialize an instance of `LinearUCBAgent`.

    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
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
      emit_log_probability: Whether the LinearUCBPolicy emits log-probabilities
        or not. Since the policy is deterministic, the probability is just 1.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      dtype: The type of the parameters stored and updated by the agent. Should
        be one of `tf.float32` and `tf.float64`. Defaults to `tf.float32`.
      name: a name for this instance of `LinearUCBAgent`.

    Raises:
      ValueError if dtype is not one of `tf.float32` or `tf.float64`.
    """
    super(LinearUCBAgent, self).__init__(
        exploration_policy=lin_agent.ExplorationPolicy.linear_ucb_policy,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        alpha=alpha,
        gamma=gamma,
        use_eigendecomp=use_eigendecomp,
        tikhonov_weight=tikhonov_weight,
        add_bias=add_bias,
        emit_policy_info=emit_policy_info,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        dtype=dtype,
        name=name)
