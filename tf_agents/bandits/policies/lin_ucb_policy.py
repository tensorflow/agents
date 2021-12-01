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

"""Linear UCB Policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, Sequence, Text

from tf_agents.bandits.policies import linear_bandit_policy as lin_policy
from tf_agents.typing import types


class LinearUCBPolicy(lin_policy.LinearBanditPolicy):
  """Linear UCB Policy.

  Implements the Linear UCB Policy from the following paper:
  "A Contextual Bandit Approach to Personalized News Article Recommendation",
  Lihong Li, Wei Chu, John Langford, Robert Schapire, WWW 2010.

  """

  def __init__(self,
               action_spec: types.BoundedTensorSpec,
               cov_matrix: Sequence[types.Float],
               data_vector: Sequence[types.Float],
               num_samples: Sequence[types.Int],
               time_step_spec: Optional[types.TimeStep] = None,
               alpha: float = 1.0,
               eig_vals: Sequence[types.Float] = (),
               eig_matrix: Sequence[types.Float] = (),
               tikhonov_weight: float = 1.0,
               add_bias: bool = False,
               emit_policy_info: Sequence[Text] = (),
               emit_log_probability: bool = False,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               name: Optional[Text] = None):
    """Initializes `LinUCBPolicy`.

    The `a` and `b` arguments may be either `Tensor`s or `tf.Variable`s.
    If they are variables, then any assignements to those variables will be
    reflected in the output of the policy.

    Args:
      action_spec: `TensorSpec` containing action specification.
      cov_matrix: list of the covariance matrices A in the paper. There exists
        one A matrix per arm.
      data_vector: list of the b vectors in the paper. The b vector is a
        weighted sum of the observations, where the weight is the corresponding
        reward. Each arm has its own vector b.
      num_samples: list of number of samples per arm.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      alpha: a float value used to scale the confidence intervals.
      eig_vals: list of eigenvalues for each covariance matrix (one per arm).
      eig_matrix: list of eigenvectors for each covariance matrix (one per arm).
      tikhonov_weight: (float) tikhonov regularization term.
      add_bias: If true, a bias term will be added to the linear reward
        estimation.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      emit_log_probability: Whether to emit log probabilities.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit policy and 2)
        the mask. The mask should be a 0-1 `Tensor` of shape
        `[batch_size, num_actions]`. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      name: The name of this policy.
    """
    super(LinearUCBPolicy, self).__init__(
        action_spec=action_spec,
        cov_matrix=cov_matrix,
        data_vector=data_vector,
        num_samples=num_samples,
        time_step_spec=time_step_spec,
        exploration_strategy=lin_policy.ExplorationStrategy.optimistic,
        alpha=alpha,
        eig_vals=eig_vals,
        eig_matrix=eig_matrix,
        tikhonov_weight=tikhonov_weight,
        add_bias=add_bias,
        emit_policy_info=emit_policy_info,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)
