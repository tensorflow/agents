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

"""Policy for greedy reward prediction."""

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.policies import reward_prediction_base_policy
from tf_agents.policies import utils as policy_utilities


class GreedyRewardPredictionPolicy(
    reward_prediction_base_policy.RewardPredictionBasePolicy):
  """Class to build GreedyNNPredictionPolicies."""

  def _action_distribution(self, mask, predicted_rewards):
    """Returns the action with largest predicted reward."""
    # Argmax.
    batch_size = tf.shape(predicted_rewards)[0]
    if mask is not None:
      actions = policy_utilities.masked_argmax(
          predicted_rewards, mask, output_type=self.action_spec.dtype)
    else:
      actions = tf.argmax(
          predicted_rewards, axis=-1, output_type=self.action_spec.dtype)

    actions += self._action_offset

    bandit_policy_values = tf.fill([batch_size, 1],
                                   policy_utilities.BanditPolicyType.GREEDY)
    return tfp.distributions.Deterministic(loc=actions), bandit_policy_values

  def _distribution(self, time_step, policy_state):
    step = super(GreedyRewardPredictionPolicy,
                 self)._distribution(time_step, policy_state)
    # Greedy is deterministic, so we know the chosen arm features here. We
    # save it here so the chosen arm features get correctly returned by
    # `tf_agents.policies.epsilon_greey_policy.EpsilonGreedyPolicy` wrapping a
    # `GreedyRewardPredictionPolicy` because `EpsilonGreedyPolicy` only accesses
    # the `distribution` method of the wrapped policy via
    # `tf_agents.policies.greedy_policy.GreedyPolicy`.
    action = step.action.sample()
    return self._maybe_save_chosen_arm_features(time_step, action, step)
