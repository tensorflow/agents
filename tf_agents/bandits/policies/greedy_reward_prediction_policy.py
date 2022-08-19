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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.policies import reward_prediction_base_policy
from tf_agents.policies import utils as policy_utilities


class GreedyRewardPredictionPolicy(
    reward_prediction_base_policy.RewardPredictionBasePolicy):
  """Class to build GreedyNNPredictionPolicies."""

  def _sample_action(self, mask, predicted_rewards):
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
    # This deterministic policy chooses the greedy action with probability 1.
    log_probability = tf.zeros([batch_size], tf.float32)
    return actions, log_probability, bandit_policy_values
