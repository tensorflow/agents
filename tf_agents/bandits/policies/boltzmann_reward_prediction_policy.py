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

"""Policy for reward prediction and boltzmann exploration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, Optional, Text, Tuple, Sequence

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.policies import constraints as constr
from tf_agents.bandits.policies import reward_prediction_base_policy
from tf_agents.distributions import shifted_categorical
from tf_agents.policies import utils as policy_utilities
from tf_agents.typing import types

# The temperature parameter is internally lower-bounded at this value to avoid
# numerical issues.
_MIN_TEMPERATURE = 1e-12


class BoltzmannRewardPredictionPolicy(
    reward_prediction_base_policy.RewardPredictionBasePolicy):
  """Class to build Reward Prediction Policies with Boltzmann exploration."""

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.NestedTensorSpec,
      reward_network: types.Network,
      temperature: types.FloatOrReturningFloat = 1.0,
      boltzmann_gumbel_exploration_constant: Optional[types.Float] = None,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      accepts_per_arm_features: bool = False,
      constraints: Iterable[constr.BaseConstraint] = (),
      emit_policy_info: Tuple[Text, ...] = (),
      num_samples_list: Sequence[tf.Variable] = (),
      name: Optional[Text] = None):
    """Builds a BoltzmannRewardPredictionPolicy given a reward network.

    This policy takes a tf_agents.Network predicting rewards and chooses an
    action with weighted probabilities (i.e., using a softmax over the network
    estimates of value for each action).

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      reward_network: An instance of a `tf_agents.network.Network`,
        callable via `network(observation, step_type) -> (output, final_state)`.
      temperature: float or callable that returns a float. The temperature used
        in the Boltzmann exploration.
      boltzmann_gumbel_exploration_constant: optional positive float. When
        provided, the policy implements Neural Bandit with Boltzmann-Gumbel
        exploration from the paper:
        N. Cesa-Bianchi et al., "Boltzmann Exploration Done Right", NIPS 2017.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the network and 2) the
        mask.  The mask should be a 0-1 `Tensor` of shape
        `[batch_size, num_actions]`. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
      constraints: iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.BaseConstraint`.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      num_samples_list: list or tuple of tf.Variable's. Used only in
        Boltzmann-Gumbel exploration. Otherwise, empty.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec` or the `BoundedTensorSpec` is not valid.
    """
    super(BoltzmannRewardPredictionPolicy,
          self).__init__(time_step_spec, action_spec, reward_network,
                         observation_and_action_constraint_splitter,
                         accepts_per_arm_features, constraints,
                         emit_policy_info, name)

    self._temperature = temperature
    self._boltzmann_gumbel_exploration_constant = (
        boltzmann_gumbel_exploration_constant)
    self._num_samples_list = num_samples_list
    if self._boltzmann_gumbel_exploration_constant is not None:
      if self._boltzmann_gumbel_exploration_constant <= 0.0:
        raise ValueError(
            'The Boltzmann-Gumbel exploration constant is expected to be ',
            'positive. Found: ', self._boltzmann_gumbel_exploration_constant)
      if self._action_offset > 0:
        raise NotImplementedError('Action offset is not supported when ',
                                  'Boltzmann-Gumbel exploration is enabled.')
      if accepts_per_arm_features:
        raise NotImplementedError(
            'Boltzmann-Gumbel exploration is not supported ',
            'for arm features case.')
      if len(self._num_samples_list) != self._expected_num_actions:
        raise ValueError(
            'Size of num_samples_list: ', len(self._num_samples_list),
            ' does not match the expected number of actions:',
            self._expected_num_actions)

  def _get_temperature_value(self):
    return tf.math.maximum(
        _MIN_TEMPERATURE,
        self._temperature()
        if callable(self._temperature) else self._temperature)

  def _action_distribution(self, mask, predicted_rewards):
    batch_size = tf.shape(predicted_rewards)[0]
    if self._boltzmann_gumbel_exploration_constant is not None:
      logits = predicted_rewards

      # Apply masking if needed. Overwrite the logits for invalid actions to
      # logits.dtype.min.
      if mask is not None:
        almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
        logits = tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, almost_neg_inf)

      gumbel_dist = tfp.distributions.Gumbel(loc=0., scale=1.)
      gumbel_samples = gumbel_dist.sample(tf.shape(logits))
      num_samples_list_float = tf.stack(
          [tf.cast(x.read_value(), tf.float32) for x in self._num_samples_list],
          axis=-1)
      exploration_weights = tf.math.divide_no_nan(
          self._boltzmann_gumbel_exploration_constant,
          tf.sqrt(num_samples_list_float))
      final_logits = logits + exploration_weights * gumbel_samples
      actions = tf.cast(
          tf.math.argmax(final_logits, axis=1), self._action_spec.dtype)
      # To conform with the return type, we construct a deterministic
      # distribution here. Note that this results in the log_probability of
      # the chosen arm being 0. The true sampling probability here has no simple
      # closed-form.
      distribution = tfp.distributions.Deterministic(loc=actions)
    else:
      # Apply the temperature scaling, needed for Boltzmann exploration.
      logits = predicted_rewards / self._get_temperature_value()

      # Apply masking if needed. Overwrite the logits for invalid actions to
      # logits.dtype.min.
      if mask is not None:
        almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
        logits = tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, almost_neg_inf)

      if self._action_offset != 0:
        distribution = shifted_categorical.ShiftedCategorical(
            logits=logits,
            dtype=self._action_spec.dtype,
            shift=self._action_offset)
      else:
        distribution = tfp.distributions.Categorical(
            logits=logits,
            dtype=self._action_spec.dtype)

    bandit_policy_values = tf.fill([batch_size, 1],
                                   policy_utilities.BanditPolicyType.BOLTZMANN)
    return distribution, bandit_policy_values
