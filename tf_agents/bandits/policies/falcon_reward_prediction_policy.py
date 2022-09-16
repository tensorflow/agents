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

"""Policy that samples actions based on the FALCON algorithm.

This policy implements an action sampling distribution based on the following
paper: David Simchi-Levi and Yunzong Xu, "Bypassing the Monster: A Faster and
Simpler Optimal Algorithm for Contextual Bandits under Realizability",
Mathematics of Operations Research, 2021. https://arxiv.org/pdf/2003.12699.pdf
"""

from typing import Iterable, Optional, Text, Tuple, Sequence

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.policies import constraints as constr
from tf_agents.bandits.policies import reward_prediction_base_policy
from tf_agents.distributions import shifted_categorical
from tf_agents.policies import utils as policy_utilities
from tf_agents.typing import types


def get_number_of_trainable_elements(network: types.Network) -> types.Float:
  """Gets the total # of elements in the network's trainable variables.

  Args:
    network: A `types.Network`.

  Returns:
    The total number of elements in the network's trainable variables.
  """
  num_elements_list = []
  for var in network.trainable_variables:
    num_elements = var.get_shape().num_elements()
    if num_elements is None:
      raise ValueError(
          f'Variable:{var} is expected to have a known shape, but found '
          'otherwise.')
    num_elements_list.append(num_elements)
  return sum(num_elements_list)


class FalconRewardPredictionPolicy(
    reward_prediction_base_policy.RewardPredictionBasePolicy):
  """Policy that samples actions based on the FALCON algorithm."""

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.NestedTensorSpec,
               reward_network: types.Network,
               exploitation_coefficient: types.FloatOrReturningFloat = 1.0,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               accepts_per_arm_features: bool = False,
               constraints: Iterable[constr.BaseConstraint] = (),
               emit_policy_info: Tuple[Text, ...] = (),
               num_samples_list: Sequence[tf.Variable] = (),
               name: Optional[Text] = None):
    """Builds a FalconRewardPredictionPolicy given a reward network.

    This policy takes a tf_agents.Network predicting rewards and samples an
    action based on predicted rewards with the action distribution described
    in Step 6 of Algorithm 1 in the paper:

    David Simchi-Levi and Yunzong Xu, "Bypassing the Monster: A Faster and
    Simpler Optimal Algorithm for Contextual Bandits under Realizability",
    Mathematics of Operations Research, 2021.
    https://arxiv.org/pdf/2003.12699.pdf

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      reward_network: An instance of a `tf_agents.network.Network`, callable via
        `network(observation, step_type) -> (output, final_state)`.
      exploitation_coefficient: float or callable that returns a float. Its
        value will be internally lower-bounded at 0. It controls how
        exploitative the policy behaves w.r.t the predicted rewards: A larger
        value makes the policy sample the greedy action (one with the best
        predicted reward) with a higher probability.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the network and 2) the
        mask.  The mask should be a 0-1 `Tensor` of shape `[batch_size,
        num_actions]`. This function should also work with a `TensorSpec` as
        input, and should output `TensorSpec` objects for the observation and
        mask.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
      constraints: iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.BaseConstraint`.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      num_samples_list: `Sequence` of tf.Variable's representing the number of
        examples for every action that the policy was trained with. For per-arm
        features, the size of the list is expected to be 1, representing the
        total number of examples the policy was trained with.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec` or the `BoundedTensorSpec` is not valid.
    """
    super(FalconRewardPredictionPolicy,
          self).__init__(time_step_spec, action_spec, reward_network,
                         observation_and_action_constraint_splitter,
                         accepts_per_arm_features, constraints,
                         emit_policy_info, name)

    self._exploitation_coefficient = exploitation_coefficient
    if num_samples_list:
      self._num_samples_list = num_samples_list
    else:
      self._num_samples_list = [tf.Variable(0, dtype=tf.int64)] * (
          1 if self.accepts_per_arm_features else self._expected_num_actions)
    if self.accepts_per_arm_features and len(self._num_samples_list) != 1:
      raise ValueError('num_samples_list is expected to be of length 1 when'
                       'accepts_per_arm_features is True, but found otherwise: '
                       f'{self._num_samples_list} ')
    if not self.accepts_per_arm_features and (len(self._num_samples_list) !=
                                              self._expected_num_actions):
      raise ValueError('Size of num_samples_list: ',
                       len(self._num_samples_list),
                       ' does not match the expected number of actions:',
                       self._expected_num_actions)
    self._num_trainable_elements = get_number_of_trainable_elements(
        self._reward_network)

  def _get_exploitation_coefficient(self) -> types.FloatOrReturningFloat:
    coef = self._exploitation_coefficient() if callable(
        self._exploitation_coefficient) else self._exploitation_coefficient
    coef = tf.cast(coef, dtype=tf.float32)
    return tf.maximum(coef, 0.0)

  @property
  def num_trainable_elements(self):
    return self._num_trainable_elements

  @property
  def num_samples_list(self):
    return self._num_samples_list

  def _get_number_of_allowed_actions(
      self, mask: Optional[types.Tensor]) -> types.Float:
    """Gets the number of allowed actions.

    Args:
      mask: An optional mask represented by a tensor shaped as [batch_size,
        num_actions].

    Returns:
      The number of allowed actions. It can be either a scalar (when `mask` is
      None), or a tensor shaped as [batch_size].
    """
    return (tf.cast(self._expected_num_actions, dtype=tf.float32)
            if mask is None else tf.reduce_sum(
                tf.cast(tf.cast(mask, tf.bool), tf.float32), axis=1))

  def _compute_gamma(self, mask: Optional[types.Tensor],
                     dtype: tf.DType) -> types.Float:
    """Computes the gamma parameter in the sampling probability.

    This helper method implements a simple heuristic for computing the
    the gamma parameter in Step 2 of Algorithm 1 in the paper
    https://arxiv.org/pdf/2003.12699.pdf. A higher gamma makes the action
    sampling distribution concentrate more on the greedy action.

    Args:
      mask: An optional mask represented by a tensor shaped as
        [batch_size, num_actions].
      dtype: Type of the returned value, expected to be a float type.

    Returns:
      The gamma parameter.
    """
    num_samples_list_float = tf.maximum(
        [tf.cast(x.read_value(), tf.float32) for x in self.num_samples_list],
        0.0)
    num_trainable_elements_float = tf.cast(
        tf.math.maximum(self.num_trainable_elements, 1), tf.float32)
    num_allowed_actions = self._get_number_of_allowed_actions(mask)
    return self._get_exploitation_coefficient() * tf.sqrt(
        num_allowed_actions * tf.reduce_sum(num_samples_list_float) /
        num_trainable_elements_float)

  def _action_distribution(self, mask, predicted_rewards):
    gamma = tf.expand_dims(
        self._compute_gamma(mask, predicted_rewards.dtype), axis=-1)
    batch_size = tf.shape(predicted_rewards)[0]
    # Replace predicted rewards of masked actions with -inf.
    predictions = predicted_rewards if mask is None else tf.where(
        tf.cast(mask, tf.bool), predicted_rewards, -float('Inf') *
        tf.ones_like(predicted_rewards))

    # Get the predicted rewards of the greedy actions.
    greedy_action_predictions = tf.reshape(
        tf.reduce_max(predictions, axis=-1), shape=[-1, 1])

    # `other_actions_probs` is a tensor shaped as [batch_size, num_actions] that
    # contains valid sampling probabilities for all non-greedy actions.
    num_allowed_actions = tf.expand_dims(
        self._get_number_of_allowed_actions(mask), axis=-1)
    other_actions_probs = tf.math.divide_no_nan(
        1.0,
        num_allowed_actions + gamma * (greedy_action_predictions - predictions))
    # Although `predictions` has accounted for the action mask, we still need
    # to mask the action probabilities in the case of zero gamma.
    other_actions_probs = (
        other_actions_probs if mask is None else tf.where(
            tf.cast(mask, tf.bool), other_actions_probs,
            tf.zeros_like(other_actions_probs)))

    # Get the greedy action.
    greedy_actions = tf.reshape(
        tf.argmax(predictions, axis=-1, output_type=self.action_spec.dtype),
        [-1, 1])

    # Compute the probabilities of sampling the greedy actions, which is
    # 1 - (the total probability of sampling other actions).
    greedy_action_prob = 1.0 - tf.reshape(
        tf.reduce_sum(other_actions_probs, axis=1), [-1, 1]) + tf.gather(
            other_actions_probs, greedy_actions, axis=1, batch_dims=1)

    # Compute the sampling probabilities for all actions by combining
    # `greedy_action_prob` and `other_actions_probs`.
    greedy_action_mask = tf.equal(
        tf.tile([
            tf.range(self._expected_num_actions, dtype=self.action_spec.dtype)
        ], [batch_size, 1]), greedy_actions)
    action_probs = tf.where(
        greedy_action_mask,
        tf.tile(greedy_action_prob, [1, self._expected_num_actions]),
        other_actions_probs)

    if self._action_offset != 0:
      distribution = shifted_categorical.ShiftedCategorical(
          probs=action_probs,
          dtype=self._action_spec.dtype,
          shift=self._action_offset)
    else:
      distribution = tfp.distributions.Categorical(
          probs=action_probs, dtype=self._action_spec.dtype)

    bandit_policy_values = tf.fill([batch_size, 1],
                                   policy_utilities.BanditPolicyType.FALCON)
    return distribution, bandit_policy_values
