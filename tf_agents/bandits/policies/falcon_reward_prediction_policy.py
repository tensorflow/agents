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

from typing import Iterable, Optional, Sequence, Text, Tuple

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.policies import constraints as constr
from tf_agents.bandits.policies import reward_prediction_base_policy
from tf_agents.distributions import shifted_categorical
from tf_agents.policies import utils as policy_utilities
from tf_agents.typing import types


# An upper bound of the gamma parameter. Without an upper bound, the probability
# of choosing non-greedy actions vanishes as the training data size increases,
# even when non-greedy actions have almost the same predicted rewards as the
# greedy action.
_MAX_GAMMA = 50000.0


# When trying to satisfy the constraint on the maximum exploration probability,
# the policy searches for the most suitable `exploitation_coefficient` on a grid
# in log2 scale between 0 and this value inclusively.
_MAX_LOG2_EXPLOITATION_COEF = 14


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
          'otherwise.'
      )
    num_elements_list.append(num_elements)
  return sum(num_elements_list)


def _find_action_probabilities(
    greedy_action_prob: types.Tensor,
    other_actions_probs: types.Tensor,
    max_exploration_prob: float,
):
  """Finds action probabilities satisfying `max_exploration_prob`.

  Given action probabilities calculated by different values of the gamma
  parameter, this function attempts to find action probabilities at a specific
  gamma value such that non-greedy actions are chosen with at most
  `max_exploration_prob` probability. If such an upper bound can be achieved,
  the return maximizes the exploration probability subject to the upper bound.
  Otherwise, it minimizes the exploration probability on a best-effort basis.

  Args:
    greedy_action_prob: A tensor shaped as [batch_size, d], the probabilities of
      choosing the greedy action under `d` different values of gamma.
    other_actions_probs: A tensor shaped as [batch_size, num_actions, d], all
      non-greedy action probabilities under `d` different values of gamma. The
      last dimension is assumed to be aligned with that of `greedy_action_prob`.
    max_exploration_prob: A float, the maximum probability of choosing
      non-greedy actions.

  Returns:
    A tuple of two tensors for the greedy action probability and non-greedy
    actions probabilities, shaped as [batch_size, 1] and
    [batch_size, num_actions], respectively.
  """
  if greedy_action_prob.shape.rank != 2:
    raise ValueError(
        '`greedy_action_prob` is expected to be rank-2, but found otherwise:'
        f' {greedy_action_prob}'
    )
  if other_actions_probs.shape.rank != 3:
    raise ValueError(
        '`other_actions_probs` is expected to be rank-3, but found otherwise:'
        f' {other_actions_probs}'
    )
  if greedy_action_prob.shape[-1] != other_actions_probs.shape[-1]:
    raise ValueError(
        '`greedy_action_prob` and `other_actions_probs` are '
        'expected to have the same last dimension, but found '
        f'otherwise. `greedy_action_prob`: {greedy_action_prob}'
        f', `other_actions_probs`: {other_actions_probs}'
    )

  # A [batch_size, d] bool tensor indicating which elements of
  # `greedy_action_prob` satisfy the `max_exploration_prob` constraint.
  valid_gamma_mask = tf.greater_equal(
      greedy_action_prob, 1.0 - max_exploration_prob
  )
  # A [batch_size] bool tensor indicating the batch members that have at least
  # one valid entry in `greedy_action_prob`.
  feasible = tf.greater(
      tf.reduce_sum(tf.cast(valid_gamma_mask, tf.float32), axis=1), 0.0
  )
  # We mask the probability entries corresponding to invalid gamma values by 2.0
  # so that they will not be selected as minimizers. See further details in the
  # comment below.
  greedy_action_prob_masked = tf.where(
      valid_gamma_mask,
      greedy_action_prob,
      2.0 * tf.ones_like(greedy_action_prob),
  )
  # For batch members where the `max_exploration_prob` constraint is feasible,
  # we maximize the exploration probability (or equivalently, minimize the
  # greedy action probability) subject to the constraint via masking.
  # For batch members where the `max_exploration_prob` constraint is infeasible,
  # we simply minimize the exploration probability (or equivalently, maximize
  # the greedy action probability).
  gamma_indices = tf.where(
      feasible,
      tf.argmin(greedy_action_prob_masked, axis=1),
      tf.argmax(greedy_action_prob, axis=1),
  )
  gamma_indices = tf.expand_dims(gamma_indices, axis=-1)
  greedy_action_prob = tf.gather(
      greedy_action_prob, gamma_indices, axis=1, batch_dims=1
  )
  num_actions = tf.shape(other_actions_probs)[1]
  other_actions_probs = tf.gather(
      other_actions_probs,
      tf.tile(gamma_indices, [1, num_actions]),
      axis=2,
      batch_dims=2,
  )
  return greedy_action_prob, other_actions_probs


class FalconRewardPredictionPolicy(
    reward_prediction_base_policy.RewardPredictionBasePolicy
):
  """Policy that samples actions based on the FALCON algorithm."""

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.NestedTensorSpec,
      reward_network: types.Network,
      exploitation_coefficient: Optional[types.FloatOrReturningFloat] = 1.0,
      max_exploration_probability_hint: Optional[
          types.FloatOrReturningFloat
      ] = None,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter
      ] = None,
      accepts_per_arm_features: bool = False,
      constraints: Iterable[constr.BaseConstraint] = (),
      emit_policy_info: Tuple[Text, ...] = (),
      num_samples_list: Sequence[tf.Variable] = (),
      name: Optional[Text] = None,
  ):
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
      max_exploration_probability_hint: An optional float, representing a hint
        on the maximum exploration probability, internally clipped to [0, 1].
        When this argument is set, `exploitation_coefficient` is ignored and the
        policy attempts to choose non-greedy actions with at most this
        probability. When such an upper bound cannot be achieved, e.g. due to
        insufficient training data, the policy attempts to minimize the
        probability of choosing non-greedy actions on a best-effort basis. For a
        demonstration of how it affects the policy behavior, see the unit test
        `testMaxExplorationProbabilityHint` in
        `falcon_reward_prediction_policy_test`.
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
    super(FalconRewardPredictionPolicy, self).__init__(
        time_step_spec,
        action_spec,
        reward_network,
        observation_and_action_constraint_splitter,
        accepts_per_arm_features,
        constraints,
        emit_policy_info,
        name,
    )

    self._exploitation_coefficient = exploitation_coefficient
    self._max_exploration_probability_hint = max_exploration_probability_hint
    if num_samples_list:
      self._num_samples_list = num_samples_list
    else:
      self._num_samples_list = [tf.Variable(0, dtype=tf.int64)] * (
          1 if self.accepts_per_arm_features else self._expected_num_actions
      )
    if self.accepts_per_arm_features and len(self._num_samples_list) != 1:
      raise ValueError(
          'num_samples_list is expected to be of length 1 when'
          'accepts_per_arm_features is True, but found otherwise: '
          f'{self._num_samples_list} '
      )
    if not self.accepts_per_arm_features and (
        len(self._num_samples_list) != self._expected_num_actions
    ):
      raise ValueError(
          'Size of num_samples_list: ',
          len(self._num_samples_list),
          ' does not match the expected number of actions:',
          self._expected_num_actions,
      )
    self._num_trainable_elements = get_number_of_trainable_elements(
        self._reward_network
    )

  def _get_exploitation_coefficient(self) -> types.FloatOrReturningFloat:
    coef = (
        self._exploitation_coefficient()
        if callable(self._exploitation_coefficient)
        else self._exploitation_coefficient
    )
    coef = tf.cast(coef, dtype=tf.float32)
    return tf.maximum(coef, 0.0)

  @property
  def num_trainable_elements(self):
    return self._num_trainable_elements

  @property
  def num_samples_list(self):
    return self._num_samples_list

  def _get_number_of_allowed_actions(
      self, mask: Optional[types.Tensor]
  ) -> types.Float:
    """Gets the number of allowed actions.

    Args:
      mask: An optional mask represented by a tensor shaped as [batch_size,
        num_actions].

    Returns:
      The number of allowed actions. It can be either a scalar (when `mask` is
      None), or a tensor shaped as [batch_size].
    """
    return (
        tf.cast(self._expected_num_actions, dtype=tf.float32)
        if mask is None
        else tf.reduce_sum(tf.cast(tf.cast(mask, tf.bool), tf.float32), axis=1)
    )

  def _compute_gamma(
      self, mask: Optional[types.Tensor], dtype: tf.DType, batch_size: int
  ) -> types.Float:
    """Computes the gamma parameter(s) in the sampling probability.

    This helper method implements a simple heuristic for computing the
    the gamma parameter in Step 2 of Algorithm 1 in the paper
    https://arxiv.org/pdf/2003.12699.pdf. A higher gamma makes the action
    sampling distribution concentrate more on the greedy action.

    Args:
      mask: An optional mask represented by a tensor shaped as [batch_size,
        num_actions].
      dtype: Type of the returned value, expected to be a float type.
      batch_size: The batch size.

    Returns:
      The gamma parameter shaped as [batch_size, d], where d = 1 if
      self._max_exploration_probability_hint is unset, and d > 1 otherwise. In
      the latter case, the second dimension gives gamma parameters calculated on
      a 1-D grid of `exploitation_coefficient` in log2 scale between 0 and
      `_MAX_LOG2_EXPLOITATION_COEF` inclusively, and `d` corresponds to the grid
      size.
    """
    num_samples_list_float = tf.maximum(
        [tf.cast(x.read_value(), tf.float32) for x in self.num_samples_list],
        0.0,
    )
    num_trainable_elements_float = tf.cast(
        tf.math.maximum(self.num_trainable_elements, 1), tf.float32
    )
    num_allowed_actions = self._get_number_of_allowed_actions(mask)
    exploitation_coefficient = (
        self._get_exploitation_coefficient()
        if self._max_exploration_probability_hint is None
        else tf.pow(2.0, range(_MAX_LOG2_EXPLOITATION_COEF + 1))
    )
    gamma = tf.sqrt(
        num_allowed_actions
        * tf.reduce_sum(num_samples_list_float)
        / num_trainable_elements_float
    )
    return tf.minimum(
        _MAX_GAMMA,
        tf.reshape(gamma, [-1, 1])
        * tf.ones(shape=[batch_size, 1], dtype=dtype)
        * tf.reshape(exploitation_coefficient, [1, -1]),
    )

  def _action_distribution(self, mask, predicted_rewards):
    batch_size = tf.shape(predicted_rewards)[0]
    gamma = self._compute_gamma(mask, predicted_rewards.dtype, batch_size)
    # Replace predicted rewards of masked actions with -inf.
    predictions = (
        predicted_rewards
        if mask is None
        else tf.where(
            tf.cast(mask, tf.bool),
            predicted_rewards,
            -float('Inf') * tf.ones_like(predicted_rewards),
        )
    )

    # Get the predicted rewards of the greedy actions.
    greedy_action_predictions = tf.reshape(
        tf.reduce_max(predictions, axis=-1), shape=[-1, 1]
    )

    # `other_actions_probs` is a tensor shaped as [batch_size, num_actions, d]
    # that contains valid sampling probabilities for all non-greedy actions.
    # The last dimension corresponds to different gamma parameters.
    if mask is not None:
      num_allowed_actions = tf.reshape(
          self._get_number_of_allowed_actions(mask), [batch_size, 1, 1]
      )
    else:
      num_allowed_actions = self._get_number_of_allowed_actions(mask)
    prediction_delta = greedy_action_predictions - predictions
    other_actions_probs = tf.math.divide_no_nan(
        1.0,
        num_allowed_actions
        + tf.matmul(
            tf.expand_dims(prediction_delta, axis=-1),
            tf.expand_dims(gamma, axis=1),
        ),
    )
    # Although `predictions` has accounted for the action mask, we still need
    # to mask the action probabilities in the case of zero gamma.
    if mask is not None:
      other_actions_probs = tf.where(
          tf.repeat(
              input=tf.expand_dims(tf.cast(mask, tf.bool), axis=-1),
              repeats=[tf.shape(other_actions_probs)[-1]],
              axis=2,
          ),
          other_actions_probs,
          tf.zeros_like(other_actions_probs),
      )

    # Get the greedy action.
    greedy_actions = tf.reshape(
        tf.argmax(predictions, axis=-1, output_type=self.action_spec.dtype),
        [-1, 1],
    )

    # Compute the probabilities of sampling the greedy actions, which is
    # 1 - (the total probability of sampling other actions),
    # shaped [batch_size, d].
    greedy_action_prob = (
        1.0
        - tf.reduce_sum(other_actions_probs, axis=1)
        + tf.squeeze(
            tf.gather(
                other_actions_probs, greedy_actions, axis=1, batch_dims=1
            ),
            axis=1,
        )
    )

    if self._max_exploration_probability_hint is not None:
      max_exploration_prob = tf.clip_by_value(
          self._max_exploration_probability_hint,
          clip_value_min=0.0,
          clip_value_max=1.0,
      )
      greedy_action_prob, other_actions_probs = _find_action_probabilities(
          greedy_action_prob, other_actions_probs, max_exploration_prob
      )
    else:
      other_actions_probs = tf.squeeze(other_actions_probs, axis=2)

    # Compute the sampling probabilities for all actions by combining
    # `greedy_action_prob` and `other_actions_probs`.
    greedy_action_mask = tf.equal(
        tf.tile(
            [
                tf.range(
                    self._expected_num_actions, dtype=self.action_spec.dtype
                )
            ],
            [batch_size, 1],
        ),
        greedy_actions,
    )
    action_probs = tf.where(
        greedy_action_mask,
        tf.tile(greedy_action_prob, [1, self._expected_num_actions]),
        other_actions_probs,
    )

    if self._action_offset != 0:
      distribution = shifted_categorical.ShiftedCategorical(
          probs=action_probs,
          dtype=self._action_spec.dtype,
          shift=self._action_offset,
      )
    else:
      distribution = tfp.distributions.Categorical(
          probs=action_probs, dtype=self._action_spec.dtype
      )

    bandit_policy_values = tf.fill(
        [batch_size, 1], policy_utilities.BanditPolicyType.FALCON
    )
    return distribution, bandit_policy_values
