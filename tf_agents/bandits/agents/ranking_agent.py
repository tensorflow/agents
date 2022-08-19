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

"""Ranking agent.

This agent trains ranking policies. The policy has a scoring network used for
scoring items. Some of these items will then be selected based on scores and
similarity. The agent receives feedback based on which item in a recommendation
list was interacted with. In this agent we assume either a `score_vector` or a
`cascading feedback` framework. In the former case, the feedback is a vector of
scores for every item in the slots. In the latter case, if the kth item was
clicked, then the items up to k-1 receive a score of -1, the kth item receives
a score based on a feedback value, while the rest of the items receive feedback
of 0. The task of the agent is to train the scoring network to be able to
estimate the above scores.

The observation the agent ingests contains the global features and the features
of the items in the recommendation slots. The item features are stored in the
`per_arm` part of the observation, in the order of how they are recommended.
Since this ordered list of items expresses what action was taken by the policy,
the `action` value of the trajectory is not used by the agent.

Note the difference between the per-arm part of the observation received by the
policy and the agent: While the agent receives the items in the recommendation
slots (as explained above), the policy receives the items that are available for
recommendation. The user is responsible for converting the observation to the
syntax required by the agent.
"""
import enum
from typing import Optional, Text

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents import tf_agent
from tf_agents.bandits.policies import ranking_policy
from tf_agents.specs import bandit_spec_utils
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

CHOSEN_INDEX = 'chosen_index'
CHOSEN_VALUE = 'chosen_value'


def compute_score_tensor_for_cascading(
    chosen_index: types.Int,
    chosen_value: types.Float,
    num_slots: int,
    non_click_score: float = -1.) -> types.Float:
  """Gives scores for all items in a batch.

  The score of items that are before the chosen index is `-1`, the score of
  the chosen values are given by `chosen_value`. The rest of the items receive
  a score of `0`. TODO(b/206685896): Normalize theses scores or let the user
  selected the negative feedback reward.

  Args:
    chosen_index: The index of the slot chosen, or `num_slots` if no slot is
      chosen.
    chosen_value: The value of the chosen item.
    num_slots: The number of slots. The output score vector will have shape
      `[batch_size, num_slots]`.
    non_click_score: (float) The score value for items lying "before" the
      clicked item. If not set, -1 is used. It is recommended (but not enforced)
      to use a negative value.

  Returns:
    A tensor of shape `[batch_size, num_slots]`, with scores for every item in
    the recommendation.
  """
  negatives = tf.sequence_mask(
      chosen_index, maxlen=num_slots, dtype=tf.float32)

  chosen_onehot = tf.one_hot(chosen_index, num_slots)
  diag_value = tf.linalg.diag(chosen_value)
  values = tf.linalg.matmul(diag_value, chosen_onehot)
  return values + non_click_score * negatives


class RankingPolicyType(enum.Enum):
  """Enumeration of ranking policy types."""
  # No policy type specified.
  UNKNOWN = 0
  # The policy is an instance of `PenalizeCosineDistanceRankingPolicy`.
  COSINE_DISTANCE = 1
  # No penalty applied.
  NO_PENALTY = 2
  # Sorts the items based on their scores deterministically.
  DESCENDING_SCORES = 3


class FeedbackModel(enum.Enum):
  """Enumeration of feedback models."""
  # No feedback model specified.
  UNKNOWN = 0
  # Cascading feedback model: A tuple of the chosen index and its value.
  CASCADING = 1
  # Score vector feedback model: an explicit score for every item in the slots.
  SCORE_VECTOR = 2


class RankingAgent(tf_agent.TFAgent):
  """Ranking agent class."""

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      scoring_network: types.Network,
      optimizer: types.Optimizer,
      policy_type: RankingPolicyType = RankingPolicyType.COSINE_DISTANCE,
      error_loss_fn: types.LossFn = tf.compat.v1.losses.mean_squared_error,
      feedback_model: FeedbackModel = FeedbackModel.CASCADING,
      non_click_score: Optional[float] = None,
      logits_temperature: float = 1.,
      summarize_grads_and_vars: bool = False,
      enable_summaries: bool = True,
      train_step_counter: Optional[tf.Variable] = None,
      penalty_mixture_coefficient: float = 1.,
      name: Optional[Text] = None):
    """Initializes an instance of RankingAgent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      scoring_network: The network that outputs scores for items.
      optimizer: The optimizer for the agent.
      policy_type: The type of policy used. The only available type at this
        moment is COSINE_DISTANCE, that invokes the
        `PenalizeCosineDistanceRankingPolicy`. This policy uses the cosine
        similarity to penalize the scores of not yet selected items. If set to
        `UNKNOWN`, falls back to `COSINE_DISTANCE`.
      error_loss_fn: The loss function used.
      feedback_model: The type of feedback model. Implemented models are:
        -- CASCADING: the feedback is a tuple `(k, v)`, where `k` is the index
          of the chosen item, and `v` is the value of the choice. If no item was
          chosen, then `k=num_slots` is used and `v` is ignored.
        -- SCORE_VECTOR: the feedback is a vector of length `num_slots`,
          containing scores for every item in the recommendation. If set to
          `UNKNOWN`, falls back to CASCADING`.
      non_click_score: (float) For the cascading feedback model, this is the
        score value for items lying "before" the clicked item. If not set, -1 is
        used. It is recommended (but not enforced) to use a negative value.
      logits_temperature: temperature parameter for non-deterministic policies.
        This value must be positive.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      enable_summaries: A Python bool, default True. When False, all summaries
        (debug or otherwise) should not be written.
      train_step_counter: An optional `tf.Variable` to increment every time the
        train op is run.  Defaults to the `global_step`.
      penalty_mixture_coefficient: A parameter responsible for the balance
        between selecting high scoring items and enforcing diverisity. Used Only
        by diversity-based policies.
      name: The name of this agent instance.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._num_items = time_step_spec.observation[
        bandit_spec_utils.PER_ARM_FEATURE_KEY].shape[0]
    self._num_slots = action_spec.shape[0]
    self._item_feature_dim = time_step_spec.observation[
        bandit_spec_utils.PER_ARM_FEATURE_KEY].shape[-1]
    self._global_feature_dim = time_step_spec.observation[
        bandit_spec_utils.GLOBAL_FEATURE_KEY].shape[-1]
    self._use_num_actions = (
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY in time_step_spec.observation)

    scoring_network.create_variables()
    self._scoring_network = scoring_network
    self._optimizer = optimizer
    self._error_loss_fn = error_loss_fn
    if feedback_model == FeedbackModel.UNKNOWN:
      feedback_model = FeedbackModel.CASCADING
    self._feedback_model = feedback_model
    if feedback_model == FeedbackModel.CASCADING:
      if non_click_score is None:
        self._non_click_score = -1.0
      else:
        self._non_click_score = non_click_score
    else:
      if non_click_score is not None:
        raise ValueError('Parameter `non_click_score` should only be used '
                         'together with CASCADING feedback model.')
    if policy_type == RankingPolicyType.UNKNOWN:
      policy_type = RankingPolicyType.COSINE_DISTANCE
    if policy_type == RankingPolicyType.COSINE_DISTANCE:
      policy = ranking_policy.PenalizeCosineDistanceRankingPolicy(
          self._num_items,
          self._num_slots,
          time_step_spec,
          scoring_network,
          penalty_mixture_coefficient=penalty_mixture_coefficient,
          logits_temperature=logits_temperature)
    elif policy_type == RankingPolicyType.NO_PENALTY:
      policy = ranking_policy.NoPenaltyRankingPolicy(
          self._num_items,
          self._num_slots,
          time_step_spec,
          scoring_network,
          logits_temperature=logits_temperature)
    elif policy_type == RankingPolicyType.DESCENDING_SCORES:
      policy = ranking_policy.DescendingScoreRankingPolicy(
          self._num_items, self._num_slots, time_step_spec, scoring_network)
    else:
      raise NotImplementedError(
          'Policy type {} is not implemented'.format(policy_type))
    use_num_actions = isinstance(
        time_step_spec.observation,
        dict) and 'num_actions' in time_step_spec.observation
    training_obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        self._global_feature_dim, self._item_feature_dim, self._num_slots,
        use_num_actions)
    training_time_step_spec = ts.time_step_spec(
        training_obs_spec, reward_spec=time_step_spec.reward)
    training_data_spec = trajectory.from_transition(
        training_time_step_spec,
        policy_step.PolicyStep(),
        training_time_step_spec)
    super(RankingAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=None,
        training_data_spec=training_data_spec,
        summarize_grads_and_vars=summarize_grads_and_vars,
        enable_summaries=enable_summaries,
        train_step_counter=train_step_counter)

  def _variables_to_train(self):
    return self._scoring_network.trainable_variables

  def _loss(self,
            experience: types.NestedTensor,
            weights: Optional[types.Tensor] = None,
            training: bool = False) -> tf_agent.LossInfo:
    """Computes loss for training the reward and constraint networks.

    Args:
      experience: A batch of experience data in the form of a `Trajectory` or
        `Transition`.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and the
        final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      A `LossInfo` containing the loss for the training step.

    Raises:
      ValueError:
        if the number of actions is greater than 1.
    """
    flat_obs, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.observation, self._training_data_spec.observation)
    flat_reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._training_data_spec.reward)
    if self._feedback_model == FeedbackModel.CASCADING:
      chosen_index = flat_reward[CHOSEN_INDEX]
      chosen_value = flat_reward[CHOSEN_VALUE]
      score = compute_score_tensor_for_cascading(
          chosen_index,
          chosen_value,
          self._num_slots,
          non_click_score=self._non_click_score)
    elif self._feedback_model == FeedbackModel.SCORE_VECTOR:
      score = flat_reward
    weights = self._construct_sample_weights(flat_reward, flat_obs, weights)

    est_reward = self._scoring_network(flat_obs, training)[0]
    return tf_agent.LossInfo(
        tf.reduce_sum(
            tf.multiply(
                self._error_loss_fn(
                    est_reward,
                    score,
                    reduction=tf.compat.v1.losses.Reduction.NONE), weights)) /
        tf.reduce_sum(weights),
        extra=())

  def _train(self, experience, weights):
    with tf.GradientTape() as tape:
      loss = self._loss(experience, weights, training=True).loss
    if self.summaries_enabled:
      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='loss', data=loss, step=self.train_step_counter)
    gradients = tape.gradient(loss, self._variables_to_train())
    grads_and_vars = tuple(zip(gradients, self._variables_to_train()))
    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)
    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)
    return tf_agent.LossInfo(loss, extra=())

  def _construct_sample_weights(self, reward, observation, weights):
    batch_size = tf.shape(tf.nest.flatten(reward)[0])[0]
    if weights is None:
      weights = tf.ones([batch_size, self._num_slots])
    elif not list(weights.shape):
      weights = weights * tf.ones([batch_size, self._num_slots])
    else:
      tf.debugging.assert_equal(tf.shape(weights), [batch_size])
      weights = tf.reshape(weights, shape=[-1, 1])
      weights = tf.tile(weights, multiples=[1, self._num_slots])

    if self._use_num_actions:
      num_slotted_items = observation[bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY]
      weights = tf.sequence_mask(
          num_slotted_items, self._num_slots, dtype=tf.float32) * weights
    if self._feedback_model == FeedbackModel.CASCADING:
      chosen_index = tf.reshape(reward[CHOSEN_INDEX], shape=[-1])
      multiplier = tf.sequence_mask(
          chosen_index + 1, self._num_slots, dtype=tf.float32)
      weights = multiplier * weights

    return weights
