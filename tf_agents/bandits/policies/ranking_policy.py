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

"""Ranking policy."""
from typing import Optional, Sequence, Text

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utils
from tf_agents.specs import bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

tfd = tfp.distributions


class PenalizedPlackettLuce(tfd.PlackettLuce):
  """A distribution that samples permutations and penalizes item scores.

  This distribution samples elements of a permutation incrementally, and after
  every sample it penalizes the scores of the remaining items by the similarity
  of already chosen items.
  """

  def __init__(self,
               features: types.Tensor,
               num_slots: int,
               logits: types.Tensor,
               penalty_mixture_coefficient: float = 1.):
    """Initializes an instance of PenalizedPlackettLuce.

    Args:
      features: Item features based on which similarity is calculated.
      num_slots: The number of slots to fill: this many items will be sampled.
      logits: Unnormalized log probabilities for the PlackettLuce distribution.
        Shape is `[num_items]`.
      penalty_mixture_coefficient: A parameter responsible for the balance
        between selecting high scoring items and enforcing diverisity.
    """
    self._features = features
    self._num_slots = num_slots
    self._penalty_mixture_coefficient = penalty_mixture_coefficient
    super(PenalizedPlackettLuce, self).__init__(scores=logits)

  def _penalizer_fn(self, logits: types.Float, features: types.Float,
                    slots: Sequence[types.Int]):
    """Downscores items by their similarity to already selected items.

    Args:
      logits: The current logits of all items.
      features: the feature vectors of the items.
      slots: list of indices of already selected items.

    Returns:
      New logits.
    """
    raise NotImplementedError()

  def _sample_n(self, n, seed=None):
    logits = tf.convert_to_tensor(self.scores)
    sample_shape = tf.concat([[n], tf.shape(logits)], axis=0)
    slots = []
    for _ in range(self._num_slots):
      items = tfd.Categorical(logits=logits).sample()
      slots.append(items)
      logits -= tf.one_hot(items, sample_shape[-1], on_value=np.inf)
      logits = self._penalizer_fn(logits, self._features, slots)
    sample = tf.expand_dims(tf.stack(slots, axis=-1), axis=0)
    return sample

  def _event_shape(self, scores=None):
    return self._num_slots


class CosinePenalizedPlackettLuce(PenalizedPlackettLuce):
  """A distribution that samples items based on scores and cosine similarity."""

  def _penalizer_fn(self, logits, features, slots):
    num_items = logits.shape[-1]
    num_slotted = len(slots)
    slot_tensor = tf.stack(slots, axis=-1)
    # The tfd.Categorical distribution will give the sample `num_items` if all
    # the logits are `-inf`. Hence, we need to apply minimum. This happens when
    # `num_actions` is less than `num_slots`. To this end, the action taken by
    # the policy always has to be taken together with the `num_actions`
    # observation, to know how many slots are filled with valid items.
    slotted_features = tf.gather(
        features, tf.minimum(slot_tensor, num_items - 1), batch_dims=1)

    # Calculate the similarity between all pairs from
    # `slotted_features x all_features`.
    all_sims = tf.keras.losses.cosine_similarity(
        tf.repeat(features, num_slotted, axis=1),
        tf.tile(slotted_features, [1, num_items, 1])) - 1

    sim_matrix = tf.reshape(all_sims, shape=[-1, num_items, num_slotted])
    similarity_boosts = tf.reduce_min(sim_matrix, axis=-1)
    adjusted_logits = logits + (
        self._penalty_mixture_coefficient * similarity_boosts)
    return adjusted_logits


class NoPenaltyPlackettLuce(tfd.PlackettLuce):
  """Identical to PlackettLuce, with input signature modified to our needs."""

  def __init__(self,
               features: types.Tensor,
               num_slots: int,
               logits: types.Tensor,
               penalty_mixture_coefficient: float = 1.):
    """Initializes an instance of NoPenaltyPlackettLuce.

    Args:
      features: Unused for this distribution.
      num_slots: The number of slots to fill: this many items will be sampled.
      logits: Unnormalized log probabilities for the PlackettLuce distribution.
        Shape is `[num_items]`.
      penalty_mixture_coefficient: Unused for this distribution.
    """
    self._num_slots = num_slots
    super(NoPenaltyPlackettLuce, self).__init__(scores=tf.math.exp(logits))

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    return super(NoPenaltyPlackettLuce, self).sample(
        sample_shape, seed, name, **kwargs)[:, :self._num_slots]


class RankingPolicy(tf_policy.TFPolicy):
  """A class implementing ranking policies in TF Agents.

  The ranking policy needs at initialization the number of items per round to
  rank, a scorer network, and a score penalizer function. This function should
  ensure that similar items don't all get high scores and thus a diverse set of
  items is recommended.

  In the case the number of items to rank varies from iteration to iteration,
  the observation contains a `num_actions` value, that specifies the number of
  items available. Note that in this case it can happen that the number of
  ranked items is less than the number of slots. Thus, consumers of the output
  of `policy.action` should always use the `num_actions` value to determine what
  part of the output is the action ranking.

  If `num_actions` field is not used, the policy is always presented with
  `num_items` many items, which should be greater than or equal to `num_slots`.
  """

  def __init__(self,
               num_items: int,
               num_slots: int,
               time_step_spec: types.TimeStep,
               network: types.Network,
               item_sampler: tfd.Distribution,
               penalty_mixture_coefficient: float = 1.,
               logits_temperature: float = 1.,
               name: Optional[Text] = None):
    """Initializes an instance of `RankingPolicy`.

    Args:
      num_items: The number of items the policy can choose from, to be slotted.
      num_slots: The number of recommendation slots presented to the user, i.e.,
        chosen by the policy.
      time_step_spec: The time step spec.
      network: The network that estimates scores of items, given global and item
        features.
      item_sampler: A distibution that takes scores and item features, and
        samples an ordered list of `num_slots` items. Similarity penalization
        can be implemented within this sampler.
      penalty_mixture_coefficient: A parameter responsible for the balance
        between selecting high scoring items and enforcing diverisity.
      logits_temperature: The "temperature" parameter for sampling. All the
        logits will be divided by this float value. This value must be positive.
      name: The name of this policy instance.
    """
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(num_slots,), dtype=tf.int32, minimum=0, maximum=num_items - 1)
    info_spec = policy_utils.PolicyInfo(
        predicted_rewards_mean=tensor_spec.TensorSpec(
            shape=(num_slots,), dtype=tf.float32))
    network.create_variables()
    self._network = network
    assert num_slots <= num_items, (
        'The number of slots have to be less than or equal to the number of '
        'items.')
    self._num_slots = num_slots
    self._num_items = num_items
    self._item_sampler = item_sampler
    self._penalty_mixture_coefficient = penalty_mixture_coefficient
    if logits_temperature <= 0:
      raise ValueError(
          f'logits_temperature must be positive; was {logits_temperature}')
    self._logits_temperature = logits_temperature
    if bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY in time_step_spec.observation:
      self._use_num_actions = True
    else:
      self._use_num_actions = False
    super(RankingPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        name=name,
        info_spec=info_spec)

  @property
  def num_slots(self):
    return self._num_slots

  def _distribution(self, time_step, policy_state):
    observation = time_step.observation
    scores, _ = self._network(observation, time_step.step_type, policy_state)
    if self._use_num_actions:
      num_actions = time_step.observation[
          bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY]
      masked_scores = tf.where(
          tf.sequence_mask(num_actions, maxlen=self._num_items), scores,
          tf.fill(tf.shape(scores), -np.inf))
    else:
      masked_scores = scores
    masked_scores = masked_scores / self._logits_temperature
    return policy_step.PolicyStep(
        self._item_sampler(observation[bandit_spec_utils.PER_ARM_FEATURE_KEY],
                           self._num_slots, masked_scores,
                           self._penalty_mixture_coefficient),
        (),
        # TODO(b/197787556): potentially add other side info tensors
        policy_utils.PolicyInfo(predicted_rewards_mean=scores))


class PenalizeCosineDistanceRankingPolicy(RankingPolicy):
  """A Ranking policy that penalizes scores based on cosine distance.

  Note that this is a rough first implementation, and thus it is very slow and
  also misses tunable parameters such as weights of the penalties vs raw scores.
  """

  def __init__(self,
               num_items: int,
               num_slots: int,
               time_step_spec: types.TimeStep,
               network: types.Network,
               penalty_mixture_coefficient: float = 1.,
               logits_temperature: float = 1.,
               name: Optional[Text] = None):
    super(PenalizeCosineDistanceRankingPolicy, self).__init__(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network,
        item_sampler=CosinePenalizedPlackettLuce,
        penalty_mixture_coefficient=penalty_mixture_coefficient,
        logits_temperature=logits_temperature,
        name=name)


class NoPenaltyRankingPolicy(RankingPolicy):

  def __init__(self,
               num_items: int,
               num_slots: int,
               time_step_spec: types.TimeStep,
               network: types.Network,
               logits_temperature: float = 1.,
               name: Optional[Text] = None):
    super(NoPenaltyRankingPolicy, self).__init__(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network,
        item_sampler=NoPenaltyPlackettLuce,
        logits_temperature=logits_temperature,
        name=name)


class DescendingScoreSampler(tf.Module):

  def __init__(self, unused_features: types.Tensor, num_slots: int,
               scores: types.Tensor, unused_penalty_mixture_coefficient: float):
    self._scores = scores
    self._num_slots = num_slots

  def sample(self, shape=(), seed=None):
    return tf.math.top_k(self._scores, k=self._num_slots).indices


class DescendingScoreRankingPolicy(RankingPolicy):
  """A policy that is deterministically ranks elements based on their scores."""

  def __init__(self,
               num_items: int,
               num_slots: int,
               time_step_spec: types.TimeStep,
               network: types.Network,
               name: Optional[Text] = None):
    super(DescendingScoreRankingPolicy, self).__init__(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network,
        item_sampler=DescendingScoreSampler,
        name=name)
