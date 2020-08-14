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

"""Utilities for bandit policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.utils import common


class InfoFields(object):
  """Strings which can be used in the policy info fields."""
  LOG_PROBABILITY = policy_step.CommonFields.LOG_PROBABILITY
  # Mean of predicted rewards (per arm).
  PREDICTED_REWARDS_MEAN = 'predicted_rewards_mean'
  # Optimistic estimates of predicted rewards (per arm).
  PREDICTED_REWARDS_OPTIMISTIC = 'predicted_rewards_optimistic'
  # Samples of predicted rewards (per arm).
  PREDICTED_REWARDS_SAMPLED = 'predicted_rewards_sampled'
  # Type of bandit policy (see enumerations in `BanditPolicyType`).
  BANDIT_POLICY_TYPE = 'bandit_policy_type'
  # Used to store the chosen action for a per-arm model.
  CHOSEN_ARM_FEATURES = 'chosen_arm_features'


PolicyInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'PolicyInfo',
    (InfoFields.LOG_PROBABILITY,
     InfoFields.PREDICTED_REWARDS_MEAN,
     InfoFields.PREDICTED_REWARDS_OPTIMISTIC,
     InfoFields.PREDICTED_REWARDS_SAMPLED,
     InfoFields.BANDIT_POLICY_TYPE))
# Set default empty tuple for all fields.
PolicyInfo.__new__.__defaults__ = ((),) * len(PolicyInfo._fields)


PerArmPolicyInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'PerArmPolicyInfo',
    (InfoFields.LOG_PROBABILITY,
     InfoFields.PREDICTED_REWARDS_MEAN,
     InfoFields.PREDICTED_REWARDS_OPTIMISTIC,
     InfoFields.PREDICTED_REWARDS_SAMPLED,
     InfoFields.BANDIT_POLICY_TYPE,
     InfoFields.CHOSEN_ARM_FEATURES))
# Set default empty tuple for all fields.
PerArmPolicyInfo.__new__.__defaults__ = ((),) * len(PerArmPolicyInfo._fields)


def populate_policy_info(arm_observations, chosen_actions, rewards_for_argmax,
                         est_rewards, emit_policy_info,
                         accepts_per_arm_features):
  """Populates policy info given all needed input.

  Args:
    arm_observations: In case the policy accepts per-arm feautures, this is a
      Tensor with the per-arm features. Otherwise its value is unused.
    chosen_actions: A Tensor with the indices of the chosen actions.
    rewards_for_argmax: The sampled or optimistically boosted reward estimates
      based on which the policy chooses the action greedily.
    est_rewards: A Tensor with the rewards estimated by the model.
    emit_policy_info: A set of policy info keys, specifying wich info fields to
      populate
    accepts_per_arm_features: (bool) Whether the policy accepts per-arm
      features.

  Returns:
    A policy info.
  """
  if accepts_per_arm_features:
    # Saving the features for the chosen action to the policy_info.
    chosen_arm_features = tf.nest.map_structure(
        lambda t: tf.gather(params=t, indices=chosen_actions, batch_dims=1),
        arm_observations)
    policy_info = PerArmPolicyInfo(
        predicted_rewards_optimistic=(
            rewards_for_argmax
            if InfoFields.PREDICTED_REWARDS_OPTIMISTIC in emit_policy_info else
            ()),
        predicted_rewards_sampled=(
            rewards_for_argmax if
            InfoFields.PREDICTED_REWARDS_SAMPLED in emit_policy_info else ()),
        predicted_rewards_mean=(
            est_rewards
            if InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info else ()),
        chosen_arm_features=chosen_arm_features)
  else:
    policy_info = PolicyInfo(
        predicted_rewards_optimistic=(
            rewards_for_argmax
            if InfoFields.PREDICTED_REWARDS_OPTIMISTIC in emit_policy_info else
            ()),
        predicted_rewards_sampled=(
            rewards_for_argmax if
            InfoFields.PREDICTED_REWARDS_SAMPLED in emit_policy_info else ()),
        predicted_rewards_mean=(
            est_rewards
            if InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info else ()))
  return policy_info


class BanditPolicyType(object):
  """Enumeration of bandit policy types."""
  # No bandit policy type specified.
  UNKNOWN = 0
  # Greedy decision made by bandit agent.
  GREEDY = 1
  # Random decision for exploration made by epsilon-greedy agent sampled from
  # uniform distribution over actions.
  UNIFORM = 2


def create_bandit_policy_type_tensor_spec(shape):
  """Create tensor spec for bandit policy type."""
  return tensor_spec.BoundedTensorSpec(
      shape=shape, dtype=tf.int32,
      minimum=BanditPolicyType.UNKNOWN, maximum=BanditPolicyType.UNIFORM)


@common.function
def masked_argmax(input_tensor, mask, output_type=tf.int32):
  """Computes the argmax where the allowed elements are given by a mask.

  If a row of `mask` contains all zeros, then this method will return -1 for the
  corresponding row of `input_tensor`.

  Args:
    input_tensor: Rank-2 Tensor of floats.
    mask: 0-1 valued Tensor of the same shape as input.
    output_type: Integer type of the output.

  Returns:
    A Tensor of rank 1 and type `output_type`, with the masked argmax of every
    row of `input_tensor`.
  """
  input_tensor.shape.assert_is_compatible_with(mask.shape)
  neg_inf = tf.constant(-float('Inf'), input_tensor.dtype)
  modified_input = tf.compat.v2.where(
      tf.cast(mask, tf.bool), input_tensor, neg_inf)
  argmax_tensor = tf.argmax(modified_input, axis=-1, output_type=output_type)
  # Replace results for invalid mask rows with -1.
  reduce_mask = tf.cast(tf.reduce_max(mask, axis=1), tf.bool)
  neg_one = tf.constant(-1, output_type)
  return tf.compat.v2.where(reduce_mask, argmax_tensor, neg_one)


def has_bandit_policy_type(info, check_for_tensor=False):
  """Check if policy info has `bandit_policy_type` field/tensor."""
  if info in ((), None):
    return False
  fields = getattr(info, '_fields', None)
  has_field = fields is not None and InfoFields.BANDIT_POLICY_TYPE in fields
  if has_field and check_for_tensor:
    return isinstance(info.bandit_policy_type, tf.Tensor)
  else:
    return has_field


def has_chosen_arm_features(info, check_for_tensor=False):
  """Check if policy info has `chosen_arm_features` field/tensor."""
  if info in ((), None):
    return False
  fields = getattr(info, '_fields', None)
  has_field = fields is not None and InfoFields.CHOSEN_ARM_FEATURES in fields
  if has_field and check_for_tensor:
    return isinstance(info.chosen_arm_features, tf.Tensor)
  else:
    return has_field


def set_bandit_policy_type(info, bandit_policy_type):
  """Sets the InfoFields.BANDIT_POLICY_TYPE on info to bandit_policy_type.

  If policy `info` does not support InfoFields.BANDIT_POLICY_TYPE, this method
  returns `info` as-is (without any modification).

  Args:
    info: Policy info on which to set bandit policy type.
    bandit_policy_type: Tensor containing BanditPolicyType enums or TensorSpec
      from `create_bandit_policy_type_tensor_spec()`.

  Returns:
    Policy info with modified field (if possible).
  """
  if info in ((), None):
    return PolicyInfo(bandit_policy_type=bandit_policy_type)
  fields = getattr(info, '_fields', None)
  if fields is not None and InfoFields.BANDIT_POLICY_TYPE in fields:
    return info._replace(bandit_policy_type=bandit_policy_type)
  try:
    info[InfoFields.BANDIT_POLICY_TYPE] = bandit_policy_type
  except TypeError:
    pass
  return info


@common.function
def bandit_policy_uniform_mask(values, mask):
  """Set bandit policy type tensor to BanditPolicyType.UNIFORM based on mask.

  Set bandit policy type `values` to BanditPolicyType.UNIFORM; returns tensor
  where output[i] is BanditPolicyType.UNIFORM if mask[i] is True, otherwise it
  is left as values[i].

  Args:
    values: Tensor containing `BanditPolicyType` enumerations.
    mask: Tensor of the same shape as `values` with boolean flags indicating
      values to set to `BanditPolicyType.UNIFORM`.

  Returns:
    Tensor containing `BanditPolicyType` enumerations with masked values.
  """
  return tf.where(
      mask, tf.fill(tf.shape(values), BanditPolicyType.UNIFORM), values)


def get_model_index(arm_index, accepts_per_arm_features):
  """Returns the model index for a specific arm.

  The number of models depends on the observation format: If the policy accepts
  per-arm features, there is only one single model used for every arm. Otherwise
  there is a model for every arm.

  Args:
    arm_index: The index of the arm for which the model index is needed.
    accepts_per_arm_features: (bool) Whether the policy works with per-arm
      features.

  Returns:
    The index of the model for the arm requested.
  """
  return 0 if accepts_per_arm_features else arm_index


def compute_feasibility_probability(observation, constraints, batch_size,
                                    num_actions, action_mask=None):
  """Helper function to compute the action feasibility probability."""
  feasibility_prob = tf.ones([batch_size, num_actions])
  if action_mask is not None:
    feasibility_prob = tf.cast(action_mask, tf.float32)
  for c in constraints:
    # We assume the constraints are independent.
    action_feasibility = c(observation)
    feasibility_prob *= action_feasibility
  return feasibility_prob


def construct_mask_from_multiple_sources(
    observation, observation_and_action_constraint_splitter, constraints,
    max_num_actions):
  """Constructs an action mask from multiple sources.

  The sources include:
  -- The action mask encoded in the observation,
  -- the `num_actions` feature restricting the number of actions per sample,
  -- the feasibility mask implied by constraints.

  The resulting mask disables all actions that are masked out in any of the
  three sources.

  Args:
    observation: A nest of Tensors containing the observation.
    observation_and_action_constraint_splitter: The observation action mask
      splitter function if the observation has action mask.
    constraints: Iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.NeuralConstraint`.
    max_num_actions: The maximum number of actions per sample.

  Returns:
    An action mask in the form of a `[batch_size, max_num_actions]` 0-1 tensor.
  """
  mask = None
  if observation_and_action_constraint_splitter is not None:
    observation, mask = observation_and_action_constraint_splitter(observation)
  elif (isinstance(observation, dict) and
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY in observation):
    number_of_actions = observation[bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY]
    mask = tf.sequence_mask(
        lengths=number_of_actions, maxlen=max_num_actions, dtype=tf.int32)

  first_observation = tf.nest.flatten(observation)[0]
  batch_size = tf.shape(first_observation)[0]
  if constraints:
    feasibility_prob = compute_feasibility_probability(
        observation, constraints, batch_size,
        max_num_actions, mask)
    # Probabilistic masking.
    mask = tfp.distributions.Bernoulli(probs=feasibility_prob).sample()
  return mask


def create_chosen_arm_features_info_spec(observation_spec):
  """Creates the chosen arm features info spec from the arm observation spec."""
  arm_spec = observation_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY]
  return tensor_spec.remove_outer_dims_nest(arm_spec, 1)


def check_no_mask_with_arm_features(accepts_per_arm_features,
                                    observation_and_action_constraint_splitter):
  if accepts_per_arm_features and (observation_and_action_constraint_splitter is
                                   not None):
    raise NotImplementedError(
        'Action masking is not allowed for Bandit problems with arm features. '
        'To implement a policy that handles variable number of actions, please '
        'use the `{}` feature key.'.format(
            bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY))
