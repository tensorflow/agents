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
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.utils import common


class InfoFields(object):
  """Strings which can be used in the policy info fields."""
  # Mean of predicted rewards (per arm).
  PREDICTED_REWARDS_MEAN = 'predicted_rewards_mean'
  # Samples of predicted rewards (per arm).
  PREDICTED_REWARDS_SAMPLED = 'predicted_rewards_sampled'
  # Type of bandit policy (see enumerations in `BanditPolicyType`).
  BANDIT_POLICY_TYPE = 'bandit_policy_type'
  # Used to store the chosen action for a per-arm model.
  CHOSEN_ARM_FEATURES = 'chosen_arm_features'


PolicyInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'PolicyInfo',
    (policy_step.CommonFields.LOG_PROBABILITY,
     InfoFields.PREDICTED_REWARDS_MEAN,
     InfoFields.PREDICTED_REWARDS_SAMPLED,
     InfoFields.BANDIT_POLICY_TYPE))
# Set default empty tuple for all fields.
PolicyInfo.__new__.__defaults__ = ((),) * len(PolicyInfo._fields)


PerArmPolicyInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'PerArmPolicyInfo',
    (policy_step.CommonFields.LOG_PROBABILITY,
     InfoFields.PREDICTED_REWARDS_MEAN,
     InfoFields.PREDICTED_REWARDS_SAMPLED,
     InfoFields.BANDIT_POLICY_TYPE,
     InfoFields.CHOSEN_ARM_FEATURES))
# Set default empty tuple for all fields.
PerArmPolicyInfo.__new__.__defaults__ = ((),) * len(PerArmPolicyInfo._fields)


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
