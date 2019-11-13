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

import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.utils import common


class InfoFields(object):
  """Strings which can be used in the policy info fields."""
  # Mean of predicted rewards (per arm).
  PREDICTED_REWARDS_MEAN = 'predicted_rewards_mean'
  # Samples of predicted rewards (per arm).
  PREDICTED_REWARDS_SAMPLED = 'predicted_rewards_sampled'


PolicyInfo = collections.namedtuple(  # pylint: disable=invalid-name
    'PolicyInfo',
    (policy_step.CommonFields.LOG_PROBABILITY,
     InfoFields.PREDICTED_REWARDS_MEAN,
     InfoFields.PREDICTED_REWARDS_SAMPLED))
# Set default empty tuple for all fields.
PolicyInfo.__new__.__defaults__ = ((),) * len(PolicyInfo._fields)


@common.function
def masked_argmax(input_tensor, mask, output_type=tf.int32):
  """Computes the argmax where the allowed elements are given by a mask.

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
  tf.compat.v1.assert_equal(
      tf.reduce_max(mask, axis=1), tf.constant(1, dtype=mask.dtype))
  modified_input = tf.compat.v2.where(
      tf.cast(mask, tf.bool), input_tensor, neg_inf)
  return tf.argmax(modified_input, axis=-1, output_type=output_type)
