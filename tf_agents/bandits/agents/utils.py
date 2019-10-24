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

"""Common utility code and linear algebra functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.specs import tensor_spec


def sum_reward_weighted_observations(r, x):
  """Calculates an update used by some Bandit algorithms.

  Given an observation `x` and corresponding reward `r`, the weigthed
  observations vector (denoted `b` here) should be updated as `b = b + r * x`.
  This function calculates the sum of weighted rewards for batched
  observations `x`.

  Args:
    r: a `Tensor` of shape [`batch_size`]. This is the rewards of the batched
      observations.
    x: a `Tensor` of shape [`batch_size`, `context_dim`]. This is the matrix
      with the (batched) observations.

  Returns:
    The update that needs to be added to `b`. Has the same shape as `b`.
    If the observation matrix `x` is empty, a zero vector is returned.
  """
  batch_size = tf.shape(x)[0]

  return tf.reduce_sum(tf.reshape(r, [batch_size, 1]) * x, axis=0)


def get_num_actions_from_tensor_spec(action_spec):
  """Validates `action_spec` and returns number of actions.

  `action_spec` must specify a scalar int32 or int64 with minimum zero.

  Args:
    action_spec: a `TensorSpec`.

  Returns:
    The number of actions described by `action_spec`.

  Raises:
    ValueError: if `action_spec` is not an bounded scalar int32 or int64 spec
      with minimum 0.
  """
  if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
    raise ValueError('Action spec must be a `BoundedTensorSpec`; '
                     'got {}'.format(type(action_spec)))
  if action_spec.shape.rank != 0:
    raise ValueError('Action spec must be a scalar; '
                     'got shape{}'.format(action_spec.shape))
  if action_spec.dtype not in (tf.int32, tf.int64):
    raise ValueError('Action spec must be have dtype int32 or int64; '
                     'got {}'.format(action_spec.dtype))
  if action_spec.minimum != 0:
    raise ValueError('Action spec must have minimum 0; '
                     'got {}'.format(action_spec.minimum))
  return action_spec.maximum + 1
