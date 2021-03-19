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

"""Bandit related tensor spec utilities."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import copy
from typing import Optional, Tuple

import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

GLOBAL_FEATURE_KEY = 'global'
PER_ARM_FEATURE_KEY = 'per_arm'
NUM_ACTIONS_FEATURE_KEY = 'num_actions'

# For constrained optimization, the reward spec is expected to be a dictionary
# with the following keys that split the reward spec and the constraints spec.
REWARD_SPEC_KEY = 'reward'
CONSTRAINTS_SPEC_KEY = 'constraint'


def create_per_arm_observation_spec(
    global_dim: int,
    per_arm_dim: int,
    max_num_actions: Optional[int] = None,
    add_num_actions_feature: bool = False) -> types.NestedTensorSpec:
  """Creates an observation spec with per-arm features and possibly action mask.

  Args:
    global_dim: (int) The global feature dimension.
    per_arm_dim: (int) The per-arm feature dimension.
    max_num_actions: If specified (int), this is the maximum number of actions
      in any sample, and the num_actions dimension of the per-arm features
      will be set to this number. The actual number of actions for a given
      sample can be lower than this parameter: it can be specified via the
      NUM_ACTIONS_FEATURE_KEY, or an action mask.
    add_num_actions_feature: (bool) whether to use the `num_actions` feature key
      to encode the number of actions per sample.

  Returns:
    A nested structure of observation spec.
  """
  global_obs_spec = tensor_spec.TensorSpec((global_dim,), tf.float32)
  arm_obs_spec = tensor_spec.TensorSpec((max_num_actions, per_arm_dim),
                                        tf.float32)
  observation_spec = {GLOBAL_FEATURE_KEY: global_obs_spec,
                      PER_ARM_FEATURE_KEY: arm_obs_spec}
  if add_num_actions_feature:
    observation_spec.update({
        NUM_ACTIONS_FEATURE_KEY:
            tensor_spec.BoundedTensorSpec((),
                                          minimum=1,
                                          maximum=max_num_actions,
                                          dtype=tf.int32)
    })
  return observation_spec


def get_context_dims_from_spec(
    context_spec: types.NestedTensorSpec,
    accepts_per_arm_features: bool) -> Tuple[int, int]:
  """Returns the global and per-arm context dimensions.

  If the policy accepts per-arm features, this function returns the tuple of
  the global and per-arm context dimension. Otherwise, it returns the (global)
  context dim and zero.

  Args:
    context_spec: A nest of tensor specs, containing the observation spec.
    accepts_per_arm_features: (bool) Whether the context_spec is for a policy
      that accepts per-arm features.

  Returns: A 2-tuple of ints, the global and per-arm context dimension. If the
    policy does not accept per-arm features, the per-arm context dim is 0.
  """
  if accepts_per_arm_features:
    global_context_dim = context_spec[GLOBAL_FEATURE_KEY].shape.as_list()[0]
    arm_context_dim = context_spec[PER_ARM_FEATURE_KEY].shape.as_list()[1]
  else:
    assert hasattr(context_spec, 'shape')
    spec_shape = context_spec.shape.as_list()
    global_context_dim = spec_shape[0] if spec_shape else 1
    arm_context_dim = 0
  return global_context_dim, arm_context_dim


def drop_arm_observation(trajectory: types.Trajectory) -> types.Trajectory:
  """Drops the per-arm observation from a given trajectory/trajectory spec."""
  transformed_trajectory = copy.deepcopy(trajectory)
  del transformed_trajectory.observation[PER_ARM_FEATURE_KEY]
  return transformed_trajectory
