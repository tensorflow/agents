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

"""Bandit related tensor spec utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import tensor_spec

GLOBAL_FEATURE_KEY = 'global'
PER_ARM_FEATURE_KEY = 'per_arm'

# For constrained optimization, the reward spec is expected to be a dictionary
# with the following keys that split the reward spec and the constraints spec.
REWARD_SPEC_KEY = 'reward'
CONSTRAINTS_SPEC_KEY = 'constraint'


def create_per_arm_observation_spec(global_dim,
                                    per_arm_dim,
                                    num_actions,
                                    apply_mask=False):
  """Creates an observation spec with per-arm features and possibly action mask."""
  global_obs_spec = tensor_spec.TensorSpec((global_dim,), tf.float32)
  arm_obs_spec = tensor_spec.TensorSpec((num_actions, per_arm_dim), tf.float32)
  obs_spec = {
      GLOBAL_FEATURE_KEY: global_obs_spec,
      PER_ARM_FEATURE_KEY: arm_obs_spec
  }
  if apply_mask:
    obs_spec = (obs_spec,
                tensor_spec.BoundedTensorSpec(
                    shape=(num_actions,),
                    minimum=0,
                    maximum=1,
                    dtype=tf.float32))
  return obs_spec


def get_context_dims_from_spec(context_spec, accepts_per_arm_features):
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
    spec_shape = context_spec.shape.as_list()
    global_context_dim = spec_shape[0] if spec_shape else 1
    arm_context_dim = 0
  return global_context_dim, arm_context_dim


def drop_arm_observation(trajectory,
                         observation_and_action_constraint_splitter=None):
  """Drops the per-arm observation from a given trajectory (or trajectory spec)."""
  transformed_trajectory = copy.deepcopy(trajectory)
  if observation_and_action_constraint_splitter is None:
    del transformed_trajectory.observation[PER_ARM_FEATURE_KEY]
  else:
    del observation_and_action_constraint_splitter(
        transformed_trajectory.observation)[0][PER_ARM_FEATURE_KEY]
  return transformed_trajectory
