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

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import tensor_spec

GLOBAL_FEATURE_KEY = 'global'
PER_ARM_FEATURE_KEY = 'per_arm'


def create_per_arm_observation_spec(global_dim, per_arm_dim, num_actions):
  global_obs_spec = tensor_spec.TensorSpec((global_dim,), tf.float32)
  arm_obs_spec = tensor_spec.TensorSpec((num_actions, per_arm_dim), tf.float32)
  return {
      GLOBAL_FEATURE_KEY: global_obs_spec,
      PER_ARM_FEATURE_KEY: arm_obs_spec
  }
