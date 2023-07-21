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

"""Forwarding utils for backwards compatibility."""

from tf_agents.specs import bandit_spec_utils as _utils

GLOBAL_FEATURE_KEY = _utils.GLOBAL_FEATURE_KEY
PER_ARM_FEATURE_KEY = _utils.PER_ARM_FEATURE_KEY
NUM_ACTIONS_FEATURE_KEY = _utils.NUM_ACTIONS_FEATURE_KEY

REWARD_SPEC_KEY = _utils.REWARD_SPEC_KEY
CONSTRAINTS_SPEC_KEY = _utils.CONSTRAINTS_SPEC_KEY

create_per_arm_observation_spec = _utils.create_per_arm_observation_spec
get_context_dims_from_spec = _utils.get_context_dims_from_spec
drop_arm_observation = _utils.drop_arm_observation
