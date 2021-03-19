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

"""Module importing all policies."""

from tf_agents.bandits.policies import categorical_policy
from tf_agents.bandits.policies import greedy_multi_objective_neural_policy
from tf_agents.bandits.policies import greedy_reward_prediction_policy
from tf_agents.bandits.policies import lin_ucb_policy
from tf_agents.bandits.policies import linalg
from tf_agents.bandits.policies import linear_thompson_sampling_policy
from tf_agents.bandits.policies import mixture_policy
from tf_agents.bandits.policies import neural_linucb_policy
from tf_agents.policies import utils as policy_utilities
