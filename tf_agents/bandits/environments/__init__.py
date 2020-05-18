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

"""Module importing all environments."""

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import bandit_tf_environment
from tf_agents.bandits.environments import bernoulli_action_mask_tf_environment
from tf_agents.bandits.environments import bernoulli_py_environment
from tf_agents.bandits.environments import classification_environment
from tf_agents.bandits.environments import drifting_linear_environment
from tf_agents.bandits.environments import mushroom_environment_utilities
from tf_agents.bandits.environments import non_stationary_stochastic_environment
from tf_agents.bandits.environments import piecewise_bernoulli_py_environment
from tf_agents.bandits.environments import piecewise_stochastic_environment
from tf_agents.bandits.environments import random_bandit_environment
from tf_agents.bandits.environments import stationary_stochastic_per_arm_py_environment
from tf_agents.bandits.environments import stationary_stochastic_py_environment
from tf_agents.bandits.environments import stationary_stochastic_structured_py_environment
from tf_agents.bandits.environments import wheel_py_environment
