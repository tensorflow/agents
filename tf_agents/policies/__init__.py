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

"""Policies Module."""

from tf_agents.policies import actor_policy
from tf_agents.policies import boltzmann_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import fixed_policy
from tf_agents.policies import gaussian_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import ou_noise_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import py_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import q_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import scripted_py_policy
from tf_agents.policies import tf_policy
from tf_agents.policies import tf_py_policy
from tf_agents.policies import utils

from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from tf_agents.policies.greedy_policy import GreedyPolicy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.policies.py_tf_eager_policy import SavedModelPyTFEagerPolicy
from tf_agents.policies.tf_policy import TFPolicy
