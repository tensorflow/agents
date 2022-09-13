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

"""Module importing all agents."""

from tf_agents.bandits.agents import bernoulli_thompson_sampling_agent
from tf_agents.bandits.agents import dropout_thompson_sampling_agent
from tf_agents.bandits.agents import examples
from tf_agents.bandits.agents import exp3_agent
from tf_agents.bandits.agents import exp3_mixture_agent
from tf_agents.bandits.agents import greedy_multi_objective_neural_agent
from tf_agents.bandits.agents import greedy_reward_prediction_agent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_bandit_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent
from tf_agents.bandits.agents import mixture_agent
from tf_agents.bandits.agents import neural_boltzmann_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.agents import neural_falcon_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import ranking_agent
from tf_agents.bandits.agents import static_mixture_agent
from tf_agents.bandits.agents import utils
