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
from tf_agents.agents import behavioral_cloning
from tf_agents.agents import categorical_dqn
from tf_agents.agents import cql
from tf_agents.agents import data_converter
from tf_agents.agents import ddpg
from tf_agents.agents import dqn
from tf_agents.agents import ppo
from tf_agents.agents import reinforce
from tf_agents.agents import sac
from tf_agents.agents import td3
from tf_agents.agents import tf_agent

from tf_agents.agents.behavioral_cloning.behavioral_cloning_agent import BehavioralCloningAgent
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import CategoricalDqnAgent
from tf_agents.agents.cql.cql_sac_agent import CqlSacAgent
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.agents.ppo.ppo_kl_penalty_agent import PPOKLPenaltyAgent
from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent
from tf_agents.agents.sac.sac_agent import SacAgent
from tf_agents.agents.td3.td3_agent import Td3Agent
from tf_agents.agents.tf_agent import TFAgent
