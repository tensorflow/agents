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

"""Environments module."""

from tf_agents.environments import batched_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import random_py_environment
from tf_agents.environments import random_tf_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import trajectory_replay
from tf_agents.environments import utils
from tf_agents.environments import wrappers

# pylint: disable=g-import-not-at-top
try:
  from tf_agents.environments import gym_wrapper
  from tf_agents.environments import suite_gym
  from tf_agents.environments import suite_atari
  from tf_agents.environments import suite_dm_control
  from tf_agents.environments import suite_mujoco
  from tf_agents.environments import suite_pybullet
  from tf_agents.environments.gym_wrapper import GymWrapper
except (ImportError, ModuleNotFoundError):
  pass

from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.random_py_environment import RandomPyEnvironment
from tf_agents.environments.random_tf_environment import RandomTFEnvironment
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.trajectory_replay import TrajectoryReplay
from tf_agents.environments.utils import validate_py_environment
from tf_agents.environments.wrappers import ActionClipWrapper
from tf_agents.environments.wrappers import ActionDiscretizeWrapper
from tf_agents.environments.wrappers import ActionOffsetWrapper
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments.wrappers import FlattenObservationsWrapper
from tf_agents.environments.wrappers import GoalReplayEnvWrapper
from tf_agents.environments.wrappers import HistoryWrapper
from tf_agents.environments.wrappers import ObservationFilterWrapper
from tf_agents.environments.wrappers import OneHotActionWrapper
from tf_agents.environments.wrappers import PerformanceProfiler
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.environments.wrappers import RunStats
from tf_agents.environments.wrappers import TimeLimit
