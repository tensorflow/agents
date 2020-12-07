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

# TODO(b/155801943): Bring parallel_py_environment here once we're py3-only.
from tf_agents.environments import batched_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import random_py_environment
from tf_agents.environments import random_tf_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import trajectory_replay
from tf_agents.environments import utils
from tf_agents.environments import wrappers
