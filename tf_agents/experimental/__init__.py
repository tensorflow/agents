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

"""TF-Agents Experimental Modules.

These utilities, libraries, and tools have not been rigorously tested for
production use.  For example, experimental examples may not have associated
nightly regression tests.
"""
# Aliasing the already moved `tf_agent.train` module from its new location here
# for backward compatibility.
# TODO(b/175303833): Remove this when everyone uses the new dependencies.
from tf_agents import train
from tf_agents.experimental import distributed
