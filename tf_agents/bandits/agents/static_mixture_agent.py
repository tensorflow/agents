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

"""An agent that mixes a list of agents with a constant mixture distribution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tf_agents.bandits.agents import mixture_agent


@gin.configurable
class StaticMixtureAgent(mixture_agent.MixtureAgent):
  """An agent that mixes a set of agents with a given static mixture.

  For every data sample, the agent updates the sub-agent that was used to make
  the action choice in that sample. For this update to happen, the mixture agent
  needs to have the information on which sub-agent is "responsible" for the
  action. This information is in a policy info field `mixture_agent_id`.

  Note that this agent makes use of `tf.dynamic_partition`, and thus it is not
  compatible with XLA.
  """

  def _update_mixture_distribution(self, experience):
    pass
