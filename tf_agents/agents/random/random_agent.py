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

"""A random agent.

An agent implementing a random policy without training. Useful as a baseline
when comparing to other agents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow.compat.v2 as tf
from tf_agents.agents.random import fixed_policy_agent
import tf_agents.policies.random_tf_policy as random_tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


@gin.configurable
class RandomAgent(fixed_policy_agent.FixedPolicyAgent):
  """An agent with a random policy and no learning."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               debug_summaries: bool = False,
               summarize_grads_and_vars: bool = False,
               train_step_counter: Optional[tf.Variable] = None,
               num_outer_dims: int = 1,
               name: Optional[Text] = None):
    """Creates a random agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      num_outer_dims: same as base class.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)

    policy_class = random_tf_policy.RandomTFPolicy

    super(RandomAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy_class=policy_class,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        num_outer_dims=num_outer_dims)
