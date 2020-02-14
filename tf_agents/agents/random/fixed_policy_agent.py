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

"""An agent with a fixed policy.

An agent following one specific policy, without training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import gin
import tensorflow.compat.v2 as tf
import tf_agents.agents.tf_agent as tf_agent

FLAGS = flags.FLAGS


@gin.configurable
class FixedPolicyAgent(tf_agent.TFAgent):
  """An agent with a fixed policy and no learning."""

  def __init__(self,
               time_step_spec,
               action_spec,
               policy_class,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               num_outer_dims=1,
               name=None):
    """Creates a fixed-policy agent with no-op for training.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      policy_class: a tf_policy.Base or py_policy.Base class to use as a policy
      debug_summaries: A bool to gather debug summaries. Used to initialize the
        base class
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step. Used to initialize the
        base class
      num_outer_dims: Used to initialize the base class
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name. Used to initialize the
        base class.
    """
    tf.Module.__init__(self, name=name)

    policy = policy_class(time_step_spec=time_step_spec,
                          action_spec=action_spec)

    collect_policy = policy

    super(FixedPolicyAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        num_outer_dims=num_outer_dims)

  def _initialize(self):
    pass

  def _train(self, experience, weights):
    """Do nothing. Arguments are ignored and loss is always 0."""
    del experience  # Unused
    del weights  # Unused

    # Incrementing the step counter.
    self.train_step_counter.assign_add(1)

    # Returning 0 loss.
    return tf_agent.LossInfo(0.0, None)
