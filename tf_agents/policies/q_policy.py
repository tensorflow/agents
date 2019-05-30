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

"""Simple Policy for DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step


@gin.configurable
class QPolicy(tf_policy.Base):
  """Class to build Q-Policies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               q_network=None,
               emit_log_probability=False,
               name=None):
    """Builds a Q-Policy given a q_network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: An instance of a `tf_agents.network.Network`,
        callable via `network(observation, step_type) -> (output, final_state)`.
      emit_log_probability: Whether to emit log-probs in info of `PolicyStep`.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec`.
    """
    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise NotImplementedError(
          'action_spec can only contain a single BoundedTensorSpec.')
    # We need to maintain the flat action spec for dtype, shape and range.
    self._flat_action_spec = flat_action_spec[0]
    self._q_network = q_network
    super(QPolicy, self).__init__(
        time_step_spec,
        action_spec,
        policy_state_spec=q_network.state_spec,
        clip=False,
        emit_log_probability=emit_log_probability,
        name=name)

  def _variables(self):
    return self._q_network.variables

  def _distribution(self, time_step, policy_state):
    # In DQN, we always either take a uniformly random action, or the action
    # with the highest Q-value. However, to support more complicated policies,
    # we expose all Q-values as a categorical distribution with Q-values as
    # logits, and apply the GreedyPolicy wrapper in dqn_agent.py to select the
    # action with the highest Q-value.
    q_values, policy_state = self._q_network(
        time_step.observation, time_step.step_type, policy_state)

    # TODO(b/122314058): Validate and enforce that sampling distributions
    # created with the q_network logits generate the right action shapes. This
    # is curretly patching the problem.

    # If the action spec says each action should be shaped (1,), add another
    # dimension so the final shape is (B, 1, A), where A is the number of
    # actions. This will make Categorical emit events shaped (B, 1) rather than
    # (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
    if self._flat_action_spec.shape.ndims == 1:
      q_values = tf.expand_dims(q_values, -2)

    # TODO(kbanoop): Handle distributions over nests.
    distribution = shifted_categorical.ShiftedCategorical(
        logits=q_values,
        dtype=self._flat_action_spec.dtype,
        shift=self._flat_action_spec.minimum)
    distribution = tf.nest.pack_sequence_as(self._action_spec, [distribution])
    return policy_step.PolicyStep(distribution, policy_state)
