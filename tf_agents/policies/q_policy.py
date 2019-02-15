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

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy

import gin.tf


@gin.configurable
class QPolicy(tf_policy.Base):
  """Class to build Q-Policies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               q_network=None,
               name=None):
    """Builds a Q-Policy given a q_network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: An instance of a `tf_agents.network.Network`,
        callable via `network(observation, step_type) -> (output, final_state)`.
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
    # We need to extract the dtype shape and shape from the spec while
    # maintaining the nest structure of the spec.
    self._action_dtype = flat_action_spec[0].dtype
    self._action_shape = flat_action_spec[0].shape
    self._q_network = q_network
    super(QPolicy, self).__init__(
        time_step_spec, action_spec,
        policy_state_spec=q_network.state_spec,
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
    q_values.shape.assert_has_rank(2)

    # TODO(b/122314058): Validate and enforce that sampling distributions
    # created with the q_network logits generate the right action shapes. This
    # is curretly patching the problem.

    # If the action spec says each action should be shaped (1,), add another
    # dimension so the final shape is (B, 1, A), where A is the number of
    # actions. This will make Categorical emit events shaped (B, 1) rather than
    # (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
    if self._action_shape.ndims == 1:
      q_values = tf.expand_dims(q_values, -2)

    # TODO(kbanoop): Handle distributions over nests.
    distribution = tfp.distributions.Categorical(
        logits=q_values, dtype=self._action_dtype)
    distribution = tf.nest.pack_sequence_as(self._action_spec, [distribution])
    return policy_step.PolicyStep(distribution, policy_state)
