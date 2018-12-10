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

nest = tf.contrib.framework.nest


@gin.configurable
class QPolicy(tf_policy.Base):
  """Class to build Q-Policies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               q_network=None,
               temperature=1.0):
    """Builds a Q-Policy given a q_network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: An instance of a `tf_agents.network.Network`,
        callable via `network(observation, step_type) -> (output, final_state)`.
      temperature: temperature for sampling when `action` is called.
        This parameter applies when the action spec is an integer.

        If `temperature` is close to 0.0 this is equivalent to calling
        `tf.argmax` on the output of the network.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec`.
    """
    flat_action_spec = nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise NotImplementedError(
          'action_spec can only contain a single BoundedTensorSpec.')
    # We need to extract the dtype shape and shape from the spec while
    # maintaining the nest structure of the spec.
    self._action_dtype = flat_action_spec[0].dtype
    self._action_shape = flat_action_spec[0].shape
    self._temperature = tf.convert_to_tensor(
        temperature, dtype=q_network.dtype)
    self._q_network = q_network
    super(QPolicy, self).__init__(
        time_step_spec, action_spec,
        policy_state_spec=q_network.state_spec)

  def _variables(self):
    return self._q_network.variables

  def _action(self, time_step, policy_state, seed):
    logits, policy_state = self._q_network(
        time_step.observation, time_step.step_type, policy_state)
    logits.shape.assert_has_rank(2)
    # TODO(kbanoop): Add a test for temperature
    scaled_logits = logits / self._temperature
    actions = tf.multinomial(scaled_logits, num_samples=1, seed=seed)
    actions = tf.reshape(actions, [-1] + self._action_shape.as_list())
    actions = tf.cast(actions, self._action_dtype, name='action')
    actions = nest.pack_sequence_as(self._action_spec, [actions])
    return policy_step.PolicyStep(actions, policy_state)

  def _distribution(self, time_step, policy_state):
    logits, policy_state = self._q_network(
        time_step.observation, time_step.step_type, policy_state)
    # TODO(kbanoop): Handle distributions over nests.
    distribution_ = tfp.distributions.Categorical(
        logits=logits, dtype=self._action_dtype)
    distribution_ = nest.pack_sequence_as(self._action_spec, [distribution_])
    return policy_step.PolicyStep(distribution_, policy_state)
