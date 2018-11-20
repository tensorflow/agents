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

# TODO(b/115993353): Reduce code duplication with q_policy, e.g. by having both
# inherit from a common ancestor.
"""Simple Q-Policy for Implicit Quantile DQN that takes an augmented Q network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy

import gin.tf
from tensorflow.python.ops import template  # TF internal

nest = tf.contrib.framework.nest


@gin.configurable()
class IQPolicy(tf_policy.Base):
  """Class to build Implicit Quantile Q-Policies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               policy_state_spec=(),
               q_network=None,
               temperature=1.0,
               template_name='iq_policy'):
    """Builds a Q-Policy given an augmented q_network Template or function.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      policy_state_spec: A nest of TensorSpec representing the
        policy state.
      q_network: A Template from tf.make_template or a function. When passing
        a Template the variables will be reused, passing a function it will
        create a new template with a new set of variables.
        The q_network should return a named tuple containing a q_values field.
      temperature: temperature for sampling, when close to 0.0 is arg_max.
      template_name: Name to use for the new template. Ignored if q_network is
        already a template.
    Raises:
      ValueError: If action_spec contains more than one BoundedTensorSpec.
    """
    super(IQPolicy, self).__init__(time_step_spec, action_spec,
                                   policy_state_spec)
    flat_action_spec = nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError(
          'action_spec can only contain a single BoundedTensorSpec.')
    # We need to extract the dtype shape and shape from the spec while
    # maintaining the nest structure of the spec.
    self._action_dtype = flat_action_spec[0].dtype
    self._action_shape = flat_action_spec[0].shape
    self._temperature = tf.convert_to_tensor(temperature, dtype=tf.float32)
    if isinstance(q_network, template.Template):
      self._q_network = q_network
    elif callable(q_network):
      self._q_network = tf.make_template(
          template_name, q_network, create_scope_now_=True)
    else:
      raise ValueError(
          'q_network must be either a template.Template or a callable.')

  def _variables(self):
    return self._q_network.global_variables

  @gin.configurable(module='IQPolicy')
  def _action(self, time_step, policy_state, seed):
    q_values = self._q_network(time_step).q_values
    q_values.shape.assert_has_rank(2)
    # TODO(kbanoop): Add a test for temperature
    logits = q_values / self._temperature
    actions = tf.multinomial(logits, num_samples=1, seed=seed)
    actions = tf.reshape(actions, [
        -1,
    ] + self._action_shape.as_list())
    actions = tf.cast(actions, self._action_dtype, name='action')
    actions = nest.pack_sequence_as(self._action_spec, [actions])
    return policy_step.PolicyStep(actions, policy_state)

  def _distribution(self, time_step, policy_state):
    q_values = self._q_network(time_step).q_values
    # TODO(kbanoop): Handle distributions over nests.
    distribution_ = tfp.distributions.Categorical(
        logits=q_values, dtype=self._action_dtype)
    distribution_ = nest.pack_sequence_as(self._action_spec, [distribution_])
    return policy_step.PolicyStep(distribution_, policy_state)
