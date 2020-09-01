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

"""A policy class that chooses from a set of policies to get the actions from.

This mixture policy takes a list of policies and will randomly choose one of
them for every observation. The distribution is defined by the
`mixture_distribution`.
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Sequence, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import nest_utils

tfd = tfp.distributions


MIXTURE_AGENT_ID = 'mixture_agent_id'
SUBPOLICY_INFO = 'subpolicy_info'


@gin.configurable
class MixturePolicy(tf_policy.TFPolicy):
  """A policy that chooses from a set of policies to decide the action."""

  def __init__(self,
               mixture_distribution: types.Distribution,
               policies: Sequence[tf_policy.TFPolicy],
               name: Optional[Text] = None):
    """Initializes an instance of `MixturePolicy`.

    Args:
      mixture_distribution: A `tfd.Categorical` distribution on the domain `[0,
        len(policies) -1]`. This distribution is used by the mixture policy to
        choose which policy to listen to.
      policies: List of TF Policies. These are the policies that the mixture
        policy chooses from in every time step.
      name: The name of this instance of `MixturePolicy`.
    """
    self._policies = policies
    if not isinstance(mixture_distribution, tfd.Categorical):
      raise TypeError(
          'mixture distribution must be an instance of `tfd.Categorical`.')
    self._mixture_distribution = mixture_distribution
    action_spec = policies[0].action_spec
    time_step_spec = policies[0].time_step_spec
    for policy in policies[1:]:
      assert action_spec == policy.action_spec, 'Inconsistent action specs.'
      assert time_step_spec == policy.time_step_spec, ('Inconsistent time step '
                                                       'specs.')
      assert policies[0].info_spec == policy.info_spec, ('Inconsistent info '
                                                         'specs.')

    info_spec = {
        MIXTURE_AGENT_ID:
            tensor_spec.BoundedTensorSpec(
                shape=(), dtype=tf.int32, minimum=0, maximum=len(policies) - 1),
        SUBPOLICY_INFO:
            policies[0].info_spec
    }

    super(MixturePolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        name=name)

  def _variables(self):
    variables = sum([p.variables() for p in self._policies], [])
    variables.extend(self._mixture_distribution.variables)
    return variables

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        '_distribution is not implemented for this policy.')

  def _action(self, time_step, policy_state, seed=None):
    first_obs = tf.nest.flatten(time_step.observation)[0]
    batch_size = tf.compat.dimension_value(
        first_obs.shape[0]) or tf.shape(first_obs)[0]
    policy_choice = self._mixture_distribution.sample(batch_size)
    policy_steps = [
        policy.action(time_step, policy_state) for policy in self._policies
    ]
    policy_actions = nest_utils.stack_nested_tensors(
        [step.action for step in policy_steps], axis=-1)
    policy_infos = nest_utils.stack_nested_tensors(
        [step.info for step in policy_steps], axis=-1)

    expanded_choice = tf.expand_dims(policy_choice, axis=-1)
    mixture_action = tf.nest.map_structure(
        lambda t: tf.gather(t, policy_choice, batch_dims=1), policy_actions)

    expanded_mixture_info = tf.nest.map_structure(
        lambda t: tf.gather(t, expanded_choice, batch_dims=1), policy_infos)
    mixture_info = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=1),
                                         expanded_mixture_info)
    return policy_step.PolicyStep(mixture_action, policy_state, {
        MIXTURE_AGENT_ID: policy_choice,
        SUBPOLICY_INFO: mixture_info
    })
