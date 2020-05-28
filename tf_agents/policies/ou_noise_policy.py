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

"""A policy that wraps a given policy and adds Ornstein Uhlenbeck (OU) noise."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import common

tfd = tfp.distributions


class OUNoisePolicy(tf_policy.TFPolicy):
  """Actor Policy with Ornstein Uhlenbeck (OU) exploration noise."""

  def __init__(self,
               wrapped_policy: tf_policy.TFPolicy,
               ou_stddev: types.Float = 1.0,
               ou_damping: types.Float = 1.0,
               clip: bool = True,
               name: Optional[Text] = None):
    """Builds an OUNoisePolicy wrapping wrapped_policy.

    Args:
      wrapped_policy: A policy to wrap and add OU noise to.
      ou_stddev:  stddev for the Ornstein-Uhlenbeck noise.
      ou_damping: damping factor for the Ornstein-Uhlenbeck noise.
      clip: Whether to clip actions to spec. Default True.
      name: The name of this policy.
    """

    def _validate_action_spec(action_spec):
      if not tensor_spec.is_continuous(action_spec):
        raise ValueError('OU Noise is applicable only to continuous actions.')

    tf.nest.map_structure(_validate_action_spec, wrapped_policy.action_spec)

    super(OUNoisePolicy, self).__init__(
        wrapped_policy.time_step_spec,
        wrapped_policy.action_spec,
        wrapped_policy.policy_state_spec,
        wrapped_policy.info_spec,
        clip=clip,
        name=name)
    self._ou_stddev = ou_stddev
    self._ou_damping = ou_damping
    self._ou_process = None
    self._wrapped_policy = wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _action(self, time_step, policy_state, seed):
    seed_stream = tfp.util.SeedStream(seed=seed, salt='ou_noise')

    def _create_ou_process(action_spec):
      return common.OUProcess(
          lambda: tf.zeros(action_spec.shape, dtype=action_spec.dtype),
          self._ou_damping,
          self._ou_stddev,
          seed=seed_stream())

    if self._ou_process is None:
      self._ou_process = tf.nest.map_structure(_create_ou_process,
                                               self._action_spec)

    action_step = self._wrapped_policy.action(time_step, policy_state,
                                              seed_stream())

    def _add_ou_noise(action, ou_process):
      return action + ou_process()

    actions = tf.nest.map_structure(_add_ou_noise, action_step.action,
                                    self._ou_process)
    return policy_step.PolicyStep(actions, action_step.state, action_step.info)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError('Distributions are not implemented yet.')
