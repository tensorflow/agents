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

"""Policy implementation that applies temperature to a distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.policies import tf_policy


class BoltzmannPolicy(tf_policy.Base):
  """Returns boltzmann samples of a given policy.

  The wrapped policy must expose a distribution parameterized by logits.
  """

  def __init__(self, policy, temperature=1.0, name=None):
    """Builds a BoltzmannPolicy wrapping the given policy.

    Args:
      policy: A policy implementing the tf_policy.Base interface, using
        a distribution parameterized by logits.
      temperature: temperature for sampling when `action` is called.
        This parameter applies when the action spec is discrete.

        If `temperature` is close to 0.0 this is equivalent to calling
        `tf.argmax` on the output of the network.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    super(BoltzmannPolicy, self).__init__(policy.time_step_spec(),
                                          policy.action_spec(),
                                          policy.policy_state_spec(),
                                          policy.info_spec(),
                                          name=name)
    if temperature == 1.0:
      self._temperature = None
    else:
      self._temperature = tf.convert_to_tensor(value=temperature)
    self._wrapped_policy = policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _distribution(self, time_step, policy_state):
    distribution_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    if self._temperature is None:
      return distribution_step

    # Change the action distribution to incorporate the temperature.
    def _apply_temperature(dist):
      logits = dist.logits / self._temperature
      return dist.copy(logits=logits)

    action_dist = tf.nest.map_structure(_apply_temperature,
                                        distribution_step.action)
    return distribution_step._replace(action=action_dist)
