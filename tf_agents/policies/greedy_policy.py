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

"""Policy implementation that generates greedy actions from another policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy


class GreedyPolicy(tf_policy.Base):
  """Returns greedy samples of a given policy."""

  def __init__(self, policy, name=None):
    """Builds a greedy TFPolicy wrapping the given policy.

    Args:
      policy: A policy implementing the tf_policy.Base interface.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    super(GreedyPolicy, self).__init__(policy.time_step_spec,
                                       policy.action_spec,
                                       policy.policy_state_spec,
                                       policy.info_spec,
                                       name=name)
    self._wrapped_policy = policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _distribution(self, time_step, policy_state):

    def dist_fn(dist):
      greedy_action = dist.mode()
      return tfp.distributions.Deterministic(loc=greedy_action)

    distribution_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    return policy_step.PolicyStep(
        tf.nest.map_structure(dist_fn, distribution_step.action),
        distribution_step.state, distribution_step.info)
