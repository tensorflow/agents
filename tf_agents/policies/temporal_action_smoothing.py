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

"""A TFPolicy wrapper that applies exponential moving averaging to actions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step


class TemporalActionSmoothing(tf_policy.TFPolicy):
  """A wrapper that applies exponential moving averaging to action outputs."""

  def __init__(self,
               policy: tf_policy.TFPolicy,
               smoothing_coefficient: float,
               name: Optional[Text] = None):
    """Adds TemporalActionSmoothing to the given policy.

    smoothed_action = previous_action * smoothing_coefficient +
                      action * (1.0 - smoothing_coefficient))

    Args:
      policy: A policy implementing the tf_policy.TFPolicy interface.
      smoothing_coefficient: Coefficient used for smoothing actions.
      name: The name of this policy. Defaults to the class name.
    """
    policy_state_spec = (policy.policy_state_spec, policy.action_spec)
    super(TemporalActionSmoothing, self).__init__(
        policy.time_step_spec, policy.action_spec, policy_state_spec, name=name)
    self._wrapped_policy = policy
    self._smoothing_coefficient = smoothing_coefficient

  def _get_initial_state(self, batch_size):
    """Creates zero state tuple with wrapped initial state and smoothing vars.

    Args:
      batch_size: The batch shape.

    Returns:
      A tuple of (wrapped_policy_initial_state, initial_smoothing_state)
    """
    wrapped_initial_state = self._wrapped_policy.get_initial_state(batch_size)
    initial_smoothing_state = super(TemporalActionSmoothing,
                                    self)._get_initial_state(batch_size)[1]
    return (wrapped_initial_state, initial_smoothing_state)

  def _variables(self):
    return self._wrapped_policy.variables()

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        '`distribution` not implemented for TemporalActionSmoothingWrapper.')

  def _action(self, time_step, policy_state, seed):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    # Get action from the wrapped policy.
    wrapped_policy_state, moving_average = policy_state
    wrapped_policy_step = self._wrapped_policy.action(time_step,
                                                      wrapped_policy_state,
                                                      seed)

    # Compute smoothed action & updated action tensor.
    def _smooth_action_tensor(smoothing_state_tensor, action_tensor):
      return (smoothing_state_tensor * self._smoothing_coefficient +
              action_tensor * (1.0 - self._smoothing_coefficient))

    smoothed_action = tf.nest.map_structure(_smooth_action_tensor,
                                            moving_average,
                                            wrapped_policy_step.action)

    # Package results in PolicyStep.
    return policy_step.PolicyStep(smoothed_action,
                                  (wrapped_policy_step.state, smoothed_action),
                                  wrapped_policy_step.info)
