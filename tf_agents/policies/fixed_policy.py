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

"""A policy which always returns a fixed action.

Mainly used for unit tests.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class FixedPolicy(tf_policy.TFPolicy):
  """A policy which always returns a fixed action."""

  def __init__(self,
               actions: types.NestedTensor,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               emit_log_probability: bool = True,
               policy_info: types.NestedTensorSpec = (),
               info_spec: types.NestedTensorSpec = (),
               name: Optional[Text] = None):
    """A policy which always returns a fixed action.

    Args:
      actions: A Tensor, or a nested dict, list or tuple of Tensors
        corresponding to `action_spec()`.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      emit_log_probability: Emit log-probabilities of actions, if supported. If
        True, policy_step.info will have CommonFields.LOG_PROBABILITY set.
        Please consult utility methods provided in policy_step for setting and
        retrieving these. When working with custom policies, either provide a
        dictionary info_spec or a namedtuple with the field 'log_probability'.
      policy_info: A policy info to be returned in PolicyStep.
      info_spec: A policy info spec.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    super(FixedPolicy, self).__init__(time_step_spec, action_spec, clip=False,
                                      info_spec=info_spec,
                                      name=name,
                                      emit_log_probability=emit_log_probability)
    nest_utils.assert_same_structure(self._action_spec, actions)

    def convert(action, spec):
      return tf.convert_to_tensor(value=action, dtype=spec.dtype)

    self._action_value = tf.nest.map_structure(convert, actions,
                                               self._action_spec)
    if self._emit_log_probability:
      log_probability = tf.nest.map_structure(
          lambda t: tf.constant(0.0, tf.float32), self._action_spec)
      self._policy_info = policy_step.set_log_probability(policy_info,
                                                          log_probability)  # pytype: disable=wrong-arg-types
    else:
      self._policy_info = policy_info

  def _variables(self):
    return []

  def _get_policy_info_and_action(self, time_step):
    outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)

    if self._emit_log_probability:
      log_probability = tf.nest.map_structure(
          lambda _: tf.zeros(outer_shape, tf.float32), self._action_spec)
      policy_info = policy_step.set_log_probability(
          self._policy_info, log_probability=log_probability)
    else:
      policy_info = self._policy_info
    action = tf.nest.map_structure(lambda t: common.replicate(t, outer_shape),
                                   self._action_value)
    return policy_info, action

  def _action(self, time_step, policy_state, seed):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    del seed
    policy_info, action = self._get_policy_info_and_action(time_step)
    return policy_step.PolicyStep(action, policy_state, policy_info)

  def _distribution(self, time_step, policy_state):
    policy_info, action = self._get_policy_info_and_action(time_step)

    def dist_fn(action):
      """Return a categorical distribution with all density on fixed action."""
      return tfp.distributions.Deterministic(loc=action)
    return policy_step.PolicyStep(
        tf.nest.map_structure(dist_fn, action), policy_state, policy_info)
