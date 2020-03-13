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

# Lint as: python3
"""Policy wrapper that updates `policy_info` from wrapped policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Callable, Text, Dict, Union, Sequence
import tensorflow.compat.v2 as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step


class PolicyInfoUpdaterWrapper(tf_policy.Base):
  """Returns samples with updated `policy_info` (a dictionary).
  """

  def __init__(self,
               policy: tf_policy.Base,
               info_spec: Dict[Text, Union[tf.TensorSpec,
                                           Sequence[tf.TensorSpec]]],
               updater_fn: Callable[[policy_step.PolicyStep],
                                    Dict[Text, Union[tf.Tensor,
                                                     Sequence[tf.Tensor]]]],
               name: Text = None):
    """Builds a TFPolicy wrapping the given policy.

    PolicyInfoUpdaterWrapper class updates `policy_info` using a user-defined
    updater function. The main use case of this policy wrapper is to annotate
    `policy_info` with some auxiliary information. For example, appending
    an identifier to specify which model is used for current rollout.

    Args:
      policy: A policy implementing the tf_policy.Base interface.
      info_spec: User-defined `info_spec` which specifies the policy info after
        applying the updater function.
      updater_fn: An updater function that updates the `policy_info`. This is a
        callable that receives a `PolicyStep` and will return a dictionary of a
        tf.Tensor or sequence of tf.Tensor`s.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    super(PolicyInfoUpdaterWrapper, self).__init__(
        time_step_spec=policy.time_step_spec,
        action_spec=policy.action_spec,
        policy_state_spec=policy.policy_state_spec,
        info_spec=info_spec,
        emit_log_probability=policy.emit_log_probability,
        name=name)
    self._wrapped_policy = policy
    self._info_spec = info_spec
    self._updater_fn = updater_fn

  def _variables(self):
    return self._wrapped_policy.variables()

  # Helper function to verify the compatibility between `current_info` and
  # `_info_spec`.
  def _check_value(self, tensor: tf.Tensor, tensorspec: tf.TensorSpec):
    if not tf.TensorShape(tf.squeeze(tensor.get_shape())).is_compatible_with(
        tensorspec.shape):
      raise ValueError(
          'Tensor {} is not compatible with specification {}.'.format(
              tensor, tensorspec))

  def apply_value_network(self, *args, **kwargs):
    return self._wrapped_policy.apply_value_network(*args, **kwargs)

  def _update_info(self, step):
    if not isinstance(step.info, dict):
      raise ValueError('`step.info` must be a dictionary.')
    current_info = step.info
    current_info.update(self._updater_fn(step))
    return policy_step.PolicyStep(step.action, step.state, current_info)

  def _action(self, time_step, policy_state, seed):
    action_step = self._wrapped_policy.action(time_step, policy_state, seed)
    return self._update_info(action_step)

  def _distribution(self, time_step, policy_state):
    distribution_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    return self._update_info(distribution_step)
