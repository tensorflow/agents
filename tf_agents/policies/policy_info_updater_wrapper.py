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

"""Policy wrapper that updates `policy_info` from wrapped policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, Text, Dict, Union, Sequence, Optional

import tensorflow.compat.v2 as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

# A callable that receives a `PolicyStep` and returns a dictionary of a
# tf.Tensor or a sequence of tf.Tensor`s used to update the policy_info.
UpdaterFnType = Callable[[policy_step.PolicyStep],
                         Dict[Text, Union[tf.Tensor, Sequence[tf.Tensor]]]]


class PolicyInfoUpdaterWrapper(tf_policy.TFPolicy):
  """Returns samples with updated `policy_info` (a dictionary).
  """

  def __init__(self,
               policy: tf_policy.TFPolicy,
               info_spec: types.NestedTensorSpec,
               updater_fn: UpdaterFnType,
               name: Optional[Text] = None):
    """Builds a TFPolicy wrapping the given policy.

    PolicyInfoUpdaterWrapper class updates `policy_info` using a user-defined
    updater function. The main use case of this policy wrapper is to annotate
    `policy_info` with some auxiliary information. For example, appending
    an identifier to specify which model is used for current rollout.

    Args:
      policy: A policy implementing the tf_policy.TFPolicy interface.
      info_spec: User-defined `info_spec` which specifies the policy info after
        applying the updater function.
      updater_fn: An updater function that updates the `policy_info`. This is a
        callable that receives a `PolicyStep` and will return a dictionary of a
        tf.Tensor or sequence of tf.Tensor`s.

        **NOTE** If `policy.distribution` is called, the `PolicyStep.action`
        object may contain a `tfp.distributions.Distribution` object instead
        of a `Tensor`.  The `updater_fn` must be able to handle both cases
        to be compatible with `PolicySaver`.
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
