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

"""Exposes a python policy as an in-graph TensorFlow policy."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import py_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils


def map_tensor_spec_to_dtypes_list(t_spec):
  return [spec.dtype for spec in tf.nest.flatten(t_spec)]


class TFPyPolicy(tf_policy.TFPolicy):
  """Exposes a Python policy as an in-graph TensorFlow policy.

  # TODO(kbanoop): This class does not seem to handle batching/unbatching when
  # converting between TF and Py policies.
  """

  def __init__(self,
               policy: py_policy.PyPolicy,
               py_policy_is_batched: bool = False,
               name: Optional[Text] = None):
    """Initializes a new `TFPyPolicy` instance with an Pyton policy .

    Args:
      policy: Python policy implementing `py_policy.PyPolicy`.
      py_policy_is_batched: If False, time_steps will be unbatched before
        passing to py_policy.action(), and a batch dimension will be added to
        the returned action. This will only work with time_steps that have a
        batch dimension of 1. If True, the time_step (input) and action (output)
        are passed exactly as is from/to the py_policy.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      TypeError: if a non python policy is passed to constructor.
    """
    if not isinstance(policy, py_policy.PyPolicy):
      raise TypeError(
          'Input policy should implement py_policy.PyPolicy, but saw %s.' %
          type(policy).__name__)

    self._py_policy = policy
    self._py_policy_is_batched = py_policy_is_batched

    (time_step_spec, action_spec,
     policy_state_spec, info_spec) = tf.nest.map_structure(
         tensor_spec.from_spec,
         (policy.time_step_spec, policy.action_spec, policy.policy_state_spec,
          policy.info_spec))

    super(TFPyPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        clip=False,
        name=name,
        automatic_state_reset=False)

    # Output types of py_funcs.
    self._policy_state_dtypes = map_tensor_spec_to_dtypes_list(
        self.policy_state_spec)
    self._policy_step_dtypes = map_tensor_spec_to_dtypes_list(
        self.policy_step_spec)

  # Wrapped in common.function to avoid failures in eager mode. This happens
  # when the policy_state is empty and it gets dropped by tf.nest.flatten
  # # in the numpy_function
  @common.function
  def _get_initial_state(self, batch_size):
    """Invokes  python policy reset through numpy_function.

    Args:
      batch_size: Batch size for the get_initial_state tensor(s).

    Returns:
      A tuple of (policy_state, reset_op).
      policy_state: Tensor, or a nested dict, list or tuple of Tensors,
      representing the new policy state.
      reset_op: a list of Tensors representing the results of py_policy.reset().
    """

    def _get_initial_state_fn(*batch_size):
      return tf.nest.flatten(
          self._py_policy.get_initial_state(batch_size=batch_size))

    with tf.name_scope('get_initial_state'):
      flat_policy_state = tf.numpy_function(
          _get_initial_state_fn, [batch_size],
          self._policy_state_dtypes,
          name='get_initial_state_numpy_function')
      return tf.nest.pack_sequence_as(
          structure=self.policy_state_spec, flat_sequence=flat_policy_state)

  # Wrapped in common.function to avoid failures in eager mode. This happens
  # when empty fields in the policy_step get dropped by tf.nest.flatten
  # in the numpy_function.
  @common.function
  def _action(self, time_step, policy_state, seed):
    if seed is not None:
      raise NotImplementedError(
          'seed is not supported; but saw seed: {}'.format(seed))

    def _action_fn(*flattened_time_step_and_policy_state):
      packed_py_time_step, packed_py_policy_state = tf.nest.pack_sequence_as(
          structure=(self._py_policy.time_step_spec,
                     self._py_policy.policy_state_spec),
          flat_sequence=flattened_time_step_and_policy_state)
      py_action_step = self._py_policy.action(
          time_step=packed_py_time_step, policy_state=packed_py_policy_state)
      return tf.nest.flatten(py_action_step)

    with tf.name_scope('action'):
      if not self._py_policy_is_batched:
        time_step = nest_utils.unbatch_nested_tensors(time_step)
      flattened_input_tensors = tf.nest.flatten((time_step, policy_state))

      flat_action_step = tf.numpy_function(
          _action_fn,
          flattened_input_tensors,
          self._policy_step_dtypes,
          name='action_numpy_function')
      action_step = tf.nest.pack_sequence_as(
          structure=self.policy_step_spec, flat_sequence=flat_action_step)
      if not self._py_policy_is_batched:
        action_step = action_step._replace(
            action=nest_utils.batch_nested_tensors(action_step.action))
      return action_step

  def _variables(self):
    """Returns default [] representing a policy that has no variables."""
    return []

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError('%s does not support distribution yet.' %
                              self.__class__.__name__)
