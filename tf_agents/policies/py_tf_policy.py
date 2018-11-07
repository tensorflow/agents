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

"""Converts TensorFlow Policies into Python Policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.policies import policy_step
from tf_agents.policies import py_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import session_utils

nest = tf.contrib.framework.nest


class PyTFPolicy(py_policy.Base, session_utils.SessionUser):
  """Exposes a Python policy as wrapper over a TF Policy."""

  # TODO(damienv): currently, the initial policy state must be batched
  # if batch_size is given. Without losing too much generality, the initial
  # policy state could be the same for every element in the batch.
  # In that case, the initial policy state could be given with no batch
  # dimension.
  def __init__(self, policy, batch_size=None, seed=None):
    """Initializes a new `PyTFPolicy`.

    Args:
      policy: A TF Policy implementing `tf_policy.Base`.
      batch_size: The batch size of time_steps and actions.
      seed: Seed to use if policy performs random actions (optional).
    """
    if not isinstance(policy, tf_policy.Base):
      tf.logging.warning('Policy should implement tf_policy.Base')

    self._tf_policy = policy
    self.session = None

    self._time_step_spec = tensor_spec.to_nest_array_spec(
        self._tf_policy.time_step_spec())
    self._action_spec = tensor_spec.to_nest_array_spec(
        self._tf_policy.action_spec())
    self._policy_state_spec = tensor_spec.to_nest_array_spec(
        self._tf_policy.policy_state_spec())

    self._batch_size = batch_size
    self._seed = seed
    self._batched = batch_size is not None
    self._set_up_feeds_and_fetches()

  def _set_up_feeds_and_fetches(self):
    outer_dims = [self._batch_size] if self._batched else [1]
    self._time_step = tensor_spec.to_nest_placeholder(
        self._tf_policy.time_step_spec(), outer_dims=outer_dims)
    self._tf_initial_state = self._tf_policy.get_initial_state(
        batch_size=self._batch_size or 1)

    self._policy_state = nest.map_structure(
        lambda ps: tf.placeholder(  # pylint: disable=g-long-lambda
            ps.dtype, ps.shape, name='policy_state'),
        self._tf_initial_state)
    self._action_step = self._tf_policy.action(
        self._time_step, self._policy_state, seed=self._seed)

  def _get_initial_state(self, batch_size):
    if batch_size != self._batch_size:
      raise ValueError(
          '`batch_size` argument is different from the batch size provided to '
          'the constructor. Expected {}, but saw {}.'.format(
              self._batch_size, batch_size))
    return self.session.run(self._tf_initial_state)

  def _action(self, time_step, policy_state):
    if not self._batched:
      # Since policy_state is given in a batched form from the policy and we
      # simply have to send it back we do not need to worry about it. Only
      # update time_step.
      time_step = nest_utils.batch_nested_array(time_step)

    nest.assert_same_structure(self._time_step, time_step)
    feed_dict = {self._time_step: time_step}
    if policy_state is not None:
      # Flatten policy_state to handle specs that are not hashable due to lists.
      for state_ph, state in zip(
          nest.flatten(self._policy_state), nest.flatten(policy_state)):
        feed_dict[state_ph] = state

    action_step = self.session.run(self._action_step, feed_dict)
    action, state, info = action_step

    if not self._batched:
      action, info = nest_utils.unbatch_nested_array([action, info])

    return policy_step.PolicyStep(action, state, info)
