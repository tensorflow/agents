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

"""Converts tf_policies when working in eager mode to py_policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import py_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class PyTFEagerPolicyBase(py_policy.Base):
  """Base class for py_policy instances of TF policies in Eager mode.

  Handles adding and removing batch dimensions from the actions and time_steps.
  Note if you have a tf_policy you should directly use the PyTFEagerPolicy class
  instead of this Base.
  """

  def __init__(self, policy, time_step_spec, action_spec, policy_state_spec,
               info_spec, use_tf_function=False):
    self._policy = policy
    if use_tf_function:
      self._policy_action_fn = common.function(policy.action)
    else:
      self._policy_action_fn = policy.action
    super(PyTFEagerPolicyBase, self).__init__(time_step_spec, action_spec,
                                              policy_state_spec, info_spec)

  def _get_initial_state(self, batch_size):
    return self._policy.get_initial_state(batch_size=batch_size)

  def _action(self, time_step, policy_state):
    time_step = nest_utils.batch_nested_array(time_step)
    # Avoid passing numpy arrays to avoid retracing of the tf.function.
    time_step = tf.nest.map_structure(tf.convert_to_tensor, time_step)
    policy_step = self._policy_action_fn(time_step, policy_state)
    return policy_step._replace(
        action=nest_utils.unbatch_nested_array(policy_step.action.numpy()))


@gin.configurable
class PyTFEagerPolicy(PyTFEagerPolicyBase):
  """Exposes a numpy API for TF policies in Eager mode."""

  def __init__(self, policy):
    time_step_spec = tensor_spec.to_nest_array_spec(policy.time_step_spec)
    action_spec = tensor_spec.to_nest_array_spec(policy.action_spec)
    policy_state_spec = tensor_spec.to_nest_array_spec(policy.policy_state_spec)
    info_spec = tensor_spec.to_nest_array_spec(policy.info_spec)
    super(PyTFEagerPolicy, self).__init__(policy, time_step_spec, action_spec,
                                          policy_state_spec, info_spec)


@gin.configurable
class SavedModelPyTFEagerPolicy(PyTFEagerPolicyBase):
  """Exposes a numpy API for saved_model policies in Eager mode."""

  def __init__(self,
               model_path,
               time_step_spec,
               action_spec,
               policy_state_spec=(),
               info_spec=()):
    policy = tf.compat.v2.saved_model.load(model_path)
    super(SavedModelPyTFEagerPolicy,
          self).__init__(policy, time_step_spec, action_spec, policy_state_spec,
                         info_spec)

  def get_train_step(self):
    """Returns the training global step of the saved model."""
    return self._policy.train_step().numpy()
