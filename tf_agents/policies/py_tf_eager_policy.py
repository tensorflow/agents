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

  def variables(self):
    return tf.nest.map_structure(lambda t: t.numpy(), self._policy.variables())

  def _get_initial_state(self, batch_size):
    return self._policy.get_initial_state(batch_size=batch_size)

  def _action(self, time_step, policy_state):
    time_step = nest_utils.batch_nested_array(time_step)
    # Avoid passing numpy arrays to avoid retracing of the tf.function.
    time_step = tf.nest.map_structure(tf.convert_to_tensor, time_step)
    policy_step = self._policy_action_fn(time_step, policy_state)
    return policy_step._replace(
        action=nest_utils.unbatch_nested_tensors_to_arrays(policy_step.action),
        # We intentionally do not convert the `state` so it is outputted as the
        # underlying policy generated it (i.e. in the form of a Tensor) which is
        # not necessarily compatible with a py-policy. However, we do so since
        # the `state` is fed back to the policy. So if it was converted, it'd be
        # required to convert back to the original form before calling the
        # method `action` of the policy again in the next step. If one wants to
        # store the `state` e.g. in replay buffer, then we suggest placing it
        # into the `info` field.
        info=nest_utils.unbatch_nested_tensors_to_arrays(policy_step.info))


@gin.configurable
class PyTFEagerPolicy(PyTFEagerPolicyBase):
  """Exposes a numpy API for TF policies in Eager mode."""

  def __init__(self, policy, use_tf_function=False):
    time_step_spec = tensor_spec.to_nest_array_spec(policy.time_step_spec)
    action_spec = tensor_spec.to_nest_array_spec(policy.action_spec)
    policy_state_spec = tensor_spec.to_nest_array_spec(policy.policy_state_spec)
    info_spec = tensor_spec.to_nest_array_spec(policy.info_spec)
    super(PyTFEagerPolicy,
          self).__init__(policy, time_step_spec, action_spec, policy_state_spec,
                         info_spec, use_tf_function)


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
    self._checkpoint = tf.train.Checkpoint(policy=policy)
    super(SavedModelPyTFEagerPolicy,
          self).__init__(policy, time_step_spec, action_spec, policy_state_spec,
                         info_spec)

  def get_train_step(self):
    """Returns the training global step of the saved model."""
    return self._policy.train_step().numpy()

  def variables(self):
    return self._policy.model_variables

  def update_from_checkpoint(self, checkpoint_path):
    """Allows users to update saved_model variables directly from a checkpoint.

    Args:
      checkpoint_path: Path to the checkpoint to restore and use to udpate this
        policy.
    """
    self._checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
