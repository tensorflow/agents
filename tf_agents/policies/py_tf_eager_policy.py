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
# Using Type Annotations.
from __future__ import print_function

import os
from typing import Optional, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import policy_saver
from tf_agents.policies import py_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class PyTFEagerPolicyBase(py_policy.PyPolicy):
  """Base class for py_policy instances of TF policies in Eager mode.

  Handles adding and removing batch dimensions from the actions and time_steps.
  Note if you have a tf_policy you should directly use the PyTFEagerPolicy class
  instead of this Base.
  """

  def __init__(self,
               policy: tf_policy.TFPolicy,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedArraySpec,
               policy_state_spec: types.NestedArraySpec,
               info_spec: types.NestedArraySpec,
               use_tf_function: bool = False):
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

  def __init__(self, policy: tf_policy.TFPolicy, use_tf_function: bool = False):
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
               model_path: Text,
               time_step_spec: Optional[ts.TimeStep] = None,
               action_spec: Optional[types.NestedTensorSpec] = None,
               policy_state_spec: types.NestedTensorSpec = (),
               info_spec: types.NestedTensorSpec = (),
               load_specs_from_pbtxt: bool = False):
    """Initializes a PyPolicy from a saved_model.

    *Note* (b/151318119): BoundedSpecs are converted to regular specs when saved
    into a proto as the `nested_structure_coder` from TF currently doesn't
    handle BoundedSpecs. Shape and dtypes will still match the original specs.

    Args:
      model_path: Path to a saved_model generated by the `policy_saver`.
      time_step_spec: Optional nested structure of ArraySpecs describing the
        policy's `time_step_spec`. This is not used by the
        SavedModelPyTFEagerPolicy, but may be accessed by other objects as it is
        part of the public policy API.
      action_spec: Optional nested structure of `ArraySpecs` describing the
        policy's `action_spec`. This is not used by the
        SavedModelPyTFEagerPolicy, but may be accessed by other objects as it is
        part of the public policy API.
      policy_state_spec: Optional nested structure of `ArraySpecs` describing
        the policy's `policy_state_spec`. This is not used by the
        SavedModelPyTFEagerPolicy, but may be accessed by other objects as it is
        part of the public policy API.
      info_spec: Optional nested structure of `ArraySpecs` describing the
        policy's `info_spec`. This is not used by the SavedModelPyTFEagerPolicy,
        but may be accessed by other objects as it is part of the public policy
        API.
      load_specs_from_pbtxt: If True the specs will be loaded from the proto
        file generated by the `policy_saver`.
    """
    policy = tf.compat.v2.saved_model.load(model_path)
    self._checkpoint = tf.train.Checkpoint(policy=policy)
    if not (time_step_spec or load_specs_from_pbtxt):
      raise ValueError(
          'To load a SavedModel policy you have to provide the specs, or'
          'enable loading from proto.')
    policy_specs = None
    if not time_step_spec and load_specs_from_pbtxt:
      spec_path = os.path.join(model_path, policy_saver.POLICY_SPECS_PBTXT)
      policy_specs = policy_saver.specs_from_collect_data_spec(
          tensor_spec.from_pbtxt_file(spec_path))
      time_step_spec = policy_specs['time_step_spec']
      action_spec = policy_specs['action_spec']
      policy_state_spec = policy_specs['policy_state_spec']
      info_spec = policy_specs['info_spec']
    super(SavedModelPyTFEagerPolicy,
          self).__init__(policy, time_step_spec, action_spec, policy_state_spec,
                         info_spec)
    # Override collect data_spec with whatever was loaded instead of relying
    # on trajectory_data_spec.
    if policy_specs:
      self._collect_data_spec = policy_specs['collect_data_spec']

  def get_train_step(self) -> types.Int:
    """Returns the training global step of the saved model."""
    return self._policy.get_train_step().numpy()

  def get_metadata(self):
    """Returns the metadata of the saved model."""
    return self._policy.get_metadata()

  def variables(self):
    return self._policy.model_variables

  def update_from_checkpoint(self, checkpoint_path: Text):
    """Allows users to update saved_model variables directly from a checkpoint.

    `checkpoint_path` is a path that was passed to either `PolicySaver.save()`
    or `PolicySaver.save_checkpoint()`. The policy looks for set of checkpoint
    files with the file prefix `<checkpoint_path>/variables/variables'

    Args:
      checkpoint_path: Path to the checkpoint to restore and use to udpate this
        policy.
    """
    file_prefix = os.path.join(checkpoint_path,
                               tf.saved_model.VARIABLES_DIRECTORY,
                               tf.saved_model.VARIABLES_FILENAME)
    status = self._checkpoint.read(file_prefix)
    # Check that all the variables in the policy were updated, but allow the
    # checkpoint to have additional variables. This helps sharing checkpoints
    # across policies.
    status.assert_existing_objects_matched().expect_partial()
