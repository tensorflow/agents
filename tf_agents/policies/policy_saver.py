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

"""TF-Agents SavedModel API."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import copy
import functools
import os
from typing import Any, Callable, Dict, Tuple, Optional, Text, cast, Sequence

from absl import logging
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.eager import def_function  # TF internal
# pylint:enable=g-direct-tensorflow-import


POLICY_SPECS_PBTXT = 'policy_specs.pbtxt'


def _true_if_missing_or_collision(spec, spec_names):
  if not spec.name or spec.name in spec_names:
    return True
  spec_names.add(spec.name)
  return False


def rename_spec_with_nest_paths(spec):
  renamed_spec = [
      tf.TensorSpec(shape=s.shape, name=path, dtype=s.dtype)
      for path, s in nest_utils.flatten_with_joined_paths(spec)
  ]
  return tf.nest.pack_sequence_as(spec, renamed_spec)


def _check_spec(spec):
  """Checks for missing or colliding names in specs."""
  spec_names = set()
  checked = [
      _true_if_missing_or_collision(s, spec_names)
      for s in tf.nest.flatten(spec)
  ]
  if any(checked):
    raise ValueError(
        'Specs contain either a missing name or a name collision.\n  '
        'Spec names: %s\n' %
        (tf.nest.map_structure(lambda s: s.name or '<MISSING>', spec),))


def add_batch_dim(spec, outer_dims):
  return tf.TensorSpec(
      shape=tf.TensorShape(outer_dims).concatenate(spec.shape),
      name=spec.name,
      dtype=spec.dtype)

InputFnType = Callable[[types.NestedTensor], Tuple[types.NestedTensor,
                                                   types.NestedTensor]]
InputFnAndSpecType = Tuple[InputFnType, types.NestedTensorSpec]


class PolicySaver(object):
  """A `PolicySaver` allows you to save a `tf_policy.Policy` to `SavedModel`.

  The `save()` method exports a saved model to the requested export location.
  The SavedModel that is exported can be loaded via
  `tf.compat.v2.saved_model.load` (or `tf.saved_model.load` in TF2).  The
  following signatures (concrete functions) are available: `action`,
  `get_initial_state`, and `get_train_step`.

  The attribute `model_variables` is also available when the saved_model is
  loaded which gives access to model variables in order to update them if
  needed.

  Usage:
  ```python

  my_policy = agent.collect_policy
  saver = PolicySaver(my_policy, batch_size=None)

  for i in range(...):
    agent.train(...)
    if i % 100 == 0:
      saver.save('policy_%d' % global_step)
  ```

  To load and use the saved policy directly:

  ```python
  saved_policy = tf.compat.v2.saved_model.load('policy_0')
  policy_state = saved_policy.get_initial_state(batch_size=3)
  time_step = ...
  while True:
    policy_step = saved_policy.action(time_step, policy_state)
    policy_state = policy_step.state
    time_step = f(policy_step.action)
    ...
  ```

  or to use the distributional form, e.g.:

  ```python
  batch_size = 3
  saved_policy = tf.compat.v2.saved_model.load('policy_0')
  policy_state = saved_policy.get_initial_state(batch_size=batch_size)
  time_step = ...
  while True:
    policy_step = saved_policy.distribution(time_step, policy_state)
    policy_state = policy_step.state
    time_step = f(policy_step.action.sample(batch_size))
    ...
  ```

  If using the flattened (signature) version, you will be limited to using
  dicts keyed by the specs' name fields.

  ```python
  saved_policy = tf.compat.v2.saved_model.load('policy_0')
  get_initial_state_fn = saved_policy.signatures['get_initial_state']
  action_fn = saved_policy.signatures['action']

  policy_state_dict = get_initial_state_fn(batch_size=3)
  time_step_dict = ...
  while True:
    time_step_state = dict(time_step_dict)
    time_step_state.update(policy_state_dict)
    policy_step_dict = action_fn(time_step_state)
    policy_state_dict = extract_policy_state_fields(policy_step_dict)
    action_dict = extract_action_fields(policy_step_dict)
    time_step_dict = f(action_dict)
    ...
  ```
  """

  def __init__(
      self,
      policy: tf_policy.TFPolicy,
      batch_size: Optional[int] = None,
      use_nest_path_signatures: bool = True,
      seed: Optional[types.Seed] = None,
      train_step: Optional[tf.Variable] = None,
      input_fn_and_spec: Optional[InputFnAndSpecType] = None,
      metadata: Optional[Dict[Text, tf.Variable]] = None
      ):
    """Initialize PolicySaver for  TF policy `policy`.

    Args:
      policy: A TF Policy.
      batch_size: The number of batch entries the policy will process at a time.
        This must be either `None` (unknown batch size) or a python integer.
      use_nest_path_signatures: SavedModel spec signatures will be created based
        on the sructure of the specs. Otherwise all specs must have unique
        names.
      seed: Random seed for the `policy.action` call, if any (this should
        usually be `None`, except for testing).
      train_step: Variable holding the train step for the policy. The value
        saved will be set at the time `saver.save` is called. If not provided,
        train_step defaults to -1. Note since the train step must be a variable
        it is not safe to create it directly in TF1 so in that case this is a
        required parameter.
      input_fn_and_spec: A `(input_fn, tensor_spec)` tuple where input_fn is a
        function that takes inputs according to tensor_spec and converts them to
        the `(time_step, policy_state)` tuple that is used as the input to the
        action_fn. When `input_fn_and_spec` is set, `tensor_spec` is the input
        for the action signature. When `input_fn_and_spec is None`, the action
        signature takes as input `(time_step, policy_state)`.
      metadata: A dictionary of `tf.Variables` to be saved along with the
        policy.

    Raises:
      TypeError: If `policy` is not an instance of TFPolicy.
      TypeError: If `metadata` is not a dictionary of tf.Variables.
      ValueError: If use_nest_path_signatures is not used and any of the
        following `policy` specs are missing names, or the names collide:
        `policy.time_step_spec`, `policy.action_spec`,
        `policy.policy_state_spec`, `policy.info_spec`.
      ValueError: If `batch_size` is not either `None` or a python integer > 0.
    """
    if not isinstance(policy, tf_policy.TFPolicy):
      raise TypeError('policy is not a TFPolicy.  Saw: %s' % type(policy))
    if (batch_size is not None and
        (not isinstance(batch_size, int) or batch_size < 1)):
      raise ValueError(
          'Expected batch_size == None or python int > 0, saw: %s' %
          (batch_size,))

    self._use_nest_path_signatures = use_nest_path_signatures

    action_fn_input_spec = (policy.time_step_spec, policy.policy_state_spec)
    if use_nest_path_signatures:
      action_fn_input_spec = rename_spec_with_nest_paths(action_fn_input_spec)
    else:
      _check_spec(action_fn_input_spec)

    # Make a shallow copy as we'll be making some changes in-place.
    saved_policy = tf.Module()
    saved_policy.collect_data_spec = copy.copy(policy.collect_data_spec)
    saved_policy.policy_state_spec = copy.copy(policy.policy_state_spec)

    if train_step is None:
      if not common.has_eager_been_enabled():
        raise ValueError('train_step is required in TF1 and must be a '
                         '`tf.Variable`: %s' % train_step)
      train_step = tf.Variable(
          -1,
          trainable=False,
          dtype=tf.int64,
          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
          shape=())
    elif not isinstance(train_step, tf.Variable):
      raise ValueError('train_step must be a TensorFlow variable: %s' %
                       train_step)

    # We will need the train step for the Checkpoint object.
    self._train_step = train_step
    saved_policy.train_step = self._train_step

    self._metadata = metadata or {}
    for key, value in self._metadata.items():
      if not isinstance(key, str):
        raise TypeError('Keys of metadata must be strings: %s' % key)
      if not isinstance(value, tf.Variable):
        raise TypeError('Values of metadata must be tf.Variable: %s' % value)
    saved_policy.metadata = self._metadata

    if batch_size is None:
      get_initial_state_fn = policy.get_initial_state
      get_initial_state_input_specs = (tf.TensorSpec(
          dtype=tf.int32, shape=(), name='batch_size'),)
    else:
      get_initial_state_fn = functools.partial(
          policy.get_initial_state, batch_size=batch_size)
      get_initial_state_input_specs = ()

    get_initial_state_fn = common.function()(get_initial_state_fn)

    original_action_fn = policy.action

    if seed is not None:

      def action_fn(time_step, policy_state):
        time_step = cast(ts.TimeStep, time_step)
        return original_action_fn(time_step, policy_state, seed=seed)
    else:
      action_fn = original_action_fn

    def distribution_fn(time_step, policy_state):
      """Wrapper for policy.distribution() in the SavedModel."""
      try:
        time_step = cast(ts.TimeStep, time_step)
        outs = policy.distribution(
            time_step=time_step, policy_state=policy_state)
        return tf.nest.map_structure(_composite_distribution, outs)
      except (TypeError, NotImplementedError) as e:
        # TODO(b/156526399): Move this to just the policy.distribution() call
        # once tfp.experimental.as_composite() properly handles LinearOperator*
        # components as well as TransformedDistributions.
        logging.warning(
            'WARNING: Could not serialize policy.distribution() for policy '
            '"%s". Calling saved_model.distribution() will raise the following '
            'assertion error: %s', policy, e)
        @common.function()
        def _raise():
          tf.Assert(False, [str(e)])
          return ()
        outs = _raise()

    # We call get_concrete_function() for its side effect: to ensure the proper
    # ConcreteFunction is stored in the SavedModel.
    get_initial_state_fn.get_concrete_function(*get_initial_state_input_specs)

    train_step_fn = common.function(
        lambda: saved_policy.train_step).get_concrete_function()
    get_metadata_fn = common.function(
        lambda: saved_policy.metadata).get_concrete_function()

    batched_time_step_spec = tf.nest.map_structure(
        lambda spec: add_batch_dim(spec, [batch_size]), policy.time_step_spec)
    batched_time_step_spec = cast(ts.TimeStep, batched_time_step_spec)
    batched_policy_state_spec = tf.nest.map_structure(
        lambda spec: add_batch_dim(spec, [batch_size]),
        policy.policy_state_spec)

    policy_step_spec = policy.policy_step_spec
    policy_state_spec = policy.policy_state_spec

    if use_nest_path_signatures:
      batched_time_step_spec = rename_spec_with_nest_paths(
          batched_time_step_spec)
      batched_policy_state_spec = rename_spec_with_nest_paths(
          batched_policy_state_spec)
      policy_step_spec = rename_spec_with_nest_paths(policy_step_spec)
      policy_state_spec = rename_spec_with_nest_paths(policy_state_spec)
    else:
      _check_spec(batched_time_step_spec)
      _check_spec(batched_policy_state_spec)
      _check_spec(policy_step_spec)
      _check_spec(policy_state_spec)

    if input_fn_and_spec is not None:
      # Store a signature based on input_fn_and_spec
      @common.function()
      def polymorphic_action_fn(example):
        action_inputs = input_fn_and_spec[0](example)
        tf.nest.map_structure(
            lambda spec, t: tf.Assert(spec.is_compatible_with(t[0]), [t]),
            action_fn_input_spec, action_inputs)
        return action_fn(*action_inputs)

      @common.function()
      def polymorphic_distribution_fn(example):
        action_inputs = input_fn_and_spec[0](example)
        tf.nest.map_structure(
            lambda spec, t: tf.Assert(spec.is_compatible_with(t[0]), [t]),
            action_fn_input_spec, action_inputs)
        return distribution_fn(*action_inputs)

      batched_input_spec = tf.nest.map_structure(
          lambda spec: add_batch_dim(spec, [batch_size]), input_fn_and_spec[1])
      # We call get_concrete_function() for its side effect: to ensure the
      # proper ConcreteFunction is stored in the SavedModel.
      polymorphic_action_fn.get_concrete_function(example=batched_input_spec)
      polymorphic_distribution_fn.get_concrete_function(
          example=batched_input_spec)

      action_input_spec = (input_fn_and_spec[1],)

    else:
      action_input_spec = action_fn_input_spec
      if batched_policy_state_spec:
        # Store the signature with a required policy state spec
        polymorphic_action_fn = common.function()(action_fn)
        polymorphic_action_fn.get_concrete_function(
            time_step=batched_time_step_spec,
            policy_state=batched_policy_state_spec)

        polymorphic_distribution_fn = common.function()(distribution_fn)
        polymorphic_distribution_fn.get_concrete_function(
            time_step=batched_time_step_spec,
            policy_state=batched_policy_state_spec)
      else:
        # Create a polymorphic action_fn which you can call as
        #  restored.action(time_step)
        # or
        #  restored.action(time_step, ())
        # (without retracing the inner action twice)
        @common.function()
        def polymorphic_action_fn(time_step,
                                  policy_state=batched_policy_state_spec):
          return action_fn(time_step, policy_state)

        polymorphic_action_fn.get_concrete_function(
            time_step=batched_time_step_spec,
            policy_state=batched_policy_state_spec)
        polymorphic_action_fn.get_concrete_function(
            time_step=batched_time_step_spec)

        @common.function()
        def polymorphic_distribution_fn(time_step,
                                        policy_state=batched_policy_state_spec):
          return distribution_fn(time_step, policy_state)

        polymorphic_distribution_fn.get_concrete_function(
            time_step=batched_time_step_spec,
            policy_state=batched_policy_state_spec)
        polymorphic_distribution_fn.get_concrete_function(
            time_step=batched_time_step_spec)

    signatures = {
        # CompositeTensors aren't well supported by old-style signature
        # mechanisms, so we do not have a signature for policy.distribution.
        'action':
            _function_with_flat_signature(
                polymorphic_action_fn,
                input_specs=action_input_spec,
                output_spec=policy_step_spec,
                include_batch_dimension=True,
                batch_size=batch_size),
        'get_initial_state':
            _function_with_flat_signature(
                get_initial_state_fn,
                input_specs=get_initial_state_input_specs,
                output_spec=policy_state_spec,
                include_batch_dimension=False),
        'get_train_step':
            _function_with_flat_signature(
                train_step_fn,
                input_specs=(),
                output_spec=train_step.dtype,
                include_batch_dimension=False),
        'get_metadata':
            _function_with_flat_signature(
                get_metadata_fn,
                input_specs=(),
                output_spec=tf.nest.map_structure(lambda v: v.dtype,
                                                  self._metadata),
                include_batch_dimension=False),
    }

    saved_policy.action = polymorphic_action_fn
    saved_policy.distribution = polymorphic_distribution_fn
    saved_policy.get_initial_state = get_initial_state_fn
    saved_policy.get_train_step = train_step_fn
    saved_policy.get_metadata = get_metadata_fn
    # Adding variables as an attribute to facilitate updating them.
    saved_policy.model_variables = policy.variables()

    # TODO(b/156779400): Move to a public API for accessing all trackable leaf
    # objects (once it's available).  For now, we have no other way of tracking
    # objects like Tables, Vocabulary files, etc.
    try:
      saved_policy._all_assets = policy._unconditional_checkpoint_dependencies  # pylint: disable=protected-access
    except AttributeError as e:
      if '_self_unconditional' in str(e):
        logging.warning(
            'Unable to capture all trackable objects in policy "%s".  This '
            'may be okay.  Error: %s', policy, e)
      else:
        raise e

    self._policy = saved_policy
    self._raw_policy = policy
    self._batch_size = batch_size
    self._signatures = signatures
    self._action_input_spec = action_input_spec
    self._policy_step_spec = policy_step_spec
    self._policy_state_spec = policy_state_spec

  @property
  def action_input_spec(self) -> types.NestedTensorSpec:
    """Tuple `(time_step_spec, policy_state_spec)` for feeding `action`.

    This describes the input of `action` in the SavedModel.

    This may differ from the original policy if `use_nest_path_signatures` was
    enabled.

    Returns:
      A nest of specs.
    """
    return self._action_input_spec

  @property
  def policy(self):
    return self._policy

  @property
  def policy_step_spec(self) -> types.NestedTensorSpec:
    """Spec that describes the output of `action` in the SavedModel.

    This may differ from the original policy if `use_nest_path_signatures` was
    enabled.

    Returns:
      A nest of specs.
    """
    return self._policy_step_spec

  @property
  def policy_state_spec(self) -> types.NestedTensorSpec:
    """Spec that describes the output of `get_initial_state` in the SavedModel.

    This may differ from the original policy if `use_nest_path_signatures` was
    enabled.

    Returns:
      A nest of specs.
    """
    return self._policy_state_spec

  @property
  def signatures(self) -> Dict[Text, Callable]:  # pylint: disable=g-bare-generic
    """Get the (flat) signatures used when exporting the `SavedModel`.

    Returns:
      A `dict` mapping each of "action", "get_initial_state",  "get_train_step"
      and "get_metadata" to their respective flat signatures.
    """
    return self._signatures

  def get_train_step(self) -> types.Int:
    """Returns the train step of the policy.

    Returns:
      An integer.
    """
    if tf.executing_eagerly():
      return self._train_step.numpy()
    else:
      return tf.identity(self._train_step)

  def get_metadata(self) -> Dict[Text, tf.Variable]:
    """Returns the metadata of the policy.

    Returns:
      An a dictionary of tf.Variable.
    """
    if tf.executing_eagerly():
      return {k: self._metadata[k].numpy() for k in self._metadata}
    else:
      return self._metadata

  def register_function(self,
                        name: str,
                        fn: InputFnType,
                        input_spec: types.NestedTensorSpec,
                        outer_dims: Sequence[Optional[int]] = (None,)) -> None:
    """Registers a function into the saved model.

    Note: There is no easy way to generate polymorphic functions. This pattern
    can be followed and the `get_concerete_function` can be called with named
    parameters to register more complex signatures. Those functions can then be
    passed to the `register_concrete_function` method.

    Args:
      name: Name of the attribute to use for the saved fn.
      fn: Function to register. Must be a callable following the input_spec as
        a single parameter.
      input_spec: A nest of tf.TypeSpec representing the time_steps.
        Provided by the user.
      outer_dims: The outer dimensions the saved fn will process at a time. By
        default a batch dimension is added to the input_spec.
    """
    if getattr(self._policy, name, None) is not None:
      raise ValueError('Policy already has an attribute registered with: %s' %
                       name)

    batched_spec = tf.nest.map_structure(lambda s: add_batch_dim(s, outer_dims),
                                         input_spec)
    tf_fn = common.function(fn)
    # We call get_concrete_function() for its side effect: to ensure the proper
    # ConcreteFunction is stored in the SavedModel.
    tf_fn.get_concrete_function(batched_spec)
    setattr(self._policy, name, tf_fn)

  def register_concrete_function(
      self,
      name: str,
      fn: def_function.Function,
      assets: Optional[Any] = None
  ) -> None:
    """Registers a function into the saved model.

    This gives you the flexibility to register any kind of polymorphic function
    by creating the concrete function that you wish to register.

    Args:
      name: Name of the attribute to use for the saved fn.
      fn: Function to register. Must be a callable following the input_spec as
        a single parameter.
      assets: Any extra checkpoint dependencies that must be captured in the
        module. Note variables are automatically captured.
    """
    if getattr(self._policy, name, None) is not None:
      raise ValueError('Policy already has an attribute registered with: %s' %
                       name)

    setattr(self._policy, name, fn)

    # TODO(b/182272788): Make `._list_all_concrete_functions` public.
    for i, concrete_fn in enumerate(fn._list_all_concrete_functions()):  # pylint: disable=protected-access
      setattr(self._policy, name + '__variables_%d' % i, concrete_fn.variables)

    if assets:
      setattr(self._policy, name + '__assets', assets)

  def save(self,
           export_dir: Text,
           options: Optional[tf.saved_model.SaveOptions] = None):
    """Save the policy to the given `export_dir`.

    Args:
      export_dir: Directory to save the policy to.
      options: Optional `tf.saved_model.SaveOptions` object.
    """
    tf.compat.v2.saved_model.save(
        self._policy, export_dir, signatures=self._signatures, options=options)

    temp_spec_file_name = '{}_temp'.format(POLICY_SPECS_PBTXT)
    temp_spec_output_path = os.path.join(export_dir, temp_spec_file_name)
    specs = {
        'collect_data_spec': self._policy.collect_data_spec,
        'policy_state_spec': self._policy.policy_state_spec
    }
    tensor_spec.to_pbtxt_file(temp_spec_output_path, specs)
    spec_output_path = os.path.join(export_dir, POLICY_SPECS_PBTXT)
    # By moving the file to its final location makes it safer to wait for the
    # file (e.g. from a separate binary). The parameter `overwrite=True`
    # reproduces the exact previous behavior.
    tf.io.gfile.rename(temp_spec_output_path, spec_output_path, overwrite=True)

  def save_checkpoint(self,
                      export_dir: Text,
                      options: Optional[tf.train.CheckpointOptions] = None):
    """Saves the policy as a checkpoint to the given `export_dir`.

    This will only work with checkpoints generated in TF2.x.

    For the checkpoint to be useful users should first call `save` to generate a
    saved_model of the policy. Checkpoints can then be used to update the policy
    without having to reload the saved_model, or saving multiple copies of the
    `saved_model.pb` file.

    The checkpoint is always created in the sub-directory 'variables/' and the
    checkpoint file prefix used is 'variables'. The checkpoint files are as
    follows:
       * export_dir/variables/variables.index
       * export_dir/variables/variables-xxxxx-of-xxxxx

    This makes the files compatible with the checkpoint part of full saved
    models, which enables you to load a saved model made up from the graph part
    of a full saved model and the variables part of a checkpoint.

    Args:
      export_dir: Directory to save the checkpoint to.
      options: Optional `tf.train.CheckpointOptions` object.
    """
    # In addition to the policy, also list dependencies on model_variables and
    # train_step so the checkpoint can be combined with a saved graph from a
    # full saved model.
    checkpoint = tf.compat.v2.train.Checkpoint(
        policy=self._policy,
        model_variables=self._policy.model_variables,
        train_step=self._train_step)
    # Use write() to make sure that the file prefix is not modified by appending
    # a save counter value.
    file_prefix = os.path.join(export_dir, tf.saved_model.VARIABLES_DIRECTORY,
                               tf.saved_model.VARIABLES_FILENAME)
    checkpoint.write(file_prefix, options=options)


def _function_with_flat_signature(function,
                                  input_specs,
                                  output_spec,
                                  include_batch_dimension,
                                  batch_size=None):
  """Create a tf.function with a given signature for export.

  Args:
    function: A callable that can be wrapped in tf.function.
    input_specs: A tuple nested specs declaring ordered arguments to function.
    output_spec: The nested spec describing the output of the function.
    include_batch_dimension: Python bool, whether to prepend a batch dimension
      to inputs and outputs.
    batch_size: Known batch size, or `None` for unknown.  Ignored if
      `include_batch_dimension == False`.

  Returns:
    A `tf.function` with the given input spec that returns a `dict` mapping
    output spec keys to corresponding output values.
  """

  def _with_batch(spec):
    if include_batch_dimension:
      return tf.TensorSpec(
          shape=tf.TensorShape([batch_size]).concatenate(spec.shape),
          name=spec.name,
          dtype=spec.dtype)
    else:
      return spec

  flat_input_spec = [_with_batch(spec) for spec in tf.nest.flatten(input_specs)]

  def as_dict(outputs, output_spec):
    nest_utils.assert_same_structure(outputs, output_spec)
    flat_outputs = tf.nest.flatten(outputs)
    flat_names = [s.name for s in tf.nest.flatten(output_spec)]
    return dict(zip(flat_names, flat_outputs))

  @common.function(input_signature=flat_input_spec)
  def function_with_signature(*input_list):
    inputs_ = tf.nest.pack_sequence_as(input_specs, input_list)
    outputs_ = function(*inputs_)
    dict_outputs_ = as_dict(outputs_, output_spec)
    return dict_outputs_

  return function_with_signature


def specs_from_collect_data_spec(
    loaded_policy_specs: types.NestedTensorSpec
) -> Dict[types.NestedSpec, types.NestedSpec]:
  """Creates policy specs from specs loaded from disk.

  The PolicySaver saves policy specs next to the saved model as
  a `struct.StructuredValue` proto. This recreates the
  original specs from the proto.

  Pass the proto loaded from the file with `tensor_spec.from_pbtxt_file()`
  to this function.

  Args:
     loaded_policy_specs: `struct.StructuredValue` proto that had been
       previously created by PolicySaver as a pbtxt.

  Returns:
    A dict with specs extracted from the proto. The dict contains the following
    keys and values. Except `time_step_spec` all the specs are nests of
    `ArraySpecs`.
       * `collect_data_spec`: Collect data spec for the policy.
       * `time_step_spec`: `TimeStepSpec` for the policy.
       * `action_spec`:  Action spec for the policy
       * `policy_state_spec`: State spec for the policy.
       * `info_spec`: Info spec for the policy.
  """
  policy_specs = tensor_spec.to_nest_array_spec(loaded_policy_specs)
  collect_data_spec = policy_specs['collect_data_spec']
  policy_state_spec = policy_specs['policy_state_spec']
  time_step_spec = ts.TimeStep(
      step_type=collect_data_spec.step_type,
      reward=collect_data_spec.reward,
      discount=collect_data_spec.discount,
      observation=collect_data_spec.observation)
  action_spec = collect_data_spec.action
  info_spec = collect_data_spec.policy_info
  return dict(
      collect_data_spec=collect_data_spec,
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      policy_state_spec=policy_state_spec,
      info_spec=info_spec)


def _composite_distribution(d):
  """Converts tfp Distributions to CompositeTensors."""
  return (tfp.experimental.as_composite(d)
          if isinstance(d, tfp.distributions.Distribution)
          else d)
