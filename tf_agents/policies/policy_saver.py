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

"""TF-Agents SavedModel API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools

import tensorflow as tf

from tf_agents.policies import tf_policy
from tf_agents.utils import common
from tf_agents.utils import nest_utils


def _true_if_missing_or_collision(spec, spec_names):
  if not spec.name or spec.name in spec_names:
    return True
  spec_names.add(spec.name)
  return False


def _rename_spec_with_nest_paths(spec):
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


class PolicySaver(object):
  """A `PolicySaver` allows you to save a `tf_policy.Policy` to `SavedModel`.

  The `save()` method exports a saved model to the requested export location.
  The SavedModel that is exported can be loaded via
  `tf.compat.v2.saved_model.load` (or `tf.saved_model.load` in TF2).  It
  will have available signatures (concrete functions): `action` and
  `get_initial_state`.

  Usage:
  ```python

  my_policy = agent.collect_policy
  saver = PolicySaver(policy, batch_size=None)

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

  def __init__(self,
               policy,
               batch_size=None,
               use_nest_path_signatures=True,
               seed=None,
               train_step=None):
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
        train_step defaults to -1.

    Raises:
      TypeError: If `policy` is not an instance of TFPolicy.
      ValueError: If use_nest_path_signatures is not used and any of the
        following `policy` specs are missing names, or the names collide:
        `policy.time_step_spec`, `policy.action_spec`,
        `policy.policy_state_spec`, `policy.info_spec`.
      ValueError: If `batch_size` is not either `None` or a python integer > 0.
    """
    if not isinstance(policy, tf_policy.Base):
      raise TypeError('policy is not a TFPolicy.  Saw: %s' % type(policy))
    if (batch_size is not None and
        (not isinstance(batch_size, int) or batch_size < 1)):
      raise ValueError(
          'Expected batch_size == None or python int > 0, saw: %s' %
          (batch_size,))

    action_fn_input_spec = (policy.time_step_spec, policy.policy_state_spec)
    if use_nest_path_signatures:
      action_fn_input_spec = _rename_spec_with_nest_paths(action_fn_input_spec)
    else:
      _check_spec(action_fn_input_spec)

    # Make a shallow copy as we'll be making some changes in-place.
    policy = copy.copy(policy)
    if train_step is None:
      train_step = tf.constant(-1)
    policy.train_step = train_step

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
        return original_action_fn(time_step, policy_state, seed=seed)
    else:
      action_fn = original_action_fn

    # We call get_concrete_function() for its side effect.
    get_initial_state_fn.get_concrete_function(*get_initial_state_input_specs)

    train_step_fn = common.function(lambda: train_step).get_concrete_function()

    action_fn = common.function()(action_fn)

    def add_batch_dim(spec):
      return tf.TensorSpec(
          shape=tf.TensorShape([batch_size]).concatenate(spec.shape),
          name=spec.name,
          dtype=spec.dtype)

    batched_time_step_spec = tf.nest.map_structure(add_batch_dim,
                                                   policy.time_step_spec)
    batched_policy_state_spec = tf.nest.map_structure(add_batch_dim,
                                                      policy.policy_state_spec)

    policy_step_spec = policy.policy_step_spec
    policy_state_spec = policy.policy_state_spec

    if use_nest_path_signatures:
      batched_time_step_spec = _rename_spec_with_nest_paths(
          batched_time_step_spec)
      batched_policy_state_spec = _rename_spec_with_nest_paths(
          batched_policy_state_spec)
      policy_step_spec = _rename_spec_with_nest_paths(policy_step_spec)
      policy_state_spec = _rename_spec_with_nest_paths(policy_state_spec)
    else:
      _check_spec(batched_time_step_spec)
      _check_spec(batched_policy_state_spec)
      _check_spec(policy_step_spec)
      _check_spec(policy_state_spec)

    # We call get_concrete_function() for its side effect.
    if batched_policy_state_spec:
      # Store the signature with a required policy state spec
      polymorphic_action_fn = action_fn
      polymorphic_action_fn.get_concrete_function(
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

    signatures = {
        'action':
            _function_with_flat_signature(
                polymorphic_action_fn,
                input_specs=action_fn_input_spec,
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
    }

    policy.action = polymorphic_action_fn
    policy.get_initial_state = get_initial_state_fn
    policy.train_step = train_step_fn

    self._policy = policy
    self._signatures = signatures

  def save(self, export_dir):
    """Save the policy to the given `export_dir`."""
    return tf.saved_model.save(
        self._policy, export_dir, signatures=self._signatures)


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
    tf.nest.assert_same_structure(outputs, output_spec)
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
