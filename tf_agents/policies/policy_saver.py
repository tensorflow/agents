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

import functools

import tensorflow as tf

from tf_agents.policies import tf_policy
from tf_agents.utils import common


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
  saver = PolicySaver(policy, batch_size=1)

  for i in range(...):
    agent.train(...)
    if i % 100 == 0:
      saver.save('policy_%d' % global_step)
  ```

  To load and use the saved policy:

  ```python
  policy_step_spec = ...
  flat_spec = tf.nest.flatten(time_step_spec)


  saved_policy = tf.compat.v2.saved_model.load('policy_0')
  get_initial_state = saved_policy.signatures['get_initial_state']
  action = saved_policy.signatures['action']
  policy_state_dict = get_initial_state(batch_size)

  while True:
    flat_time_step = tf.nest.flatten(time_step)

    time_step_dict = dict(
      (spec.name, value) for spec, value in zip(flat_spec, flat_time_step))
    policy_step_dict = action(time_step_dict, policy_state_dict)
    policy_step = tf.nest.map_structure(
      lambda spec: policy_step_dict[spec.name], policy_step_spec)
    policy_state_dict = dict(
      (k, policy_step_dict[k]) for k in policy_state_dict)

    # Calculate the next time_step via interaction with the environment using
    # policy_step.action
    ...
  ```
  """

  def __init__(self, policy, batch_size=None, seed=None):
    """Initialize PolicySaver for  TF policy `policy`.

    Args:
      policy: A TF Policy.
      batch_size: The number of batch entries the policy will process at a time.
        This must be either `None` (unknown batch size) or a python integer.
      seed: Random seed for the `policy.action` call, if any (this should
        usually be `None`, except for testing).

    Raises:
      TypeError: If `policy` is not an instance of TFPolicy.
      ValueError: If any of the following `policy` specs are missing names, or
        the names collide: `policy.time_step_spec`, `policy.action_spec`,
        `policy.policy_state_spec`, `policy.info_spec`.
      ValueError: If `batch_size` is not either `None` or a python integer > 0.
      NotImplementedError: If created from TF1 with eager mode disabled.
    """
    if not tf.executing_eagerly():
      # TODO(b/129079730): Add support for TF1 using SavedModelBuilder.
      raise NotImplementedError(
          'Cannot create a PolicySaver in TF1 without eager mode enabled.')
    if not isinstance(policy, tf_policy.Base):
      raise TypeError('policy is not a TFPolicy.  Saw: %s' % type(policy))
    if (batch_size is not None and
        (not isinstance(batch_size, int) or batch_size < 1)):
      raise ValueError('Expected batch_size == None or python int > 0, saw: %s'
                       % (batch_size,))

    def true_if_missing_or_collision(spec, spec_names):
      if not spec.name or spec.name in spec_names:
        return True
      spec_names.add(spec.name)
      return False

    def check_spec(spec):
      spec_names = set()
      checked = [
          true_if_missing_or_collision(s, spec_names)
          for s in tf.nest.flatten(spec)]
      if any(checked):
        raise ValueError(
            'Specs contain either a missing name or a name collision.\n  '
            'Spec names: %s\n'
            % (tf.nest.map_structure(lambda s: s.name or '<MISSING>', spec),))

    check_spec({'time_step_spec': policy.time_step_spec,
                'policy_state_spec': policy.policy_state_spec})
    check_spec(policy.policy_step_spec)

    if batch_size is None:
      get_initial_state_fn = policy.get_initial_state
      get_initial_state_input_specs = (
          tf.TensorSpec(dtype=tf.int32, shape=(), name='batch_size'),)
    else:
      get_initial_state_fn = functools.partial(
          policy.get_initial_state, batch_size=batch_size)
      get_initial_state_input_specs = ()

    signatures = {
        'action': _function_with_signature(
            functools.partial(policy.action, seed=seed),
            input_specs=(policy.time_step_spec, policy.policy_state_spec),
            output_spec=policy.policy_step_spec,
            include_batch_dimension=True,
            batch_size=batch_size),
        'get_initial_state': _function_with_signature(
            get_initial_state_fn,
            input_specs=get_initial_state_input_specs,
            output_spec=policy.policy_state_spec,
            include_batch_dimension=False),
    }

    self._policy = policy
    self._signatures = signatures

  def save(self, export_dir):
    """Save the policy to the given `export_dir`."""
    return tf.saved_model.save(
        self._policy, export_dir, signatures=self._signatures)


def _function_with_signature(function,
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

  flat_input_spec = [
      _with_batch(spec) for spec in tf.nest.flatten(input_specs)]

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
