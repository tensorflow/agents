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

"""Agent Converter API and converters."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import typing

import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import composite
from tf_agents.utils import nest_utils

import typing_extensions as te


def _is_transition_like(value):
  """Helper to identify values that are transition like."""
  if isinstance(value, trajectory.Transition):
    return True

  fields = getattr(value, '_fields', None)
  if fields and trajectory.Transition._fields == fields:
    return True

  return False


def _is_trajectory_like(value):
  """Helper to identify values that are trajectory like."""
  if isinstance(value, trajectory.Trajectory):
    return True

  fields = getattr(value, '_fields', None)
  if fields and trajectory.Trajectory._fields == fields:
    return True

  return False


def _as_tfa_transition(value: typing.Tuple[typing.Any, typing.Any, typing.Any]):
  """Makes sure the transition and its values are TFA types."""
  time_step, action_step, next_time_step = value
  time_step = ts.TimeStep(*time_step)
  action_step = policy_step.PolicyStep(*action_step)
  next_time_step = ts.TimeStep(*next_time_step)
  return trajectory.Transition(time_step, action_step, next_time_step)


class DataContext(tf.Module):
  """A class that stores useful data for performing data conversions."""

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      info_spec: types.NestedTensorSpec,
      policy_state_spec: types.NestedTensorSpec = (),
      use_half_transition: bool = False,
  ):
    """Creates a DataContext.

    Note: The context does not store a state spec, or other information about
    a Policy's internal state.  Policy state is not typically stored in a
    replay buffer or on disk, except when the policy explicitly chooses to
    store it by adding the state as a field inside its `info` output.  In
    those cases, the internal policy state spec is represented as part of the
    `info_spec`.

    Args:
      time_step_spec: A nest of `tf.TimeStep` representing the time_steps.
      action_spec: A nest of `tf.TypeSpec` representing the actions.
      info_spec: A nest of `tf.TypeSpec` representing the policy's info.
        (Typically this is the info emitted by the collect policy).
      policy_state_spec: A nest of `tf.TypeSpec` representing the policy's
        state.
      use_half_transition: A special transition that the next_time_step contains
        a dummy observation.

    Raises:
      TypeError: If any of the specs are not nests containing tf.TypeSpec
        objects.
    """
    def _each_isinstance(spec, spec_types):
      """Checks if each element of `spec` is instance of `spec_types`."""
      return all([isinstance(s, spec_types) for s in tf.nest.flatten(spec)])

    for (spec, label) in ((time_step_spec, 'time_step_spec'),
                          (action_spec, 'action_spec'),
                          (info_spec, 'info_spec')):
      if not _each_isinstance(spec, tf.TypeSpec):
        raise TypeError(
            '{} has to contain TypeSpec (TensorSpec, '
            'SparseTensorSpec, etc) objects, but received: {}'
            .format(label, spec))

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._info_spec = info_spec
    self._policy_state_spec = policy_state_spec
    self._trajectory_spec = trajectory.Trajectory(
        step_type=time_step_spec.step_type,
        observation=time_step_spec.observation,
        action=action_spec,
        policy_info=info_spec,
        next_step_type=time_step_spec.step_type,
        reward=time_step_spec.reward,
        discount=time_step_spec.discount)

    if use_half_transition:
      next_time_step_spec = time_step_spec._replace(
          observation=tensor_spec.TensorSpec(
              (), dtype=tf.float32, name='observation'))
    else:
      next_time_step_spec = time_step_spec
    self._transition_spec = trajectory.Transition(
        time_step=time_step_spec,
        action_step=policy_step.PolicyStep(action=action_spec,
                                           state=policy_state_spec,
                                           info=info_spec),
        next_time_step=next_time_step_spec)

  @property
  def time_step_spec(self) -> ts.TimeStep:
    return self._time_step_spec

  @property
  def action_spec(self) -> types.NestedTensorSpec:
    return self._action_spec

  @property
  def info_spec(self) -> types.NestedTensorSpec:
    return self._info_spec

  @property
  def policy_state_spec(self) -> types.NestedTensorSpec:
    return self._policy_state_spec

  @property
  def trajectory_spec(self) -> trajectory.Trajectory:
    return self._trajectory_spec

  @property
  def transition_spec(self) -> trajectory.Transition:
    return self._transition_spec


def _validate_trajectory(
    value: trajectory.Trajectory,
    trajectory_spec: trajectory.Trajectory,
    sequence_length: typing.Optional[int],
    num_outer_dims: te.Literal[1, 2] = 2):  # pylint: disable=bad-whitespace
  """Validate a Trajectory given its spec and a sequence length."""
  if not nest_utils.is_batched_nested_tensors(
      value, trajectory_spec, num_outer_dims=num_outer_dims,
      allow_extra_fields=True):
    debug_str_1 = tf.nest.map_structure(lambda tp: tp.shape, value)
    debug_str_2 = tf.nest.map_structure(
        lambda spec: spec.shape, trajectory_spec)

    shape_str = (
        'two outer dimensions' if num_outer_dims == 2
        else 'one outer dimension')
    shape_prefix_str = '[B, T]' if num_outer_dims == 2 else '[B]'
    raise ValueError(
        'All of the Tensors in `value` must have {shape_str}. Specifically, '
        'tensors must have shape `{shape_prefix_str} + spec.shape`.\n'
        'Full shapes of value tensors:\n  {debug_str_1}.\n'
        'Expected shapes (excluding the {shape_str}):\n  {debug_str_2}.'
        .format(
            shape_str=shape_str,
            debug_str_1=debug_str_1,
            debug_str_2=debug_str_2,
            shape_prefix_str=shape_prefix_str))

  # If we have a time dimension and a train_sequence_length, make sure they
  # match.
  if sequence_length is not None:
    def check_shape(path, t):  # pylint: disable=invalid-name
      if t.shape[1] != sequence_length:
        debug_str = tf.nest.map_structure(lambda tp: tp.shape, value)
        raise ValueError(
            'The agent was configured to expect a `sequence_length` '
            'of \'{seq_len}\'. Value is expected to be shaped `[B, T] + '
            'spec.shape` but at least one of the Tensors in `value` has a '
            'time axis dim value \'{t_dim}\' vs '
            'the expected \'{seq_len}\'.\nFirst such tensor is:\n\t'
            'value.{path}. \nFull shape structure of '
            'value:\n\t{debug_str}'.format(
                seq_len=sequence_length,
                t_dim=t.shape[1],
                path=path,
                debug_str=debug_str))
    nest_utils.map_structure_with_paths(check_shape, value)


def _validate_transition(value: trajectory.Transition,
                         transition_spec: trajectory.Transition,
                         num_outer_dims: int):
  """Checks the given Transition for batch and time outer dimensions."""

  if value.action_step.state:
    # When state is not (), it does not have time dimension, therefore it needs
    # to be validated separately.
    _validate_state(value.action_step.state,
                    transition_spec.action_step.state)
    action_step_without_state = value.action_step._replace(state=())
    value_to_validate = value._replace(action_step=action_step_without_state)
    action_spec_without_state = (
        transition_spec.action_step._replace(state=()))
    spec_to_validate = transition_spec._replace(
        action_step=action_spec_without_state)
  else:
    value_to_validate = value
    spec_to_validate = transition_spec

  if not nest_utils.is_batched_nested_tensors(
      value_to_validate,
      spec_to_validate,
      num_outer_dims=num_outer_dims,
      allow_extra_fields=True):
    debug_str_1 = tf.nest.map_structure(
        lambda tp: tp.shape, value_to_validate)
    debug_str_2 = tf.nest.map_structure(
        lambda spec: spec.shape, spec_to_validate)
    raise ValueError(
        'All of the Tensors in `value` must have a single outer (batch size) '
        'dimension. Specifically, tensors must have {} outer dimensions.'
        '\nFull shapes of value tensors:\n  {}.\n'
        'Expected shapes (excluding the outer dimensions):\n  {}.'
        .format(num_outer_dims, debug_str_1, debug_str_2))


def _validate_state(state: types.NestedTensor, spec: types.NestedTensorSpec):
  if not nest_utils.is_batched_nested_tensors(
      state,
      spec,
      num_outer_dims=1,
      allow_extra_fields=False):
    raise ValueError('action_step.state does not match spec. '
                     'action_step.state.shape: {state_shape}, '
                     'spec.shape: {spec_shape}'
                     'action_step.state: {state_value}, spec: '
                     '{spec_value}'.format(
                         state_shape=state.shape,
                         spec_shape=spec.shape,
                         state_value=state,
                         spec_value=spec))


class AsTrajectory(tf.Module):
  """Class that validates and converts other data types to Trajectory.

  Note that validation and conversion allows values to contain dictionaries
  with extra keys as compared to the the specs in the data context.  These
  additional entries / observations are ignored and dropped during conversion.

  This non-strict checking allows users to provide additional info and
  observation keys at input without having to manually prune them before
  converting.
  """

  def __init__(self,
               data_context: DataContext,
               sequence_length: typing.Optional[int] = None,
               num_outer_dims: te.Literal[1, 2] = 2):  # pylint: disable=bad-whitespace
    """Create the AsTrajectory converter.

    Args:
      data_context: An instance of `DataContext`, typically accessed from the
        `TFAgent.data_context` property.
      sequence_length: The required time dimension value (if any), typically
         determined by the subclass of `TFAgent`.
      num_outer_dims: Expected number of outer dimensions.  Either 1 or 2.
         If `1`, call expects an outer batch dimension.  If `2`, then call
         expects the two outer dimensions `[batch, time]`.
    """
    self._data_context = data_context
    self._sequence_length = sequence_length
    self._num_outer_dims = num_outer_dims

  def __call__(self, value: typing.Any) -> trajectory.Trajectory:
    """Convers `value` to a Trajectory.  Performs data validation and pruning.

    - If `value` is already a `Trajectory`, only validation is performed.
    - If `value` is a `Transition` with tensors containing two (`[B, T]`)
      outer dims, then it is simply repackaged to a `Trajectory` and then
      validated.
    - If `value` is a `Transition` with tensors containing one (`[B]`) outer
      dim, a `ValueError` is raised.

    Args:
      value: A `Trajectory` or `Transition` object to convert.

    Returns:
      A validated and pruned `Trajectory`.

    Raises:
      TypeError: If `value` is not one of `Trajectory` or `Transition`.
      ValueError: If `value` has structure that doesn't match the converter's
        spec.
      TypeError: If `value` has a structure that doesn't match the converter's
        spec.
      ValueError: If `value` is a `Transition` without a time dimension, as
        training Trajectories typically have batch and time dimensions.
    """
    if isinstance(value, trajectory.Trajectory):
      pass
    elif isinstance(value, trajectory.Transition):
      value = trajectory.Trajectory(
          step_type=value.time_step.step_type,
          observation=value.time_step.observation,
          action=value.action_step.action,
          policy_info=value.action_step.info,
          next_step_type=value.next_time_step.step_type,
          reward=value.next_time_step.reward,
          discount=value.next_time_step.discount)
    else:
      raise TypeError('Input type not supported: {}'.format(value))
    _validate_trajectory(
        value, self._data_context.trajectory_spec,
        sequence_length=self._sequence_length,
        num_outer_dims=self._num_outer_dims)
    value = nest_utils.prune_extra_keys(
        self._data_context.trajectory_spec, value)
    return value


class AsTransition(tf.Module):
  """Class that validates and converts other data types to Transition.

  Note that validation and conversion allows values to contain dictionaries
  with extra keys as compared to the specs in the data context.  These
  additional entries / observations are ignored and dropped during conversion.

  This non-strict checking allows users to provide additional info and
  observation keys at input without having to manually prune them before
  converting.
  """

  def __init__(self, data_context: DataContext, squeeze_time_dim=False,
               prepend_t0_to_next_time_step=False):
    """Creates the AsTransition converter.

    Args:
      data_context: An instance of `DataContext`, typically accessed from the
        `TFAgent.data_context` property.
      squeeze_time_dim: Whether to emit a transition without time
        dimensions.  If `True`, incoming trajectories are expected
        to have a time dimension of exactly `2`, and emitted Transitions
        will have no time dimensions.
      prepend_t0_to_next_time_step: Whether to add t0 to next_time_step. This
        option is useful when using sequential model and can allow target
        network be able to take more information. Resulting shape of
        time_step.observation is `[B, T, ...]` and resulting shape of
        next_time_step.observation is `[B, T+1, ...]`.
    """
    self._data_context = data_context
    self._squeeze_time_dim = squeeze_time_dim
    self._prepend_t0_to_next_time_step = prepend_t0_to_next_time_step

  def __call__(self, value: typing.Any) -> trajectory.Transition:
    """Converts `value` to a Transition.  Performs data validation and pruning.

    - If `value` is already a `Transition`, only validation is performed.
    - If `value` is a `Trajectory` and `squeeze_time_dim = True` then
      `value` it must have tensors with shape `[B, T=2]` outer dims.
      This is converted to a `Transition` object without a time
      dimension.
    - If `value` is a `Trajectory` with tensors containing a time dimension
      having `T != 2`, a `ValueError` is raised.

    Args:
      value: A `Trajectory` or `Transition` object to convert.

    Returns:
      A validated and pruned `Transition`.  If `squeeze_time_dim = True`,
      the resulting `Transition` has tensors with shape `[B, ...]`.  Otherwise,
      the tensors will have shape `[B, T - 1, ...]`.

    Raises:
      TypeError: If `value` is not one of `Trajectory` or `Transition`.
      ValueError: If `value` has structure that doesn't match the converter's
        spec.
      TypeError: If `value` has a structure that doesn't match the converter's
        spec.
      ValueError: If `squeeze_time_dim=True` and `value` is a `Trajectory`
        with a time dimension having value other than `T=2`.
    """
    if _is_transition_like(value):
      value = _as_tfa_transition(value)
    elif _is_trajectory_like(value):
      required_sequence_length = 2 if self._squeeze_time_dim else None
      _validate_trajectory(
          value,
          self._data_context.trajectory_spec,
          sequence_length=required_sequence_length)
      value = trajectory.to_transition(value)
      # Remove the now-singleton time dim.
      if self._squeeze_time_dim:
        value = tf.nest.map_structure(
            lambda x: composite.squeeze(x, axis=1), value)
    else:
      raise TypeError('Input type not supported: {}'.format(value))

    num_outer_dims = 1 if self._squeeze_time_dim else 2
    _validate_transition(
        value, self._data_context.transition_spec, num_outer_dims)

    value = nest_utils.prune_extra_keys(
        self._data_context.transition_spec, value)

    if self._prepend_t0_to_next_time_step:
      # This is useful when using sequential model. It allows target_q network
      # to take all the information.
      next_time_step_with_t0 = value.next_time_step._replace(
          observation=tf.nest.map_structure(
              lambda x, y: tf.concat([x[:, :1, ...], y], axis=1),
              value.time_step.observation, value.next_time_step.observation))

      value = value._replace(next_time_step=next_time_step_with_t0)
    return value


class AsHalfTransition(tf.Module):
  """Class that validates and converts other data types to HalfTransition.

  HalfTransition is a special Transition whose next_time_step contains a dummy
  observation.

  Note that validation and conversion allows values to contain dictionaries
  with extra keys as compared to the specs in the data context.  These
  additional entries / observations are ignored and dropped during conversion.

  This non-strict checking allows users to provide additional info and
  observation keys at input without having to manually prune them before
  converting.
  """

  def __init__(self, data_context: DataContext, squeeze_time_dim=False):
    """Create the AsTransition converter.

    Args:
      data_context: An instance of `DataContext`, typically accessed from the
        `TFAgent.data_context` property.
      squeeze_time_dim: Whether to emit a transition without time
        dimensions.  If `True`, incoming trajectories are expected
        to have a time dimension of exactly `2`, and emitted Transitions
        will have no time dimensions.
    """
    self._data_context = data_context
    self._squeeze_time_dim = squeeze_time_dim

  def __call__(self, value: typing.Any) -> trajectory.Transition:
    """Convert `value` to an N-step Transition; validate data & prune.

    - If `value` is already a `Transition`, only validation is performed.
    - If `value` is a `Trajectory` with tensors containing a time dimension
      having `T != n + 1`, a `ValueError` is raised.

    Args:
      value: A `Trajectory` or `Transition` object to convert.

    Returns:
      A validated and pruned `Transition`.  If `squeeze_time_dim = True`,
      the resulting `Transition` has tensors with shape `[B, ...]`.  Otherwise,
      the tensors will have shape `[B, T - 1, ...]`.

    Raises:
      TypeError: If `value` is not one of `Trajectory` or `Transition`.
      ValueError: If `value` has structure that doesn't match the converter's
        spec.
      TypeError: If `value` has a structure that doesn't match the converter's
        spec.
      ValueError: If `n != None` and `value` is a `Trajectory`
        with a time dimension having value other than `T=n + 1`.
    """
    if _is_transition_like(value):
      value = _as_tfa_transition(value)
    elif _is_trajectory_like(value):
      required_sequence_length = 1 if self._squeeze_time_dim else None
      _validate_trajectory(
          value,
          self._data_context.trajectory_spec,
          sequence_length=required_sequence_length)
      if self._squeeze_time_dim:
        value = tf.nest.map_structure(lambda e: tf.squeeze(e, axis=1), value)
      policy_steps = policy_step.PolicyStep(
          action=value.action, state=(), info=value.policy_info)
      # TODO(b/130244652): Consider replacing 0 rewards & discounts with ().
      time_steps = ts.TimeStep(
          value.step_type,
          reward=tf.nest.map_structure(tf.zeros_like, value.reward),  # unknown
          discount=tf.zeros_like(value.discount),  # unknown
          observation=value.observation)
      next_time_steps = ts.TimeStep(
          step_type=value.next_step_type,
          reward=value.reward,
          discount=value.discount,
          observation=tf.zeros_like(value.discount))
      value = trajectory.Transition(time_steps, policy_steps, next_time_steps)
    else:
      raise TypeError('Input type not supported: {}'.format(value))

    num_outer_dims = 1 if self._squeeze_time_dim else 2
    _validate_transition(
        value, self._data_context.transition_spec, num_outer_dims)

    value = nest_utils.prune_extra_keys(
        self._data_context.transition_spec, value)
    return value


class AsNStepTransition(tf.Module):
  """Class that validates and converts other data types to N-step Transition.

  Note that validation and conversion allows values to contain dictionaries
  with extra keys as compared to the the specs in the data context.  These
  additional entries / observations are ignored and dropped during conversion.

  This non-strict checking allows users to provide additional info and
  observation keys at input without having to manually prune them before
  converting.
  """

  def __init__(self,
               data_context: DataContext,
               gamma: types.Float,
               n: typing.Optional[int] = None):
    """Create the AsNStepTransition converter.

    For more details on how `Trajectory` objects are converted to N-step
    `Transition` objects, see
    `tf_agents.trajectories.trajectory.to_n_step_transition`.

    Args:
      data_context: An instance of `DataContext`, typically accessed from the
        `TFAgent.data_context` property.
      gamma: A floating point scalar; the discount factor.
      n: (Optional.) The expected number of frames given a `Trajectory` input.
        Given a `Trajectory` with tensors shaped `[B, T, ...]`, we ensure that
        `T = n + 1`.  Only used for validation.
    """
    self._data_context = data_context
    self._gamma = gamma
    self._n = n

  def __call__(self, value: typing.Any) -> trajectory.Transition:
    """Convert `value` to an N-step Transition; validate data & prune.

    - If `value` is already a `Transition`, only validation is performed.
    - If `value` is a `Trajectory` with tensors containing a time dimension
      having `T != n + 1`, a `ValueError` is raised.

    Args:
      value: A `Trajectory` or `Transition` object to convert.

    Returns:
      A validated and pruned `Transition`.  If `squeeze_time_dim = True`,
      the resulting `Transition` has tensors with shape `[B, ...]`.  Otherwise,
      the tensors will have shape `[B, T - 1, ...]`.

    Raises:
      TypeError: If `value` is not one of `Trajectory` or `Transition`.
      ValueError: If `value` has structure that doesn't match the converter's
        spec.
      TypeError: If `value` has a structure that doesn't match the converter's
        spec.
      ValueError: If `n != None` and `value` is a `Trajectory`
        with a time dimension having value other than `T=n + 1`.
    """
    if _is_transition_like(value):
      value = _as_tfa_transition(value)
    elif _is_trajectory_like(value):
      _validate_trajectory(
          value,
          self._data_context.trajectory_spec,
          sequence_length=None if self._n is None else self._n + 1)
      value = trajectory.to_n_step_transition(value, gamma=self._gamma)
    else:
      raise TypeError('Input type not supported: {}'.format(value))

    _validate_transition(
        value, self._data_context.transition_spec, num_outer_dims=1)
    value = nest_utils.prune_extra_keys(
        self._data_context.transition_spec, value)
    return value
