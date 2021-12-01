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

"""Trajectory containing time_step transition information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
import pprint
from typing import NamedTuple, Optional

import numpy as np
import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import composite
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops


class Trajectory(
    NamedTuple('Trajectory', [
        ('step_type', types.SpecTensorOrArray),
        ('observation', types.NestedSpecTensorOrArray),
        ('action', types.NestedSpecTensorOrArray),
        ('policy_info', types.NestedSpecTensorOrArray),
        ('next_step_type', types.SpecTensorOrArray),
        ('reward', types.NestedSpecTensorOrArray),
        ('discount', types.SpecTensorOrArray),
    ])):
  """A tuple that represents a trajectory.

  A `Trajectory` represents a sequence of aligned time steps. It captures the
  observation, step_type from current time step with the computed action
  and policy_info. Discount, reward and next_step_type come from the next
  time step.

  Attributes:
    step_type: A `StepType`.
    observation: An array (tensor), or a nested dict, list or tuple of arrays
      (tensors) that represents the observation.
    action: An array/a tensor, or a nested dict, list or tuple of actions. This
      represents action generated according to the observation.
    policy_info: An arbitrary nest that contains auxiliary information related
      to the action. Note that this does not include the policy/RNN state which
      was used to generate the action.
    next_step_type: The `StepType` of the next time step.
    reward: An array/a tensor, or a nested dict, list, or tuple of rewards.
      This represents the rewards and/or constraint satisfiability after
      performing the action in an environment.
    discount: A scalar that representing the discount factor to multiply with
      future rewards.
  """
  __slots__ = ()

  def is_first(self) -> types.Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, ts.StepType.FIRST)
    return self.step_type == ts.StepType.FIRST

  def is_mid(self) -> types.Bool:
    if tf.is_tensor(self.step_type):
      return tf.logical_and(
          tf.equal(self.step_type, ts.StepType.MID),
          tf.equal(self.next_step_type, ts.StepType.MID))
    return (self.step_type == ts.StepType.MID) & (
        self.next_step_type == ts.StepType.MID)

  def is_last(self) -> types.Bool:
    if tf.is_tensor(self.next_step_type):
      return tf.equal(self.next_step_type, ts.StepType.LAST)
    return self.next_step_type == ts.StepType.LAST

  def is_boundary(self) -> types.Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, ts.StepType.LAST)
    return self.step_type == ts.StepType.LAST

  def replace(self, **kwargs) -> 'Trajectory':
    """Exposes as namedtuple._replace.

    Usage:
    ```
      new_trajectory = trajectory.replace(policy_info=())
    ```

    This returns a new trajectory with an empty policy_info.

    Args:
      **kwargs: key/value pairs of fields in the trajectory.

    Returns:
      A new `Trajectory`.
    """
    return self._replace(**kwargs)

  def __repr__(self):
    return 'Trajectory(\n' + pprint.pformat(dict(self._asdict())) + ')'


# TODO(b/162101981): Move to its own file.
class Transition(
    NamedTuple('Transition', [
        ('time_step', ts.TimeStep),
        ('action_step', policy_step.PolicyStep),
        ('next_time_step', ts.TimeStep)
    ])):
  """A tuple that represents a transition.

  A `Transition` represents a `S, A, S'` sequence of operations.  Tensors
  within a `Transition` are typically shaped `[B, ...]` where `B` is the
  batch size.

  In some cases Transition objects are used to store time-shifted intermediate
  values for RNN computations, in which case the stored tensors are
  shaped `[B, T, ...]`.

  In other cases, `Transition` objects store n-step transitions
  `S_t, A_t, S_{t+N}` where the associated reward and discount in
  `next_time_step` are calculated as:

  ```python
  next_time_step.reward = r_t +
                          g^{1} * d_t * r_{t+1} +
                          g^{2} * d_t * d_{t+1} * r_{t+2} +
                          g^{3} * d_t * d_{t+1} * d_{t+2} * r_{t+3} +
                          ...
                          g^{N-1} * d_t * ... * d_{t+N-2} * r_{t+N-1}

  next_time_step.discount = g^{N-1} * d_t * d_{t+1} * ... * d_{t+N-1}.
  ```
  See `to_n_step_transition` for an example that converts `Trajectory` objects
  to this format.

  Attributes:
    time_step: The initial state, reward, and discount.
    action_step: The action, policy info, and possibly policy state taken.
      (Note, `action_step.state` should not typically be stored in e.g.
      a replay buffer, except a copy inside `policy_step.info` as a special
      case for algorithms that choose to do this).
    next_time_step: The final state, reward, and discount.
  """
  __slots__ = ()

  def replace(self, **kwargs) -> 'Transition':
    """Exposes as namedtuple._replace.

    Usage:
    ```
    new_transition = transition.replace(action_step=())
    ```

    This returns a new transition with an empty `action_step`.

    Args:
      **kwargs: key/value pairs of fields in the transition.

    Returns:
      A new `Transition`.
    """
    return self._replace(**kwargs)

  def __repr__(self):
    return 'Transition(\n' + pprint.pformat(dict(self._asdict())) + ')'


def _create_trajectory(
    observation,
    action,
    policy_info,
    reward,
    discount,
    step_type,
    next_step_type,
    name_scope):
  """Create a Trajectory composed of either Tensors or numpy arrays.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[B]`, `[T]`, or `[B, T]` (optional).
    step_type: `Tensor` or `np.ndarray` of `ts.StepType`,
      shaped `[B]`, `[T]`, or `[B, T]`.
    next_step_type: `Tensor` or `np.ndarray` of `ts.StepType`,
      shaped `[B]`, `[T]`, or `[B, T]`.
    name_scope: Python string, name to use when creating tensors.

  Returns:
    A `Trajectory` instance.
  """
  if nest_utils.has_tensors(
      observation, action, policy_info, reward, discount):
    with tf.name_scope(name_scope):
      discount = tf.identity(discount)
      shape = tf.shape(input=discount)
      make_tensors = lambda struct: tf.nest.map_structure(tf.identity, struct)
      return Trajectory(
          step_type=tf.fill(shape, step_type),
          observation=make_tensors(observation),
          action=make_tensors(action),
          policy_info=make_tensors(policy_info),
          next_step_type=tf.fill(shape, next_step_type),
          reward=make_tensors(reward),
          discount=discount)
  else:
    discount = np.asarray(discount)
    shape = discount.shape
    make_arrays = lambda struct: tf.nest.map_structure(np.asarray, struct)
    return Trajectory(
        step_type=np.full(shape, step_type),
        observation=make_arrays(observation),
        action=make_arrays(action),
        policy_info=make_arrays(policy_info),
        next_step_type=np.full(shape, next_step_type),
        reward=make_arrays(reward),
        discount=discount)


def first(observation: types.NestedSpecTensorOrArray,
          action: types.NestedSpecTensorOrArray,
          policy_info: types.NestedSpecTensorOrArray,
          reward: types.NestedSpecTensorOrArray,
          discount: types.SpecTensorOrArray) -> Trajectory:
  """Create a Trajectory transitioning between StepTypes `FIRST` and `MID`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[B]`, `[T]`, or `[B, T]` (optional).

  Returns:
    A `Trajectory` instance.
  """
  return _create_trajectory(observation,
                            action,
                            policy_info,
                            reward,
                            discount,
                            name_scope='trajectory_first',
                            step_type=ts.StepType.FIRST,
                            next_step_type=ts.StepType.MID)


def mid(observation: types.NestedSpecTensorOrArray,
        action: types.NestedSpecTensorOrArray,
        policy_info: types.NestedSpecTensorOrArray,
        reward: types.NestedSpecTensorOrArray,
        discount: types.SpecTensorOrArray) -> Trajectory:
  """Create a Trajectory transitioning between StepTypes `MID` and `MID`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[B]`, `[T]`, or `[B, T]` (optional).

  Returns:
    A `Trajectory` instance.
  """
  return _create_trajectory(observation,
                            action,
                            policy_info,
                            reward,
                            discount,
                            name_scope='trajectory_mid',
                            step_type=ts.StepType.MID,
                            next_step_type=ts.StepType.MID)


def last(observation: types.NestedSpecTensorOrArray,
         action: types.NestedSpecTensorOrArray,
         policy_info: types.NestedSpecTensorOrArray,
         reward: types.NestedSpecTensorOrArray,
         discount: types.SpecTensorOrArray) -> Trajectory:
  """Create a Trajectory transitioning between StepTypes `MID` and `LAST`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[B]`, `[T]`, or `[B, T]` (optional).

  Returns:
    A `Trajectory` instance.
  """
  return _create_trajectory(observation,
                            action,
                            policy_info,
                            reward,
                            discount,
                            name_scope='trajectory_last',
                            step_type=ts.StepType.MID,
                            next_step_type=ts.StepType.LAST)


def single_step(observation: types.NestedSpecTensorOrArray,
                action: types.NestedSpecTensorOrArray,
                policy_info: types.NestedSpecTensorOrArray,
                reward: types.NestedSpecTensorOrArray,
                discount: types.SpecTensorOrArray) -> Trajectory:
  """Create a Trajectory transitioning between StepTypes `FIRST` and `LAST`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`; shaped
      `[B]`, `[T]`, or `[B, T]` (optional).

  Returns:
    A `Trajectory` instance.
  """
  return _create_trajectory(
      observation,
      action,
      policy_info,
      reward,
      discount,
      name_scope='trajectory_single_step',
      step_type=ts.StepType.FIRST,
      next_step_type=ts.StepType.LAST)


def boundary(observation: types.NestedSpecTensorOrArray,
             action: types.NestedSpecTensorOrArray,
             policy_info: types.NestedSpecTensorOrArray,
             reward: types.NestedSpecTensorOrArray,
             discount: types.SpecTensorOrArray) -> Trajectory:
  """Create a Trajectory transitioning between StepTypes `LAST` and `FIRST`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[B, ...]`, `[T, ...]`, or `[B, T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[B]`, `[T]`, or `[B, T]` (optional).

  Returns:
    A `Trajectory` instance.
  """
  return _create_trajectory(observation,
                            action,
                            policy_info,
                            reward,
                            discount,
                            name_scope='trajectory_boundary',
                            step_type=ts.StepType.LAST,
                            next_step_type=ts.StepType.FIRST)


def _maybe_static_outer_dim(t):
  """Return the left-most dense shape dimension of `t`.

  Args:
    t: A `Tensor` or `CompositeTensor`.

  Returns:
    A python integer or `0-D` scalar tensor with type `int64`.
  """
  assert tf.is_tensor(t), t
  if isinstance(t, tf.SparseTensor):
    static_shape = tf.get_static_value(t.dense_shape)
    if static_shape is not None:
      return static_shape[0]
    else:
      return t.dense_shape[0]
  elif isinstance(t, tf.RaggedTensor):
    outer_dim = tf.compat.dimension_value(t.shape[0])
    return outer_dim if outer_dim is not None else t.nrows()
  else:
    outer_dim = tf.compat.dimension_value(t.shape[0])
    return outer_dim if outer_dim is not None else tf.shape(t)[0]


def from_episode(
    observation: types.NestedSpecTensorOrArray,
    action: types.NestedSpecTensorOrArray,
    policy_info: types.NestedSpecTensorOrArray,
    reward: types.NestedSpecTensorOrArray,
    discount: Optional[types.SpecTensorOrArray] = None) -> Trajectory:
  """Create a Trajectory from tensors representing a single episode.

  If none of the inputs are tensors, then numpy arrays are generated instead.

  If `discount` is not provided, the first entry in `reward` is used to estimate
  `T`:

  ```
  reward_0 = tf.nest.flatten(reward)[0]
  T = shape(reward_0)[0]
  ```

  In this case, a `discount` of all ones having dtype `float32` is generated.

  **NOTE**: all tensors/numpy arrays passed to this function have the same time
  dimension `T`. When the generated trajectory passes through `to_transition`,
  it will only return a `(time_steps, next_time_steps)` pair with `T - 1` in the
  time dimension, which means the reward at step T is dropped. So if the reward
  at step `T` is important, please make sure the episode passed to this function
  contains an additional step.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`; all shaped
      `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`; shaped
      `[T]` (optional).

  Returns:
    An instance of `Trajectory`.
  """
  use_tensors = nest_utils.has_tensors(
      observation, action, policy_info, reward, discount)
  map_structure = functools.partial(
      tf.nest.map_structure, expand_composites=True)
  if use_tensors:
    ones_fn = tf.ones
    float_dtype = tf.float32
    convert_fn = tf.convert_to_tensor
    concat_fn = tf.concat
    maximum_fn = tf.maximum
    fill_fn = tf.fill
    identity_map = lambda struct: map_structure(tf.identity, struct)
  else:
    ones_fn = np.ones
    float_dtype = np.float32
    convert_fn = np.asarray
    concat_fn = np.concatenate
    maximum_fn = np.maximum
    fill_fn = np.full
    identity_map = lambda struct: map_structure(np.asarray, struct)

  def _from_episode(observation, action, policy_info, reward, discount):
    """Implementation of from_episode."""
    if discount is not None:
      time_source = discount
    else:
      time_source = tf.nest.flatten(reward)[0]
    if tf.is_tensor(time_source):
      num_frames = _maybe_static_outer_dim(time_source)
    else:
      num_frames = np.shape(time_source)[0]
    if discount is None:
      discount = ones_fn([num_frames], dtype=float_dtype)

    if not tf.is_tensor(num_frames):

      def check_num_frames(t):
        if tf.is_tensor(t):
          outer_dim = _maybe_static_outer_dim(t)
        else:
          outer_dim = t.shape[0]
        if not tf.is_tensor(outer_dim) and outer_dim != num_frames:
          raise ValueError('Expected first dimension to be {}, '
                           'but saw outer dim: {}'.format(num_frames,
                                                          outer_dim))

      tf.nest.map_structure(
          check_num_frames,
          (observation, action, policy_info, reward, discount),
          expand_composites=False)

    ts_first = convert_fn(ts.StepType.FIRST)
    ts_last = convert_fn(ts.StepType.LAST)
    mid_size = maximum_fn(0, num_frames - 1)
    ts_mid = fill_fn([mid_size], ts.StepType.MID)
    step_type = concat_fn(([ts_first], ts_mid), axis=0)
    next_step_type = concat_fn((ts_mid, [ts_last]), axis=0)

    return Trajectory(
        step_type=step_type,
        observation=identity_map(observation),
        action=identity_map(action),
        policy_info=identity_map(policy_info),
        next_step_type=next_step_type,
        reward=identity_map(reward),
        discount=identity_map(discount))

  if use_tensors:
    with tf.name_scope('from_episode'):
      return _from_episode(observation, action, policy_info, reward, discount)
  else:
    return _from_episode(observation, action, policy_info, reward, discount)


def from_transition(time_step: ts.TimeStep,
                    action_step: policy_step.PolicyStep,
                    next_time_step: ts.TimeStep) -> Trajectory:
  """Returns a `Trajectory` given transitions.

  `from_transition` is used by a driver to convert sequence of transitions into
  a `Trajectory` for efficient storage. Then an agent (e.g.
  `ppo_agent.PPOAgent`) converts it back to transitions by invoking
  `to_transition`.

  Note that this method does not add a time dimension to the Tensors in the
  resulting `Trajectory`. This means that if your transitions don't already
  include a time dimension, the `Trajectory` cannot be passed to
  `agent.train()`.

  Args:
    time_step: A `time_step.TimeStep` representing the first step in a
      transition.
    action_step: A `policy_step.PolicyStep` representing actions corresponding
      to observations from time_step.
    next_time_step: A `time_step.TimeStep` representing the second step in a
      transition.
  """
  return Trajectory(
      step_type=time_step.step_type,
      observation=time_step.observation,
      action=action_step.action,
      policy_info=action_step.info,
      next_step_type=next_time_step.step_type,
      reward=next_time_step.reward,
      discount=next_time_step.discount)


def to_transition(
    trajectory: Trajectory,
    next_trajectory: Optional[Trajectory] = None
) -> Transition:
  """Create a transition from a trajectory or two adjacent trajectories.

  **NOTE** If `next_trajectory` is not provided, tensors of `trajectory` are
  sliced along their *second* (`time`) dimension; for example:

  ```
  time_steps.step_type = trajectory.step_type[:,:-1]
  time_steps.observation = trajectory.observation[:,:-1]
  next_time_steps.observation = trajectory.observation[:,1:]
  next_time_steps. step_type = trajectory. next_step_type[:,:-1]
  next_time_steps.reward = trajectory.reward[:,:-1]
  next_time_steps. discount = trajectory. discount[:,:-1]

  ```
  Notice that reward and discount for time_steps are undefined, therefore filled
  with zero.

  Args:
    trajectory: An instance of `Trajectory`. The tensors in Trajectory must have
      shape `[B, T, ...]` when next_trajectory is `None`.  `discount` is assumed
      to be a scalar float; hence the shape of `trajectory.discount` must
      be `[B, T]`.
    next_trajectory: (optional) An instance of `Trajectory`.

  Returns:
    A tuple `(time_steps, policy_steps, next_time_steps)`.  The `reward` and
    `discount` fields of `time_steps` are filled with zeros because these
    cannot be deduced (please do not use them).

  Raises:
    ValueError: if `discount` rank is not within the range [1, 2].
  """
  _validate_rank(trajectory.discount, min_rank=1, max_rank=2)

  if next_trajectory is not None:
    _validate_rank(next_trajectory.discount, min_rank=1, max_rank=2)

  if next_trajectory is None:
    next_trajectory = tf.nest.map_structure(
        lambda t: composite.slice_from(t, axis=1, start=1), trajectory)
    trajectory = tf.nest.map_structure(
        lambda t: composite.slice_to(t, axis=1, end=-1), trajectory)
  policy_steps = policy_step.PolicyStep(
      action=trajectory.action, state=(), info=trajectory.policy_info)
  # TODO(b/130244652): Consider replacing 0 rewards & discounts with ().
  time_steps = ts.TimeStep(
      trajectory.step_type,
      reward=tf.nest.map_structure(tf.zeros_like, trajectory.reward),  # unknown
      discount=tf.zeros_like(trajectory.discount),  # unknown
      observation=trajectory.observation)
  next_time_steps = ts.TimeStep(
      step_type=trajectory.next_step_type,
      reward=trajectory.reward,
      discount=trajectory.discount,
      observation=next_trajectory.observation)
  return Transition(time_steps, policy_steps, next_time_steps)


def to_n_step_transition(
    trajectory: Trajectory,
    gamma: types.Float
) -> Transition:
  """Create an n-step transition from a trajectory with `T=N + 1` frames.

  **NOTE** Tensors of `trajectory` are sliced along their *second* (`time`)
  dimension, to pull out the appropriate fields for the n-step transitions.

  The output transition's `next_time_step.{reward, discount}` will contain
  N-step discounted reward and discount values calculated as:

  ```
  next_time_step.reward = r_t +
                          g^{1} * d_t * r_{t+1} +
                          g^{2} * d_t * d_{t+1} * r_{t+2} +
                          g^{3} * d_t * d_{t+1} * d_{t+2} * r_{t+3} +
                          ...
                          g^{N-1} * d_t * ... * d_{t+N-2} * r_{t+N-1}
  next_time_step.discount = g^{N-1} * d_t * d_{t+1} * ... * d_{t+N-1}
  ```

  In python notation:

  ```python
  discount = gamma**(N-1) * reduce_prod(trajectory.discount[:, :-1])
  reward = discounted_return(
      rewards=trajectory.reward[:, :-1],
      discounts=gamma * trajectory.discount[:, :-1])
  ```

  When `trajectory.discount[:, :-1]` is an all-ones tensor, this is equivalent
  to:

  ```python
  next_time_step.discount = (
      gamma**(N-1) * tf.ones_like(trajectory.discount[:, 0]))
  next_time_step.reward = (
      sum_{n=0}^{N-1} gamma**n * trajectory.reward[:, n])
  ```

  Args:
    trajectory: An instance of `Trajectory`. The tensors in Trajectory must have
      shape `[B, T, ...]`.  `discount` is assumed to be a scalar float,
      hence the shape of `trajectory.discount` must be `[B, T]`.
    gamma: A floating point scalar; the discount factor.

  Returns:
    An N-step `Transition` where `N = T - 1`.  The reward and discount in
    `time_step.{reward, discount}` are NaN.  The n-step discounted reward
    and final discount are stored in `next_time_step.{reward, discount}`.
    All tensors in the `Transition` have shape `[B, ...]` (no time dimension).

  Raises:
    ValueError: if `discount.shape.rank != 2`.
    ValueError: if `discount.shape[1] < 2`.
  """
  _validate_rank(trajectory.discount, min_rank=2, max_rank=2)

  # Use static values when available, so that we can use XLA when the time
  # dimension is fixed.
  time_dim = (tf.compat.dimension_value(trajectory.discount.shape[1])
              or tf.shape(trajectory.discount)[1])

  static_time_dim = tf.get_static_value(time_dim)
  if static_time_dim in (0, 1):
    raise ValueError(
        'Trajectory frame count must be at least 2, but saw {}.  Shape of '
        'trajectory.discount: {}'.format(static_time_dim,
                                         trajectory.discount.shape))

  n = time_dim - 1

  # Use composite calculations to ensure we properly handle SparseTensor etc in
  # the observations.

  # pylint: disable=g-long-lambda

  # Pull out x[:,0] for x in trajectory
  first_frame = tf.nest.map_structure(
      lambda t: composite.squeeze(
          composite.slice_to(t, axis=1, end=1),
          axis=1),
      trajectory)

  # Pull out x[:,-1] for x in trajectory
  final_frame = tf.nest.map_structure(
      lambda t: composite.squeeze(
          composite.slice_from(t, axis=1, start=-1),
          axis=1),
      trajectory)
  # pylint: enable=g-long-lambda

  # When computing discounted return, we need to throw out the last time
  # index of both reward and discount, which are filled with dummy values
  # to match the dimensions of the observation.
  reward = trajectory.reward[:, :-1]
  discount = trajectory.discount[:, :-1]

  policy_steps = policy_step.PolicyStep(
      action=first_frame.action, state=(), info=first_frame.policy_info)

  discounted_reward = value_ops.discounted_return(
      rewards=reward,
      discounts=gamma * discount,
      time_major=False,
      provide_all_returns=False)

  # NOTE: `final_discount` will have one less discount than `discount`.
  # This is so that when the learner/update uses an additional
  # discount (e.g. gamma) we don't apply it twice.
  final_discount = gamma**(n-1) * tf.math.reduce_prod(discount, axis=1)

  time_steps = ts.TimeStep(
      first_frame.step_type,
      # unknown
      reward=tf.nest.map_structure(
          lambda r: np.nan * tf.ones_like(r), first_frame.reward),
      # unknown
      discount=np.nan * tf.ones_like(first_frame.discount),
      observation=first_frame.observation)
  next_time_steps = ts.TimeStep(
      step_type=final_frame.step_type,
      reward=discounted_reward,
      discount=final_discount,
      observation=final_frame.observation)
  return Transition(time_steps, policy_steps, next_time_steps)


def to_transition_spec(trajectory_spec: Trajectory) -> Transition:
  """Create a transition spec from a trajectory spec.

  Note: since trajectories do not include the policy step's state (except
  in special cases where the policy chooses to store this in the info field),
  the returned `transition.action_spec.state` field will be an empty tuple.

  Args:
    trajectory_spec: An instance of `Trajectory` representing trajectory specs.

  Returns:
    A tuple `(time_steps, policy_steps, next_time_steps)` specs.
  """
  policy_step_spec = policy_step.PolicyStep(
      action=trajectory_spec.action, state=(), info=trajectory_spec.policy_info)
  time_step_spec = ts.TimeStep(
      trajectory_spec.step_type,
      reward=trajectory_spec.reward,
      discount=trajectory_spec.discount,
      observation=trajectory_spec.observation)
  return Transition(time_step_spec, policy_step_spec, time_step_spec)


def _validate_rank(variable, min_rank, max_rank=None):
  """Validates if a variable has the correct rank.

  Args:
    variable: A `tf.Tensor` or `numpy.array`.
    min_rank: An int representing the min expected rank of the variable.
    max_rank: An int representing the max expected rank of the variable.

  Raises:
    ValueError: if variable doesn't have expected rank.
  """
  rank = len(variable.shape)
  if rank < min_rank or rank > max_rank:
    raise ValueError(
        'Expect variable within rank [{},{}], but got rank {}.'.format(
            min_rank, max_rank, rank))


def experience_to_transitions(
    experience: Trajectory, squeeze_time_dim: bool
) -> Transition:
  """Break experience to transitions."""
  transitions = to_transition(experience)

  if squeeze_time_dim:
    transitions = tf.nest.map_structure(lambda x: composite.squeeze(x, 1),
                                        transitions)

  return transitions
