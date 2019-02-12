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

"""Trajectory containing time_step transition information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections

import numpy as np
import tensorflow as tf

from tf_agents.environments import time_step as ts
from tf_agents.policies import policy_step
from tf_agents.utils import nest_utils


class Trajectory(
    collections.namedtuple('Trajectory', [
        'step_type',
        'observation',
        'action',
        'policy_info',
        'next_step_type',
        'reward',
        'discount',
    ])):
  """A tuple that represents a trajectory.

  A `Trajectory` is a sequence of aligned time steps. It captures the
  observation, step_type from current time step with the computed action
  and policy_info. Discount, reward and next_step_type come from the next
  time step.

  Attributes:
    step_type: A `StepType`.
    observation: An array (tensor), or a nested dict, list or tuple of arrays
      (tensors) that represents the observation.
    action: An array/a tensor, or a nested dict, list or tuple of actions. This
      represents action generated according to the observation.
    policy_info: A namedtuple that contains auxiliary information related to the
      action. Note that this does not include the policy/RNN state which was
      used to generate the action.
    next_step_type: The `StepType` of the next time step.
    reward: A scalar representing the reward of performing the action in an
      environment.
    discount: A scalar that representing the discount factor to multiply with
      future rewards.
  """
  __slots__ = ()

  def is_first(self):
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, ts.StepType.FIRST)
    return self.step_type == ts.StepType.FIRST

  def is_mid(self):
    if tf.is_tensor(self.step_type):
      return tf.logical_and(
          tf.equal(self.step_type, ts.StepType.MID),
          tf.equal(self.next_step_type, ts.StepType.MID))
    return (self.step_type == ts.StepType.MID) & (
        self.next_step_type == ts.StepType.MID)

  def is_last(self):
    if tf.is_tensor(self.next_step_type):
      return tf.equal(self.next_step_type, ts.StepType.LAST)
    return self.next_step_type == ts.StepType.LAST

  def is_boundary(self):
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, ts.StepType.LAST)
    return self.step_type == ts.StepType.LAST

  def replace(self, **kwargs):
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


def _create_trajectory(
    observation,
    action,
    policy_info,
    reward,
    discount,
    name_scope,
    step_type,
    next_step_type):
  """Create a Trajectory composed of either Tensors or numpy arrays.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[T]` (optional).
    name_scope: Python string.
    step_type: `Tensor` or `np.ndarray` of `ts.StepType` shaped `[T]`.
    next_step_type: `Tensor` or `np.ndarray` of `ts.StepType` shaped `[T]`.

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


def first(observation, action, policy_info, reward, discount):
  """Create a Trajectory transitioning between StepTypes `FIRST` and `MID`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[T]` (optional).

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


def mid(observation, action, policy_info, reward, discount):
  """Create a Trajectory transitioning between StepTypes `MID` and `MID`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[T]` (optional).

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


def last(observation, action, policy_info, reward, discount):
  """Create a Trajectory transitioning between StepTypes `MID` and `LAST`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[T]` (optional).

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


def boundary(observation, action, policy_info, reward, discount):
  """Create a Trajectory transitioning between StepTypes `LAST` and `FIRST`.

  All inputs may be batched.

  The input `discount` is used to infer the outer shape of the inputs,
  as it is always expected to be a singleton array with scalar inner shape.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[T]` (optional).

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


def from_episode(observation, action, policy_info, reward, discount=None):
  """Create a Trajectory from tensors representing a single episode.

  If none of the inputs are tensors, then numpy arrays are generated instead.

  If `discount` is not provided, the first entry in `reward` is used to estimate
  `T`:

  ```
  reward_0 = tf.nest.flatten(reward)[0]
  T = shape(reward_0)[0]
  ```

  In this case, a `discount` of all ones having dtype `float32` is generated.

  Args:
    observation: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    action: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    policy_info: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    reward: (possibly nested tuple of) `Tensor` or `np.ndarray`;
      all shaped `[T, ...]`.
    discount: A floating point vector `Tensor` or `np.ndarray`;
      shaped `[T]` (optional).

  Returns:
    An instance of `Trajectory`.
  """
  use_tensors = nest_utils.has_tensors(
      observation, action, policy_info, reward, discount)
  if use_tensors:
    ones_fn = tf.ones
    float_dtype = tf.float32
    convert_fn = tf.convert_to_tensor
    concat_fn = tf.concat
    maximum_fn = tf.maximum
    fill_fn = tf.fill
    identity_map = lambda struct: tf.nest.map_structure(tf.identity, struct)
  else:
    ones_fn = np.ones
    float_dtype = np.float32
    convert_fn = np.asarray
    concat_fn = np.concatenate
    maximum_fn = np.maximum
    fill_fn = np.full
    identity_map = lambda struct: tf.nest.map_structure(np.asarray, struct)

  def _from_episode(observation, action, policy_info, reward, discount):
    """Implementation of from_episode."""
    if discount is not None:
      time_source = discount
    else:
      time_source = tf.nest.flatten(reward)[0]
    if tf.is_tensor(time_source):
      num_frames = (
          tf.compat.dimension_value(time_source.shape[0]) or
          tf.shape(input=time_source)[0])
    else:
      num_frames = np.shape(time_source)[0]
    if discount is None:
      discount = ones_fn([num_frames], dtype=float_dtype)

    if not tf.is_tensor(num_frames):

      def check_num_frames(t):
        if t.shape[0] is not None and t.shape[0] != num_frames:
          raise ValueError('Expected first dimension to be {}, '
                           'but saw value: {}'.format(num_frames, t))

      tf.nest.map_structure(
          check_num_frames,
          (observation, action, policy_info, reward, discount))

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


def from_transition(time_step, action_step, next_time_step):
  return Trajectory(
      step_type=time_step.step_type,
      observation=time_step.observation,
      action=action_step.action,
      policy_info=action_step.info,
      next_step_type=next_time_step.step_type,
      reward=next_time_step.reward,
      discount=next_time_step.discount)


def to_transition(trajectory, next_trajectory=None):
  """Create a transition from a trajectory or two adjacent trajectories.

  **NOTE** If `next_trajectory` is not provided, tensors of `trajectory` are
  sliced along their *second* (`time`) dimension; for example:

  ```
  time_steps.observation = trajectory.observation[:, :-1]
  next_time_steps.observation = trajectory.observation[:, 1:]
  ```

  Args:
    trajectory: An instance of `Trajectory`.
    next_trajectory: (optional) An instance of `Trajectory`.

  Returns:
    A tuple `(time_steps, policy_steps, next_time_steps)`.  The `reward` and
    `discount` fields of `time_steps` are filled with zeros because these
    cannot be deduced.
  """
  if next_trajectory is None:
    next_trajectory = tf.nest.map_structure(lambda x: x[:, 1:], trajectory)
    trajectory = tf.nest.map_structure(lambda x: x[:, :-1], trajectory)
  policy_steps = policy_step.PolicyStep(
      trajectory.action, (), trajectory.policy_info)
  # TODO(kbanoop): Consider replacing 0 rewards & discounts with ().
  time_steps = ts.TimeStep(
      trajectory.step_type,
      reward=tf.nest.map_structure(tf.zeros_like, trajectory.reward),  # unknown
      discount=tf.zeros_like(trajectory.discount),  # unknown
      observation=trajectory.observation)
  next_time_steps = ts.TimeStep(trajectory.next_step_type,
                                trajectory.reward,
                                trajectory.discount,
                                next_trajectory.observation)
  return [time_steps, policy_steps, next_time_steps]
