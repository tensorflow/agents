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

"""TimeStep representing a step in the environment."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import pprint
from typing import NamedTuple, Optional

import numpy as np
import tensorflow as tf

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.typing import types


def _as_float32_array(a):
  r = np.asarray(a, dtype=np.float32)
  if np.isnan(np.sum(r)):
    raise ValueError('Received a time_step input that converted to a nan array.'
                     ' Did you accidentally set some input value to None?.\n'
                     'Got:\n{}'.format(a))
  return r


class TimeStep(
    NamedTuple('TimeStep', [('step_type', types.SpecTensorOrArray),
                            ('reward', types.NestedSpecTensorOrArray),
                            ('discount', types.SpecTensorOrArray),
                            ('observation', types.NestedSpecTensorOrArray)])):
  """Returned with every call to `step` and `reset` on an environment.

  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
  NumPy array or a dict or list of arrays), and an associated `reward` and
  `discount`.

  The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
  `TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
  will equal `StepType.MID.

  Attributes:
    step_type: a `Tensor` or array of `StepType` enum values.
    reward: a `Tensor` or array of reward values.
    discount: A discount value in the range `[0, 1]`.
    observation: A NumPy array, or a nested dict, list or tuple of arrays.
  """
  __slots__ = ()

  def is_first(self) -> types.Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.FIRST)
    return np.equal(self.step_type, StepType.FIRST)

  def is_mid(self) -> types.Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.MID)
    return np.equal(self.step_type, StepType.MID)

  def is_last(self) -> types.Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.LAST)
    return np.equal(self.step_type, StepType.LAST)

  def __hash__(self):
    # TODO(b/130243327): Explore performance impact and consider converting
    # dicts in the observation into ordered dicts in __new__ call.
    return hash(tuple(tf.nest.flatten(self)))

  def __repr__(self):
    return 'TimeStep(\n' + pprint.pformat(dict(self._asdict())) + ')'


class StepType(object):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = np.asarray(0, dtype=np.int32)
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = np.asarray(1, dtype=np.int32)
  # Denotes the last `TimeStep` in a sequence.
  LAST = np.asarray(2, dtype=np.int32)

  def __new__(cls, value):
    """Add ability to create StepType constants from a value."""
    if value == cls.FIRST:
      return cls.FIRST
    if value == cls.MID:
      return cls.MID
    if value == cls.LAST:
      return cls.LAST

    raise ValueError('No known conversion for `%r` into a StepType' % value)


def restart(observation: types.NestedTensorOrArray,
            batch_size: Optional[types.Int] = None,
            reward_spec: Optional[types.NestedSpec] = None) -> TimeStep:
  """Returns a `TimeStep` with `step_type` set equal to `StepType.FIRST`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    batch_size: (Optional) A python or tensorflow integer scalar. If not
      provided, the environment will not be considered as a batched
      environment.
    reward_spec: If provided, the reward in the returned `TimeStep` will be
      compatible with the provided `reward_spec`.

  Returns:
    A `TimeStep`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    if batch_size is not None:
      if reward_spec is None:
        reward = np.zeros(batch_size, dtype=np.float32)
      else:
        reward = tf.nest.map_structure(
            lambda r: np.zeros([batch_size] + list(r.shape), dtype=r.dtype),
            reward_spec)
      discount = np.ones(batch_size, dtype=np.float32)
      step_type = np.tile(StepType.FIRST, batch_size)
      return TimeStep(step_type, reward, discount, observation)
    else:
      if reward_spec is None:
        return TimeStep(
            StepType.FIRST,
            _as_float32_array(0.0),
            _as_float32_array(1.0),
            observation)
      else:
        reward = tf.nest.map_structure(
            lambda r: np.zeros(r.shape, dtype=r.dtype), reward_spec)
        return TimeStep(
            StepType.FIRST,
            reward,
            _as_float32_array(1.0),
            observation)

  # TODO(b/130244501): Check leading dimension of first_observation
  # against batch_size if all are known statically.
  shape = _as_multi_dim(batch_size)
  step_type = tf.fill(shape, StepType.FIRST, name='step_type')
  if reward_spec is None:
    reward = tf.fill(shape, _as_float32_array(0.0), name='reward')
  else:
    reward = tf.nest.map_structure(
        # pylint: disable=g-long-lambda
        lambda r: tf.fill(tf.concat([shape, r.shape], axis=-1),
                          _as_float32_array(0.0),
                          name='reward'), reward_spec)
  discount = tf.fill(shape, _as_float32_array(1.0), name='discount')
  return TimeStep(step_type, reward, discount, observation)


def _as_multi_dim(maybe_scalar):
  if maybe_scalar is None:
    shape = ()
  elif tf.is_tensor(maybe_scalar) and maybe_scalar.shape.rank > 0:
    shape = maybe_scalar
  elif np.asarray(maybe_scalar).ndim > 0:
    shape = maybe_scalar
  else:
    shape = (maybe_scalar,)
  return shape


def transition(observation: types.NestedTensorOrArray,
               reward: types.NestedTensorOrArray,
               discount: types.Float = 1.0,
               outer_dims: Optional[types.Shape] = None) -> TimeStep:
  """Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

  For TF transitions, the batch size is inferred from the shape of `reward`.

  If `discount` is a scalar, and `observation` contains Tensors,
  then `discount` will be broadcasted to match `reward.shape`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    discount: (optional) A scalar, or 1D NumPy array, or tensor.
    outer_dims: (optional) If provided, it will be used to determine the
      batch dimensions. If not, the batch dimensions will be inferred by
      reward's shape.

  Returns:
    A `TimeStep`.

  Raises:
    ValueError: If observations are tensors but reward's statically known rank
      is not `0` or `1`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    if outer_dims is not None:
      step_type = np.tile(StepType.MID, outer_dims)
      discount = _as_float32_array(discount)
      return TimeStep(step_type, reward, discount, observation)
    # Infer the batch size.
    reward = tf.nest.map_structure(_as_float32_array, reward)
    first_reward = tf.nest.flatten(reward)[0]
    discount = _as_float32_array(discount)
    if first_reward.shape:
      step_type = np.tile(StepType.MID, first_reward.shape)
    else:
      step_type = StepType.MID
    return TimeStep(step_type, reward, discount, observation)

  # TODO(b/130245199): If reward.shape.rank == 2, and static
  # batch sizes are available for both first_observation and reward,
  # check that these match.
  reward = tf.nest.map_structure(
      # pylint: disable=g-long-lambda
      lambda r: tf.convert_to_tensor(
          value=r, dtype=tf.float32, name='reward'), reward)
  if outer_dims is not None:
    shape = outer_dims
  else:
    first_reward = tf.nest.flatten(reward)[0]
    if first_reward.shape.rank == 0:
      shape = []
    else:
      shape = [tf.compat.dimension_value(first_reward.shape[0]) or
               tf.shape(input=first_reward)[0]]
  step_type = tf.fill(shape, StepType.MID, name='step_type')
  discount = tf.convert_to_tensor(
      value=discount, dtype=tf.float32, name='discount')
  if discount.shape.rank == 0:
    discount = tf.fill(shape, discount, name='discount_fill')
  return TimeStep(step_type, reward, discount, observation)


def termination(observation: types.NestedTensorOrArray,
                reward: types.NestedTensorOrArray,
                outer_dims: Optional[types.Shape] = None) -> TimeStep:
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    outer_dims: (optional) If provided, it will be used to determine the
      batch dimensions. If not, the batch dimensions will be inferred by
      reward's shape.
  Returns:
    A `TimeStep`.

  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    if outer_dims is not None:
      step_type = np.tile(StepType.LAST, outer_dims)
      discount = np.zeros(outer_dims, dtype=np.float32)
      return TimeStep(step_type, reward, discount, observation)

    # Infer the batch size based on reward
    reward = tf.nest.map_structure(_as_float32_array, reward)
    first_reward = tf.nest.flatten(reward)[0]
    if first_reward.shape:
      batch_size = first_reward.shape[0]
      step_type = np.tile(StepType.LAST, batch_size)
      discount = np.zeros(batch_size, dtype=np.float32)
    else:
      step_type = StepType.LAST
      discount = _as_float32_array(0.0)
    return TimeStep(step_type, reward, discount, observation)

  # TODO(b/130245199): If reward.shape.rank == 2, and static
  # batch sizes are available for both first_observation and reward,
  # check that these match.
  reward = tf.nest.map_structure(
      lambda r: tf.convert_to_tensor(r, dtype=tf.float32, name='reward'),
      reward)

  if outer_dims is not None:
    shape = outer_dims
  else:
    first_reward = tf.nest.flatten(reward)[0]
    if first_reward.shape.rank == 0:
      shape = []
    else:
      shape = [tf.compat.dimension_value(first_reward.shape[0]) or
               tf.shape(input=first_reward)[0]]
  step_type = tf.fill(shape, StepType.LAST, name='step_type')
  discount = tf.fill(shape, _as_float32_array(0.0), name='discount')
  return TimeStep(step_type, reward, discount, observation)


# TODO(b/152907905): Update this function.
def truncation(observation: types.NestedTensorOrArray,
               reward: types.NestedTensorOrArray,
               discount: types.Float = 1.0,
               outer_dims: Optional[types.Shape] = None)  -> TimeStep:
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

  If `discount` is a scalar, and `observation` contains Tensors,
  then `discount` will be broadcasted to match the outer dimensions.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    discount: (optional) A scalar, or 1D NumPy array, or tensor.
    outer_dims: (optional) If provided, it will be used to determine the
      batch dimensions. If not, the batch dimensions will be inferred by
      reward's shape.

  Returns:
    A `TimeStep`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    if outer_dims is not None:
      step_type = np.tile(StepType.LAST, outer_dims)
      discount = _as_float32_array(discount)
      return TimeStep(step_type, reward, discount, observation)
    # Infer the batch size.
    reward = tf.nest.map_structure(_as_float32_array, reward)
    first_reward = tf.nest.flatten(reward)[0]
    discount = _as_float32_array(discount)
    if first_reward.shape:
      step_type = np.tile(StepType.LAST, first_reward.shape)
    else:
      step_type = StepType.LAST
    return TimeStep(step_type, reward, discount, observation)

  reward = tf.nest.map_structure(
      lambda r: tf.convert_to_tensor(value=r, name='reward'),
      reward)
  if outer_dims is not None:
    shape = outer_dims
  else:
    first_reward = tf.nest.flatten(reward)[0]
    if first_reward.shape.rank == 0:
      shape = []
    else:
      shape = [tf.compat.dimension_value(first_reward.shape[0]) or
               tf.shape(input=first_reward)[0]]
  step_type = tf.fill(shape, StepType.LAST, name='step_type')
  discount = tf.convert_to_tensor(
      value=discount, dtype=tf.float32, name='discount')
  if discount.shape.rank == 0:
    discount = tf.fill(shape, discount, name='discount_fill')
  return TimeStep(step_type, reward, discount, observation)


def time_step_spec(
    observation_spec: Optional[types.NestedSpec] = None,
    reward_spec: Optional[types.NestedSpec] = None) -> TimeStep:
  """Returns a `TimeStep` spec given the observation_spec.

  Args:
    observation_spec: A nest of `tf.TypeSpec` or `ArraySpec` objects.
    reward_spec: (Optional) A nest of `tf.TypeSpec` or `ArraySpec` objects.
      Default - a scalar float32 of the same type (Tensor or Array) as
      `observation_spec`.

  Returns:
    A `TimeStep` with the same types (`TypeSpec` or `ArraySpec`) as
    the first entry in `observation_spec`.

  Raises:
    TypeError: If observation and reward specs aren't both either tensor type
      specs or array type specs.
  """
  if observation_spec is None:
    return TimeStep(step_type=(), reward=(), discount=(), observation=())

  first_observation_spec = tf.nest.flatten(observation_spec)[0]
  if reward_spec is not None:
    first_reward_spec = tf.nest.flatten(reward_spec)[0]
    if (isinstance(first_reward_spec, tf.TypeSpec)
        != isinstance(first_observation_spec, tf.TypeSpec)):
      raise TypeError(
          'Expected observation and reward specs to both be either tensor or '
          'array specs, but saw spec values {} vs. {}'
          .format(first_observation_spec, first_reward_spec))
  if isinstance(first_observation_spec, tf.TypeSpec):
    return TimeStep(
        step_type=tensor_spec.TensorSpec([], tf.int32, name='step_type'),
        reward=reward_spec or tf.TensorSpec([], tf.float32, name='reward'),
        discount=tensor_spec.BoundedTensorSpec(
            [], tf.float32, minimum=0.0, maximum=1.0, name='discount'),
        observation=observation_spec)
  return TimeStep(
      step_type=array_spec.ArraySpec([], np.int32, name='step_type'),
      reward=reward_spec or array_spec.ArraySpec([], np.float32, name='reward'),
      discount=array_spec.BoundedArraySpec(
          [], np.float32, minimum=0.0, maximum=1.0, name='discount'),
      observation=observation_spec)
