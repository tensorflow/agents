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

"""TimeStep representing a step in the environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import numpy as np

import tensorflow as tf

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

_as_float32_array = functools.partial(np.asarray, dtype=np.float32)


class TimeStep(
    collections.namedtuple('TimeStep',
                           ['step_type', 'reward', 'discount', 'observation'])):
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

  def is_first(self):
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.FIRST)
    return np.equal(self.step_type, StepType.FIRST)

  def is_mid(self):
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.MID)
    return np.equal(self.step_type, StepType.MID)

  def is_last(self):
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.LAST)
    return np.equal(self.step_type, StepType.LAST)

  def __hash__(self):
    # TODO(b/130243327): Explore performance impact and consider converting
    # dicts in the observation into ordered dicts in __new__ call.
    return hash(tuple(tf.nest.flatten(self)))


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


def restart(observation, batch_size=None):
  """Returns a `TimeStep` with `step_type` set equal to `StepType.FIRST`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    batch_size: (Optional) A python or tensorflow integer scalar.

  Returns:
    A `TimeStep`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    if batch_size is not None:
      reward = np.zeros(batch_size, dtype=np.float32)
      discount = np.ones(batch_size, dtype=np.float32)
      step_type = np.tile(StepType.FIRST, batch_size)
      return TimeStep(step_type, reward, discount, observation)
    else:
      return TimeStep(
          StepType.FIRST,
          _as_float32_array(0.0),
          _as_float32_array(1.0),
          observation,
      )

  # TODO(b/130244501): Check leading dimension of first_observation
  # against batch_size if all are known statically.
  shape = (batch_size,) if batch_size is not None else ()
  step_type = tf.fill(shape, StepType.FIRST, name='step_type')
  reward = tf.fill(shape, _as_float32_array(0.0), name='reward')
  discount = tf.fill(shape, _as_float32_array(1.0), name='discount')
  return TimeStep(step_type, reward, discount, observation)


def transition(observation, reward, discount=1.0):
  """Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

  For TF transitions, the batch size is inferred from the shape of `reward`.

  If `discount` is a scalar, and `observation` contains Tensors,
  then `discount` will be broadcasted to match `reward.shape`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A scalar, or 1D NumPy array, or tensor.
    discount: (optional) A scalar, or 1D NumPy array, or tensor.

  Returns:
    A `TimeStep`.

  Raises:
    ValueError: If observations are tensors but reward's statically known rank
      is not `0` or `1`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    reward = _as_float32_array(reward)
    discount = _as_float32_array(discount)
    if reward.shape:
      step_type = np.tile(StepType.MID, reward.shape)
    else:
      step_type = StepType.MID
    return TimeStep(step_type, reward, discount, observation)

  # TODO(b/130245199): If reward.shape.ndims == 2, and static
  # batch sizes are available for both first_observation and reward,
  # check that these match.
  reward = tf.convert_to_tensor(value=reward, dtype=tf.float32, name='reward')
  if reward.shape.ndims is None or reward.shape.ndims > 1:
    raise ValueError('Expected reward to be a scalar or vector; saw shape: %s' %
                     reward.shape)
  if reward.shape.ndims == 0:
    shape = []
  else:
    first_observation.shape[:1].assert_is_compatible_with(reward.shape)
    shape = [
        tf.compat.dimension_value(reward.shape[0]) or tf.shape(input=reward)[0]
    ]
  step_type = tf.fill(shape, StepType.MID, name='step_type')
  discount = tf.convert_to_tensor(
      value=discount, dtype=tf.float32, name='discount')

  if discount.shape.ndims == 0:
    discount = tf.fill(shape, discount, name='discount_fill')
  else:
    reward.shape.assert_is_compatible_with(discount.shape)
  return TimeStep(step_type, reward, discount, observation)


def termination(observation, reward):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A scalar, or 1D NumPy array, or tensor.

  Returns:
    A `TimeStep`.

  Raises:
    ValueError: If observations are tensors but reward's statically known rank
      is not `0` or `1`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    reward = _as_float32_array(reward)
    if reward.shape:
      step_type = np.tile(StepType.LAST, reward.shape)
      discount = np.zeros_like(reward, dtype=np.float32)
      return TimeStep(step_type, reward, discount, observation)
    else:
      return TimeStep(StepType.LAST, reward, _as_float32_array(0.0),
                      observation)

  # TODO(b/130245199): If reward.shape.ndims == 2, and static
  # batch sizes are available for both first_observation and reward,
  # check that these match.
  reward = tf.convert_to_tensor(value=reward, dtype=tf.float32, name='reward')
  if reward.shape.ndims is None or reward.shape.ndims > 1:
    raise ValueError('Expected reward to be a scalar or vector; saw shape: %s' %
                     reward.shape)
  if reward.shape.ndims == 0:
    shape = []
  else:
    first_observation.shape[:1].assert_is_compatible_with(reward.shape)
    shape = [
        tf.compat.dimension_value(reward.shape[0]) or tf.shape(input=reward)[0]
    ]
  step_type = tf.fill(shape, StepType.LAST, name='step_type')
  discount = tf.fill(shape, _as_float32_array(0.0), name='discount')
  return TimeStep(step_type, reward, discount, observation)


def truncation(observation, reward, discount=1.0):
  """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

  If `discount` is a scalar, and `observation` contains Tensors,
  then `discount` will be broadcasted to match `reward.shape`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A scalar, or 1D NumPy array, or tensor.
    discount: (optional) A scalar, or 1D NumPy array, or tensor.

  Returns:
    A `TimeStep`.

  Raises:
    ValueError: If observations are tensors but reward's statically known rank
      is not `0` or `1`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    reward = _as_float32_array(reward)
    discount = _as_float32_array(discount)
    if reward.shape:
      step_type = np.tile(StepType.LAST, reward.shape)
    else:
      step_type = StepType.LAST
    return TimeStep(step_type, reward, discount, observation)

  reward = tf.convert_to_tensor(value=reward, dtype=tf.float32, name='reward')
  if reward.shape.ndims is None or reward.shape.ndims > 1:
    raise ValueError('Expected reward to be a scalar or vector; saw shape: %s' %
                     reward.shape)
  if reward.shape.ndims == 0:
    shape = []
  else:
    first_observation.shape[:1].assert_is_compatible_with(reward.shape)
    shape = [
        tf.compat.dimension_value(reward.shape[0]) or tf.shape(input=reward)[0]
    ]
  step_type = tf.fill(shape, StepType.LAST, name='step_type')
  discount = tf.convert_to_tensor(
      value=discount, dtype=tf.float32, name='discount')
  if discount.shape.ndims == 0:
    discount = tf.fill(shape, discount, name='discount_fill')
  else:
    reward.shape.assert_is_compatible_with(discount.shape)
  return TimeStep(step_type, reward, discount, observation)


def time_step_spec(observation_spec=None):
  """Returns a `TimeStep` spec given the observation_spec."""
  if observation_spec is None:
    return TimeStep(step_type=(), reward=(), discount=(), observation=())

  first_observation_spec = tf.nest.flatten(observation_spec)[0]
  if isinstance(first_observation_spec,
                (tensor_spec.TensorSpec, tensor_spec.BoundedTensorSpec)):
    return TimeStep(
        step_type=tensor_spec.TensorSpec([], tf.int32, name='step_type'),
        reward=tensor_spec.TensorSpec([], tf.float32, name='reward'),
        discount=tensor_spec.BoundedTensorSpec(
            [], tf.float32, minimum=0.0, maximum=1.0, name='discount'),
        observation=observation_spec)
  return TimeStep(
      step_type=array_spec.ArraySpec([], np.int32, name='step_type'),
      reward=array_spec.ArraySpec([], np.float32, name='reward'),
      discount=array_spec.BoundedArraySpec(
          [], np.float32, minimum=0.0, maximum=1.0, name='discount'),
      observation=observation_spec)
