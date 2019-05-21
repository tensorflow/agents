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

"""Wrapper for PyEnvironments into TFEnvironments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import threading

import gin
import tensorflow as tf

from tf_agents.environments import batched_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
# TODO(b/123022201): Use tf.autograph instead.
from tensorflow.python.autograph.impl import api as autograph  # pylint:disable=g-direct-tensorflow-import  # TF internal
from tensorflow.python.framework import tensor_shape  # pylint:disable=g-direct-tensorflow-import  # TF internal


@contextlib.contextmanager
def _check_not_called_concurrently(lock):
  """Checks the returned context is not executed concurrently with any other."""
  if not lock.acquire(False):  # Non-blocking.
    raise RuntimeError(
        'Detected concurrent execution of TFPyEnvironment ops. Make sure the '
        'appropriate step_state is passed to step().')
  try:
    yield
  finally:
    lock.release()


@gin.configurable
class TFPyEnvironment(tf_environment.TFEnvironment):
  """Exposes a Python environment as an in-graph TF environment.

  This class supports Python environments that return nests of arrays as
  observations and accept nests of arrays as actions. The nest structure is
  reflected in the in-graph environment's observation and action structure.

  Implementation notes:

  * Since `tf.py_func` deals in lists of tensors, this class has some additional
    `tf.nest.flatten` and `tf.nest.pack_structure_as` calls.

  * This class currently cast rewards and discount to float32.
  """

  def __init__(self, environment, check_dims=False):
    """Initializes a new `TFPyEnvironment`.

    Args:
      environment: Environment to interact with, implementing
        `py_environment.PyEnvironment`.
      check_dims: Whether should check batch dimensions of actions in `step`.

    Raises:
      TypeError: If `environment` is not a subclass of
      `py_environment.PyEnvironment`.
    """
    if not isinstance(environment, py_environment.PyEnvironment):
      raise TypeError(
          'Environment should implement py_environment.PyEnvironment')

    if not environment.batched:
      environment = batched_py_environment.BatchedPyEnvironment([environment])
    self._env = environment
    self._check_dims = check_dims

    observation_spec = tensor_spec.from_spec(self._env.observation_spec())
    action_spec = tensor_spec.from_spec(self._env.action_spec())
    time_step_spec = ts.time_step_spec(observation_spec)
    batch_size = self._env.batch_size if self._env.batch_size else 1
    super(TFPyEnvironment, self).__init__(time_step_spec,
                                          action_spec,
                                          batch_size)

    # Gather all the dtypes of the elements in time_step.
    self._time_step_dtypes = [
        s.dtype for s in tf.nest.flatten(self.time_step_spec())
    ]

    self._time_step = None
    self._lock = threading.Lock()

  @property
  def pyenv(self):
    """Returns the underlying Python environment."""
    return self._env

  # TODO(b/123585179): Simplify this using py_environment.current_time_step().
  # There currently is a bug causing py_function to resolve variables
  # incorrectly when used inside autograph code. This decorator tells autograph
  # to call it directly instead of converting it when called from other
  # autograph-converted functions.
  # TODO(b/123600776): Remove override.
  @autograph.do_not_convert()
  def _current_time_step(self):
    """Returns the current ts.TimeStep.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          timestep.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

    def _current_time_step_py():
      with _check_not_called_concurrently(self._lock):
        if self._time_step is None:
          self._time_step = self._env.reset()
        return tf.nest.flatten(self._time_step)

    with tf.name_scope('current_time_step'):
      outputs = tf.numpy_function(
          _current_time_step_py,
          [],  # No inputs.
          self._time_step_dtypes,
          name='current_time_step_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]
      return self._set_names_and_shapes(step_type, reward, discount,
                                        *flat_observations)

  # Make sure this is called without conversion from tf.function.
  # TODO(b/123600776): Remove override.
  @autograph.do_not_convert()
  def _reset(self):
    """Returns the current `TimeStep` after resetting the environment.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          timestep.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

    def _reset_py():
      with _check_not_called_concurrently(self._lock):
        self._time_step = self._env.reset()

    with tf.name_scope('reset'):
      reset_op = tf.numpy_function(
          _reset_py,
          [],  # No inputs.
          [],
          name='reset_py_func')
      with tf.control_dependencies([reset_op]):
        return self.current_time_step()

  # Make sure this is called without conversion from tf.function.
  # TODO(b/123600776): Remove override.
  @autograph.do_not_convert()
  def _step(self, actions):
    """Returns a TensorFlow op to step the environment.

    Args:
      actions: A Tensor, or a nested dict, list or tuple of Tensors
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` tuple of:
        step_type: A scalar int32 tensor representing the `StepType` value.
        reward: A scalar float32 tensor representing the reward at this
          time_step.
        discount: A scalar float32 tensor representing the discount [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.

    Raises:
      ValueError: If any of the actions are scalars or their major axis is known
      and is not equal to `self.batch_size`.
    """

    def _step_py(*flattened_actions):
      with _check_not_called_concurrently(self._lock):
        packed = tf.nest.pack_sequence_as(
            structure=self.action_spec(), flat_sequence=flattened_actions)
        self._time_step = self._env.step(packed)
        return tf.nest.flatten(self._time_step)

    with tf.name_scope('step'):
      flat_actions = [tf.identity(x) for x in tf.nest.flatten(actions)]
      if self._check_dims:
        for action in flat_actions:
          dim_value = tensor_shape.dimension_value(action.shape[0])
          if (action.shape.ndims == 0 or
              (dim_value is not None and dim_value != self.batch_size)):
            raise ValueError(
                'Expected actions whose major dimension is batch_size (%d), '
                'but saw action with shape %s:\n   %s' %
                (self.batch_size, action.shape, action))
      outputs = tf.numpy_function(
          _step_py, flat_actions, self._time_step_dtypes, name='step_py_func')
      step_type, reward, discount = outputs[0:3]
      flat_observations = outputs[3:]

      return self._set_names_and_shapes(step_type, reward, discount,
                                        *flat_observations)

  def _set_names_and_shapes(self, step_type, reward, discount,
                            *flat_observations):
    """Returns a `TimeStep` namedtuple."""
    step_type = tf.identity(step_type, name='step_type')
    reward = tf.identity(reward, name='reward')
    discount = tf.identity(discount, name='discount')
    batch_shape = () if not self.batched else (self.batch_size,)
    batch_shape = tf.TensorShape(batch_shape)
    if not tf.executing_eagerly():
      # Shapes are not required in eager mode.
      reward.set_shape(batch_shape)
      step_type.set_shape(batch_shape)
      discount.set_shape(batch_shape)
    # Give each tensor a meaningful name and set the static shape.
    named_observations = []
    for obs, spec in zip(flat_observations,
                         tf.nest.flatten(self.observation_spec())):
      named_observation = tf.identity(obs, name=spec.name)
      if not tf.executing_eagerly():
        named_observation.set_shape(batch_shape.concatenate(spec.shape))
      named_observations.append(named_observation)

    observations = tf.nest.pack_sequence_as(self.observation_spec(),
                                            named_observations)

    return ts.TimeStep(step_type, reward, discount, observations)
