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

"""Runs multiple environments in parallel processes and steps them in batch."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import atexit
import sys
import traceback
from typing import Any, Callable, Sequence, Text, Union

from absl import logging

import cloudpickle
import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import py_environment
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


# Worker polling period in seconds.
_POLLING_PERIOD = 0.1

EnvConstructor = Callable[[], py_environment.PyEnvironment]
Promise = Callable[[], Any]


@gin.configurable
class ParallelPyEnvironment(py_environment.PyEnvironment):
  """Batch together environments and simulate them in external processes.

  The environments are created in external processes by calling the provided
  callables. This can be an environment class, or a function creating the
  environment and potentially wrapping it. The returned environment should not
  access global variables.
  """

  def __init__(self,
               env_constructors: Sequence[EnvConstructor],
               start_serially: bool = True,
               blocking: bool = False,
               flatten: bool = False):
    """Batch together environments and simulate them in external processes.

    The environments can be different but must use the same action and
    observation specs.

    Args:
      env_constructors: List of callables that create environments.
      start_serially: Whether to start environments serially or in parallel.
      blocking: Whether to step environments one after another.
      flatten: Boolean, whether to use flatten action and time_steps during
        communication to reduce overhead.

    Raises:
      ValueError: If the action or observation specs don't match.
    """
    super(ParallelPyEnvironment, self).__init__()
    if any([not callable(ctor) for ctor in env_constructors]):
      raise TypeError(
          'Found non-callable `env_constructors` in `ParallelPyEnvironment` '
          '__init__ call. Did you accidentally pass in environment instances '
          'instead of constructors? Got: {}'.format(env_constructors))
    self._envs = [ProcessPyEnvironment(ctor, flatten=flatten)
                  for ctor in env_constructors]
    self._num_envs = len(env_constructors)
    self._blocking = blocking
    self._start_serially = start_serially
    self.start()
    self._action_spec = self._envs[0].action_spec()
    self._observation_spec = self._envs[0].observation_spec()
    self._time_step_spec = self._envs[0].time_step_spec()
    self._parallel_execution = True
    if any(env.action_spec() != self._action_spec for env in self._envs):
      raise ValueError('All environments must have the same action spec.')
    if any(env.time_step_spec() != self._time_step_spec for env in self._envs):
      raise ValueError('All environments must have the same time_step_spec.')
    self._flatten = flatten

  def start(self) -> None:
    logging.info('Spawning all processes.')
    for env in self._envs:
      env.start(wait_to_start=self._start_serially)
    if not self._start_serially:
      logging.info('Waiting for all processes to start.')
      for env in self._envs:
        env.wait_start()
    logging.info('All processes started.')

  @property
  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> int:
    return self._num_envs

  @property
  def envs(self):
    return self._envs

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def time_step_spec(self)  -> ts.TimeStep:
    return self._time_step_spec

  def _reset(self):
    """Reset all environments and combine the resulting observation.

    Returns:
      Time step with batch dimension.
    """
    time_steps = [env.reset(self._blocking) for env in self._envs]
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    return self._stack_time_steps(time_steps)

  def _step(self, actions):
    """Forward a batch of actions to the wrapped environments.

    Args:
      actions: Batched action, possibly nested, to apply to the environment.

    Raises:
      ValueError: Invalid actions.

    Returns:
      Batch of observations, rewards, and done flags.
    """
    time_steps = [
        env.step(action, self._blocking)
        for env, action in zip(self._envs, self._unstack_actions(actions))]
    # When blocking is False we get promises that need to be called.
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    return self._stack_time_steps(time_steps)

  def close(self) -> None:
    """Close all external process."""
    logging.info('Closing all processes.')
    for env in self._envs:
      env.close()
    logging.info('All processes closed.')

  def _stack_time_steps(self, time_steps):
    """Given a list of TimeStep, combine to one with a batch dimension."""
    if self._flatten:
      return nest_utils.fast_map_structure_flatten(
          lambda *arrays: np.stack(arrays), self._time_step_spec, *time_steps)
    else:
      return nest_utils.fast_map_structure(
          lambda *arrays: np.stack(arrays), *time_steps)

  def _unstack_actions(self, batched_actions):
    """Returns a list of actions from potentially nested batch of actions."""
    flattened_actions = tf.nest.flatten(batched_actions)
    if self._flatten:
      unstacked_actions = zip(*flattened_actions)
    else:
      unstacked_actions = [
          tf.nest.pack_sequence_as(batched_actions, actions)
          for actions in zip(*flattened_actions)
      ]
    return unstacked_actions

  def seed(self, seeds: Sequence[types.Seed]) -> Sequence[Any]:
    """Seeds the parallel environments."""
    if len(seeds) != len(self._envs):
      raise ValueError(
          'Number of seeds should match the number of parallel_envs.')

    promises = [env.call('seed', seed) for seed, env in zip(seeds, self._envs)]
    # Block until all envs are seeded.
    return [promise() for promise in promises]

  def render(self, mode: Text = 'rgb_array') -> types.NestedArray:
    """Renders the environment.

    Args:
      mode: Rendering mode. Currently only 'rgb_array' is supported because
        this is a batched environment.

    Returns:
      An ndarray of shape [batch_size, width, height, 3] denoting RGB images
      (for mode=`rgb_array`).
    Raises:
      NotImplementedError: If the environment does not support rendering,
        or any other mode than `rgb_array` is given.
    """
    if mode != 'rgb_array':
      raise NotImplementedError('Only rgb_array rendering mode is supported. '
                                'Got %s' % mode)
    imgs = [env.render(mode, blocking=self._blocking) for env in self._envs]
    if not self._blocking:
      imgs = [promise() for promise in imgs]
    return nest_utils.stack_nested_arrays(imgs)


class ProcessPyEnvironment(object):
  """Step a single env in a separate process for lock free paralellism."""

  # Message types for communication via the pipe.
  _READY = 1
  _ACCESS = 2
  _CALL = 3
  _RESULT = 4
  _EXCEPTION = 5
  _CLOSE = 6

  def __init__(self, env_constructor: EnvConstructor, flatten: bool = False):
    """Step environment in a separate process for lock free paralellism.

    The environment is created in an external process by calling the provided
    callable. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The returned environment should
    not access global variables.

    Args:
      env_constructor: Callable that creates and returns a Python environment.
      flatten: Boolean, whether to assume flattened actions and time_steps
        during communication to avoid overhead.

    Attributes:
      observation_spec: The cached observation spec of the environment.
      action_spec: The cached action spec of the environment.
      time_step_spec: The cached time step spec of the environment.
    """
    # NOTE(ebrevdo): multiprocessing uses the standard py3 pickler which does
    # not support anonymous lambdas.  Folks usually pass anonymous lambdas as
    # env constructors.  Here we work around this by manually pickling
    # the constructor using cloudpickle; which supports these.  In the
    # new process, we'll unpickle this constructor and run it.
    self._pickled_env_constructor = cloudpickle.dumps(env_constructor)
    self._flatten = flatten
    self._observation_spec = None
    self._action_spec = None
    self._time_step_spec = None

  def start(self, wait_to_start: bool = True) -> None:
    """Start the process.

    Args:
      wait_to_start: Whether the call should wait for an env initialization.
    """
    mp_context = multiprocessing.get_context()
    self._conn, conn = mp_context.Pipe()
    self._process = mp_context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    if wait_to_start:
      self.wait_start()

  def wait_start(self) -> None:
    """Wait for the started process to finish initialization."""
    result = self._conn.recv()
    if isinstance(result, Exception):
      self._conn.close()
      self._process.join(5)
      raise result
    assert result == self._READY, result

  def observation_spec(self) -> types.NestedArraySpec:
    if not self._observation_spec:
      self._observation_spec = self.call('observation_spec')()
    return self._observation_spec

  def action_spec(self) -> types.NestedArraySpec:
    if not self._action_spec:
      self._action_spec = self.call('action_spec')()
    return self._action_spec

  def time_step_spec(self) -> ts.TimeStep:
    if not self._time_step_spec:
      self._time_step_spec = self.call('time_step_spec')()
    return self._time_step_spec

  def __getattr__(self, name: Text) -> Any:
    """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    This method is only called if the attribute is not found in the dictionary
    of `ParallelPyEnvironment`'s definition.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
    # Private properties are always accessed on this object, not in the
    # wrapped object in another process.  This includes properties used
    # for pickling (incl. __getstate__, __setstate__, _conn, _ACCESS, _receive),
    # as well as private properties and methods created and used by subclasses
    # of this class.  Allowing arbitrary private attributes to be requested
    # from the other process can lead to deadlocks.
    if name.startswith('_'):
      return super(ProcessPyEnvironment, self).__getattribute__(name)

    # All other requests get sent to the worker.
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name: Text, *args, **kwargs) -> Promise:
    """Asynchronously call a method of the external environment.

    Args:
      name: Name of the method to call.
      *args: Positional arguments to forward to the method.
      **kwargs: Keyword arguments to forward to the method.

    Returns:
      The attribute.
    """
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def access(self, name: Text) -> Any:
    """Access an attribute of the external environment.

    This method blocks.

    Args:
      name: Name of the attribute to access.

    Returns:
      The attribute value.
    """
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def close(self) -> None:
    """Send a close message to the external process and join it."""
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    if self._process.is_alive():
      self._process.join(5)

  def step(self,
           action: types.NestedArray,
           blocking: bool = True) -> Union[ts.TimeStep, Promise]:
    """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      time step when blocking, otherwise callable that returns the time step.
    """
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking: bool = True) -> Union[ts.TimeStep, Promise]:
    """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def render(self,
             mode: Text = 'rgb_array',
             blocking: bool = True) -> Union[types.NestedArray, Promise]:
    """Renders the environment.

    Args:
      mode: Rendering mode. Only 'rgb_array' is supported.
      blocking: Whether to wait for the result.

    Returns:
      An ndarray of shape [width, height, 3] denoting an RGB image when
      blocking. Otherwise, callable that returns the rendered image.
    Raises:
      NotImplementedError: If the environment does not support rendering,
        or any other modes than `rgb_array` is given.
    """
    if mode != 'rgb_array':
      raise NotImplementedError('Only rgb_array rendering mode is supported. '
                                'Got %s' % mode)
    promise = self.call('render')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    """Wait for a message from the worker process and return its payload.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The reveived message is of an unknown type.

    Returns:
      Payload object of the message.
    """
    message, payload = self._conn.recv()
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    self.close()
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    """The process waits for actions and sends back environment results.

    Args:
      conn: Connection for communication to the main process.

    Raises:
      KeyError: When receiving a message of unknown type.
    """
    try:
      env = cloudpickle.loads(self._pickled_env_constructor)()
      action_spec = env.action_spec()
      conn.send(self._READY)  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(_POLLING_PERIOD):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          if self._flatten and name == 'step':
            args = [tf.nest.pack_sequence_as(action_spec, args[0])]
          result = getattr(env, name)(*args, **kwargs)
          if self._flatten and name in ['step', 'reset']:
            result = tf.nest.flatten(result)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          env.close()
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:  # pylint: disable=broad-except
      etype, evalue, tb = sys.exc_info()
      stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
      message = 'Error in environment process: {}'.format(stacktrace)
      logging.error(message)
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      conn.close()
