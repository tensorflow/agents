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

"""Python RL Environment API.

Adapted from the Deepmind's Environment API as seen in:
  https://github.com/deepmind/dm_control
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
from typing import Any, Optional, Text

import numpy as np
import six

from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common


@six.add_metaclass(abc.ABCMeta)
class PyEnvironment(object):
  """Abstract base class for Python RL environments.

  Observations and valid actions are described with `ArraySpec`s, defined in
  the `specs` module.

  If the environment can run multiple steps at the same time and take a batched
  set of actions and return a batched set of observations, it should overwrite
  the property batched to True.

  Environments are assumed to auto reset once they reach the end of the episode,
  if prefer that the base class handle auto_reset set `handle_auto_reset=True`.
  """

  def __init__(self, handle_auto_reset: bool = False):
    """Base class for Python RL environments.

    Args:
      handle_auto_reset: When `True` the base class will handle auto_reset of
        the Environment.
    """
    self._handle_auto_reset = handle_auto_reset
    self._current_time_step = None
    common.assert_members_are_not_overridden(
        base_cls=PyEnvironment, instance=self, denylist=('reset', 'step'))

  @property
  def batched(self) -> bool:
    """Whether the environment is batched or not.

    If the environment supports batched observations and actions, then overwrite
    this property to True.

    A batched environment takes in a batched set of actions and returns a
    batched set of observations. This means for all numpy arrays in the input
    and output nested structures, the first dimension is the batch size.

    When batched, the left-most dimension is not part of the action_spec
    or the observation_spec and corresponds to the batch dimension.

    When batched and handle_auto_reset, it checks `np.all(steps.is_last())`.

    Returns:
      A boolean indicating whether the environment is batched or not.
    """
    return False

  @property
  def batch_size(self) -> Optional[int]:
    """The batch size of the environment.

    Returns:
      The batch size of the environment, or `None` if the environment is not
      batched.

    Raises:
      RuntimeError: If a subclass overrode batched to return True but did not
        override the batch_size property.
    """
    if self.batched:
      raise RuntimeError(
          'Environment %s marked itself as batched but did not override the '
          'batch_size property' % type(self))
    return None

  def should_reset(self, current_time_step: ts.TimeStep) -> bool:
    """Whether the Environmet should reset given the current timestep.

    By default it only resets when all time_steps are `LAST`.

    Args:
      current_time_step: The current `TimeStep`.

    Returns:
      A bool indicating whether the Environment should reset or not.
    """
    handle_auto_reset = getattr(self, '_handle_auto_reset', False)
    return handle_auto_reset and np.all(current_time_step.is_last())

  @abc.abstractmethod
  def observation_spec(self) -> types.NestedArraySpec:
    """Defines the observations provided by the environment.

    May use a subclass of `ArraySpec` that specifies additional properties such
    as min and max bounds on the values.

    Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
    """

  @abc.abstractmethod
  def action_spec(self) -> types.NestedArraySpec:
    """Defines the actions that should be provided to `step()`.

    May use a subclass of `ArraySpec` that specifies additional properties such
    as min and max bounds on the values.

    Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
    """

  def reward_spec(self) -> types.NestedArraySpec:
    """Defines the rewards that are returned by `step()`.

    Override this method to define an environment that uses non-standard reward
    values, for example an environment with array-valued rewards.

    Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
    """
    return array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward')

  def discount_spec(self) -> types.NestedArraySpec:
    """Defines the discount that are returned by `step()`.

    Override this method to define an environment that uses non-standard
    discount values, for example an environment with array-valued discounts.

    Returns:
      An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
    """
    return array_spec.BoundedArraySpec(
        shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount')

  def time_step_spec(self) -> ts.TimeStep:
    """Describes the `TimeStep` fields returned by `step()`.

    Override this method to define an environment that uses non-standard values
    for any of the items returned by `step()`. For example, an environment with
    array-valued rewards.

    Returns:
      A `TimeStep` namedtuple containing (possibly nested) `ArraySpec`s defining
      the step_type, reward, discount, and observation structure.
    """
    return ts.time_step_spec(self.observation_spec(), self.reward_spec())

  def current_time_step(self) -> ts.TimeStep:
    """Returns the current timestep."""
    return self._current_time_step

  def reset(self) -> ts.TimeStep:
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Note: Subclasses cannot override this directly. Subclasses implement
    _reset() which will be called by this method. The output of _reset() will
    be cached and made available through current_time_step().

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` of `FIRST`.
        reward: 0.0, indicating the reward.
        discount: 1.0, indicating the discount.
        observation: A NumPy array, or a nested dict, list or tuple of arrays
          corresponding to `observation_spec()`.
    """
    self._current_time_step = self._reset()
    return self._current_time_step

  def step(self, action: types.NestedArray) -> ts.TimeStep:
    """Updates the environment according to the action and returns a `TimeStep`.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step the implementation of `_step` in the environment should call
    `reset` to start a new sequence and ignore `action`.

    This method will start a new sequence if called after the environment
    has been constructed and `reset` has not been called. In this case
    `action` will be ignored.

    If `should_reset(current_time_step)` is True, then this method will `reset`
    by itself. In this case `action` will be ignored.

    Note: Subclasses cannot override this directly. Subclasses implement
    _step() which will be called by this method. The output of _step() will be
    cached and made available through current_time_step().

    Args:
      action: A NumPy array, or a nested dict, list or tuple of arrays
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: A NumPy array, reward value for this timestep.
        discount: A NumPy array, discount in the range [0, 1].
        observation: A NumPy array, or a nested dict, list or tuple of arrays
          corresponding to `observation_spec()`.
    """
    if (self._current_time_step is None or
        self.should_reset(self._current_time_step)):
      return self.reset()

    self._current_time_step = self._step(action)
    return self._current_time_step

  def close(self) -> None:
    """Frees any resources used by the environment.

    Implement this method for an environment backed by an external process.

    This method be used directly

    ```python
    env = Env(...)
    # Use env.
    env.close()
    ```

    or via a context manager

    ```python
    with Env(...) as env:
      # Use env.
    ```
    """
    pass

  def __enter__(self):
    """Allows the environment to be used in a with-statement context."""
    return self

  def __exit__(self, unused_exception_type, unused_exc_value, unused_traceback):
    """Allows the environment to be used in a with-statement context."""
    self.close()

  def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
    """Renders the environment.

    Args:
      mode: One of ['rgb_array', 'human']. Renders to an numpy array, or brings
        up a window where the environment can be visualized.

    Returns:
      An ndarray of shape [width, height, 3] denoting an RGB image if mode is
      `rgb_array`. Otherwise return nothing and render directly to a display
      window.
    Raises:
      NotImplementedError: If the environment does not support rendering.
    """
    del mode  # unused
    raise NotImplementedError('No rendering support.')

  def seed(self, seed: types.Seed) -> Any:
    """Seeds the environment.

    Args:
      seed: Value to use as seed for the environment.
    """
    del seed  # unused
    raise NotImplementedError('No seed support for this environment.')

  def get_info(self) -> types.NestedArray:
    """Returns the environment info returned on the last step.

    Returns:
      Info returned by last call to step(). None by default.

    Raises:
      NotImplementedError: If the environment does not use info.
    """
    raise NotImplementedError('No support of get_info for this environment.')

  def get_state(self) -> Any:
    """Returns the `state` of the environment.

    The `state` contains everything required to restore the environment to the
    current configuration. This can contain e.g.
      - The current time_step.
      - The number of steps taken in the environment (for finite horizon MDPs).
      - Hidden state (for POMDPs).

    Callers should not assume anything about the contents or format of the
    returned `state`. It should be treated as a token that can be passed back to
    `set_state()` later.

    Note that the returned `state` handle should not be modified by the
    environment later on, and ensuring this (e.g. using copy.deepcopy) is the
    responsibility of the environment.

    Returns:
      state: The current state of the environment.
    """
    raise NotImplementedError('This environment has not implemented '
                              '`get_state()`.')

  def set_state(self, state: Any) -> None:
    """Restores the environment to a given `state`.

    See definition of `state` in the documentation for get_state().

    Args:
      state: A state to restore the environment to.
    """
    raise NotImplementedError('This environment has not implemented '
                              '`set_state()`.')

  #  These methods are to be implemented by subclasses:

  @abc.abstractmethod
  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Updates the environment according to action and returns a `TimeStep`.

    See `step(self, action)` docstring for more details.

    Args:
      action: A NumPy array, or a nested dict, list or tuple of arrays
        corresponding to `action_spec()`.
    """

  @abc.abstractmethod
  def _reset(self) -> ts.TimeStep:
    """Starts a new sequence, returns the first `TimeStep` of this sequence.

    See `reset(self)` docstring for more details
    """
