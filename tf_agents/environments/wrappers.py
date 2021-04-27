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

"""Environment wrappers.

Wrappers in this module can be chained to change the overall behaviour of an
environment in common ways.
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
import collections
import cProfile
from typing import Any, Callable, Optional, Sequence, Text, Union

import gin
import numpy as np
import six
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


class PyEnvironmentBaseWrapper(py_environment.PyEnvironment):
  """PyEnvironment wrapper forwards calls to the given environment."""

  def __init__(self, env: Any):
    super(PyEnvironmentBaseWrapper, self).__init__()
    self._env = env

  def __getattr__(self, name: Text):
    """Forward all other calls to the base environment."""
    return getattr(self._env, name)

  @property
  def batched(self) -> bool:
    return getattr(self._env, 'batched', False)

  @property
  def batch_size(self) -> Optional[types.Int]:
    return getattr(self._env, 'batch_size', None)

  def _reset(self):
    return self._env.reset()

  def _step(self, action):
    return self._env.step(action)

  def get_info(self) -> Any:
    return self._env.get_info()

  def observation_spec(self) -> types.NestedArray:
    return self._env.observation_spec()

  def action_spec(self) -> types.NestedArray:
    return self._env.action_spec()

  def close(self) -> None:
    return self._env.close()

  def render(self, mode: Text = 'rgb_array') -> types.NestedArray:
    return self._env.render(mode)

  def seed(self, seed: types.Seed) -> types.Seed:
    return self._env.seed(seed)

  def wrapped_env(self) -> Any:
    return self._env

  def set_state(self, state: Any) -> None:
    self._env.set_state(state)

  def get_state(self) -> Any:
    return self._env.get_state()


@gin.configurable
class TimeLimit(PyEnvironmentBaseWrapper):
  """End episodes after specified number of steps."""

  def __init__(self, env: py_environment.PyEnvironment, duration: types.Int):
    super(TimeLimit, self).__init__(env)
    self._duration = duration
    self._num_steps = None

  def _reset(self):
    self._num_steps = 0
    return self._env.reset()

  def _step(self, action):
    if self._num_steps is None:
      return self.reset()

    time_step = self._env.step(action)

    self._num_steps += 1
    if self._num_steps >= self._duration:
      time_step = time_step._replace(step_type=ts.StepType.LAST)

    if time_step.is_last():
      self._num_steps = None

    return time_step

  @property
  def duration(self) -> types.Int:
    return self._duration


@gin.configurable
class FixedLength(PyEnvironmentBaseWrapper):
  """Truncates long episodes and pads short episodes to have a fixed length.

  If the episode is short it will pad with the last step and set discount to 0.
  """

  def __init__(self, env: py_environment.PyEnvironment, fix_length: types.Int):
    super(FixedLength, self).__init__(env)
    self._fix_length = fix_length
    self._num_steps = None
    self._episode_ended = False

  def _reset(self):
    self._num_steps = 0
    self._episode_ended = False
    return self._env.reset()

  def _step(self, action):
    if self._num_steps is None:
      return self.reset()

    time_step = self.current_time_step()
    if time_step.is_last():
      if self._num_steps < self._fix_length:
        self._num_steps += 1
        if self._episode_ended:
          return time_step
        else:
          self._episode_ended = True
          return time_step._replace(discount=0.0, reward=0. * time_step.reward)
      else:
        return self.reset()

    else:
      time_step = self._env.step(action)

      self._num_steps += 1
      if self._num_steps >= self._fix_length:
        time_step = time_step._replace(step_type=ts.StepType.LAST)
        self._num_steps = None

      return time_step

  @property
  def fix_length(self) -> types.Int:
    return self._fix_length


@gin.configurable
class PerformanceProfiler(PyEnvironmentBaseWrapper):
  """End episodes after specified number of steps."""

  def __init__(self, env: py_environment.PyEnvironment,
               process_profile_fn: Callable[[cProfile.Profile], Any],
               process_steps: int):
    """Create a PerformanceProfiler that uses cProfile to profile env execution.

    Args:
      env: Environment to wrap.
      process_profile_fn: A callback that accepts a `Profile` object.
        After `process_profile_fn` is called, profile information is reset.
      process_steps: The frequency with which `process_profile_fn` is
        called.  The counter is incremented each time `step` is called
        (not `reset`); every `process_steps` steps, `process_profile_fn`
        is called and the profiler is reset.
    """
    super(PerformanceProfiler, self).__init__(env)
    self._started = False
    self._num_steps = 0
    self._process_steps = process_steps
    self._process_profile_fn = process_profile_fn
    self._profile = cProfile.Profile()

  def _reset(self):
    self._profile.enable()
    try:
      return self._env.reset()
    finally:
      self._profile.disable()

  def _step(self, action):
    if not self._started:
      self._started = True
      self._num_steps += 1
      return self.reset()

    self._profile.enable()
    try:
      time_step = self._env.step(action)
    finally:
      self._profile.disable()

    self._num_steps += 1
    if self._num_steps >= self._process_steps:
      self._process_profile_fn(self._profile)
      self._profile = cProfile.Profile()
      self._num_steps = 0

    if time_step.is_last():
      self._started = False

    return time_step


@gin.configurable
class ActionRepeat(PyEnvironmentBaseWrapper):
  """Repeates actions over n-steps while acummulating the received reward."""

  def __init__(self, env: py_environment.PyEnvironment, times: types.Int):
    """Creates an action repeat wrapper.

    Args:
      env: Environment to wrap.
      times: Number of times the action should be repeated.

    Raises:
      ValueError: If the times parameter is not greater than 1.
    """
    super(ActionRepeat, self).__init__(env)
    if times <= 1:
      raise ValueError(
          'Times parameter ({}) should be greater than 1'.format(times))
    self._times = times

  def _step(self, action):
    total_reward = 0

    for _ in range(self._times):
      time_step = self._env.step(action)
      total_reward += time_step.reward
      if time_step.is_first() or time_step.is_last():
        break

    total_reward = np.asarray(total_reward,
                              dtype=np.asarray(time_step.reward).dtype)
    return ts.TimeStep(time_step.step_type, total_reward, time_step.discount,
                       time_step.observation)


@gin.configurable
class ObservationFilterWrapper(PyEnvironmentBaseWrapper):
  """Filters observations based on an array of indexes.

  Note that this wrapper only supports single-dimensional observations.
  """

  def __init__(self,
               env: py_environment.PyEnvironment,
               idx: Union[Sequence[int], np.ndarray]):
    """Creates an observation filter wrapper.

    Args:
      env: Environment to wrap.
      idx: Array of indexes pointing to elements to include in output.

    Raises:
      ValueError: If observation spec is nested.
      ValueError: If indexes are not single-dimensional.
      ValueError: If no index is provided.
      ValueError: If one of the indexes is out of bounds.
    """
    super(ObservationFilterWrapper, self).__init__(env)
    idx = np.array(idx)
    if tf.nest.is_nested(env.observation_spec()):
      raise ValueError('ObservationFilterWrapper only works with single-array '
                       'observations (not nested).')
    if len(idx.shape) != 1:
      raise ValueError('ObservationFilterWrapper only works with '
                       'single-dimensional indexes for filtering.')
    if idx.shape[0] < 1:
      raise ValueError('At least one index needs to be provided for filtering.')
    if not np.all(idx < env.observation_spec().shape[0]):
      raise ValueError('One of the indexes is out of bounds.')

    self._idx = idx
    self._observation_spec = env.observation_spec().replace(shape=idx.shape)

  def _step(self, action):
    time_step = self._env.step(action)
    return time_step._replace(observation=
                              np.array(time_step.observation)[self._idx])

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def _reset(self):
    time_step = self._env.reset()
    return time_step._replace(observation=
                              np.array(time_step.observation)[self._idx])


@gin.configurable
class RunStats(PyEnvironmentBaseWrapper):
  """Wrapper that accumulates run statistics as the environment iterates.

  Note the episodes are only counted if the environment is stepped until the
  last timestep. This will be triggered correctly when using TimeLimit wrappers.

  In summary:
   * episodes == number of LAST timesteps,
   * resets   == number of FIRST timesteps,
  """

  def __init__(self, env: py_environment.PyEnvironment):
    super(RunStats, self).__init__(env)
    self._episodes = 0
    self._resets = 0
    self._episode_steps = 0
    self._total_steps = 0

  @property
  def episodes(self) -> int:
    return self._episodes

  @property
  def episode_steps(self) -> int:
    return self._episode_steps

  @property
  def total_steps(self) -> int:
    return self._total_steps

  @property
  def resets(self) -> int:
    return self._resets

  def _reset(self):
    self._resets += 1
    self._episode_steps = 0
    return self._env.reset()

  def _step(self, action):
    time_step = self._env.step(action)

    if time_step.is_first():
      self._resets += 1
      self._episode_steps = 0
    else:
      self._total_steps += 1
      self._episode_steps += 1

    if time_step.is_last():
      self._episodes += 1

    return time_step


@gin.configurable
class ActionDiscretizeWrapper(PyEnvironmentBaseWrapper):
  """Wraps an environment with continuous actions and discretizes them."""

  def __init__(self,
               env: py_environment.PyEnvironment,
               num_actions: np.ndarray):
    """Constructs a wrapper for discretizing the action space.

    **Note:** Only environments with a single BoundedArraySpec are supported.

    Args:
      env: Environment to wrap.
      num_actions: A np.array of the same shape as the environment's
        action_spec. Elements in the array specify the number of actions to
        discretize to for each dimension.

    Raises:
      ValueError: IF the action_spec shape and the limits shape are not equal.
    """
    super(ActionDiscretizeWrapper, self).__init__(env)

    action_spec = tf.nest.flatten(env.action_spec())
    if len(action_spec) != 1:
      raise ValueError(
          'ActionDiscretizeWrapper only supports environments with a single '
          'action spec. Got {}'.format(env.action_spec()))

    action_spec = action_spec[0]
    self._original_spec = action_spec
    self._num_actions = np.broadcast_to(num_actions, action_spec.shape)

    if action_spec.shape != self._num_actions.shape:
      raise ValueError('Spec {} and limit shape do not match. Got {}'.format(
          action_spec, self._num_actions.shape))

    self._discrete_spec, self._action_map = self._discretize_spec(
        action_spec, self._num_actions)

  def _discretize_spec(self, spec, limits):
    """Generates a discrete bounded spec and a linspace for the given limits.

    Args:
      spec: An array_spec to discretize.
      limits: A np.array with limits for the given spec.

    Returns:
      Tuple with the discrete_spec along with a list of lists mapping actions.
    Raises:
      ValueError: If not all limits value are >=2.
    """
    if not np.all(limits >= 2):
      raise ValueError('num_actions should all be at least size 2.')

    limits = np.asarray(limits)
    # Simplify shape of bounds if they are all equal.
    if np.all(limits == limits.flat[0]):
      discrete_spec_max_limit = limits.flat[0]
    else:
      discrete_spec_max_limit = limits
    # Workaround for b/148086610. Makes the discretized wrapper generate a
    # scalar spec when possible.
    shape = () if spec.shape == (1,) else spec.shape
    discrete_spec = array_spec.BoundedArraySpec(
        shape=shape,
        dtype=np.int32,
        minimum=0,
        maximum=discrete_spec_max_limit - 1,
        name=spec.name)

    minimum = np.broadcast_to(spec.minimum, shape)
    maximum = np.broadcast_to(spec.maximum, shape)

    action_map = [
        np.linspace(spec_min, spec_max, num=n_actions)
        for spec_min, spec_max, n_actions in zip(
            np.nditer(minimum), np.nditer(maximum), np.nditer(limits))
    ]

    return discrete_spec, action_map

  def action_spec(self) -> types.NestedArraySpec:
    return self._discrete_spec

  def _map_actions(self, action, action_map):
    """Maps the given discrete action to the corresponding continuous action.

    Args:
      action: Discrete action to map.
      action_map: Array with the continuous linspaces for the action.

    Returns:
      Numpy array with the mapped continuous actions.
    Raises:
      ValueError: If the given action's shpe does not match the action_spec
      shape.
    """
    action = np.asarray(action)
    if action.shape != self._discrete_spec.shape:
      raise ValueError(
          'Received action with incorrect shape. Got {}, expected {}'.format(
              action.shape, self._discrete_spec.shape))

    mapped_action = [action_map[i][a] for i, a in enumerate(action.flatten())]
    return np.reshape(mapped_action, newshape=self._original_spec.shape)

  def _step(self, action):
    """Steps the environment while remapping the actions.

    Args:
      action: Action to take.

    Returns:
      The next time_step from the environment.
    """
    continuous_actions = self._map_actions(action, self._action_map)
    env_action_spec = self._env.action_spec()

    if tf.nest.is_nested(env_action_spec):
      continuous_actions = tf.nest.pack_sequence_as(env_action_spec,
                                                    [continuous_actions])
    return self._env.step(continuous_actions)


@gin.configurable
class ActionClipWrapper(PyEnvironmentBaseWrapper):
  """Wraps an environment and clips actions to spec before applying."""

  def _step(self, action):
    """Steps the environment after clipping the actions.

    Args:
      action: Action to take.

    Returns:
      The next time_step from the environment.
    """
    env_action_spec = self._env.action_spec()

    def _clip_to_spec(act_spec, act):
      # NumPy does not allow both min and max to be None
      if act_spec.minimum is None and act_spec.maximum is None:
        return act
      return np.clip(act, act_spec.minimum, act_spec.maximum)

    clipped_actions = nest.map_structure_up_to(env_action_spec, _clip_to_spec,
                                               env_action_spec, action)

    return self._env.step(clipped_actions)


# TODO(b/119321125): Remove this once index_with_actions supports negative
# actions.
class ActionOffsetWrapper(PyEnvironmentBaseWrapper):
  """Offsets actions to be zero-based.

  This is useful for the DQN agent, which currently doesn't support
  negative-valued actions.
  """

  def __init__(self, env: py_environment.PyEnvironment):
    super(ActionOffsetWrapper, self).__init__(env)
    if tf.nest.is_nested(self._env.action_spec()):
      raise ValueError('ActionOffsetWrapper only works with single-array '
                       'action specs (not nested specs).')
    if not tensor_spec.is_bounded(self._env.action_spec()):
      raise ValueError('ActionOffsetWrapper only works with bounded '
                       'action specs.')
    if not tensor_spec.is_discrete(self._env.action_spec()):
      raise ValueError('ActionOffsetWrapper only works with discrete '
                       'action specs.')

  def action_spec(self) -> types.NestedArraySpec:
    spec = self._env.action_spec()
    minimum = np.zeros(shape=spec.shape, dtype=spec.dtype)
    maximum = spec.maximum - spec.minimum
    return array_spec.BoundedArraySpec(spec.shape, spec.dtype, minimum=minimum,
                                       maximum=maximum)

  def _step(self, action):
    return self._env.step(action + self._env.action_spec().minimum)


@gin.configurable
class FlattenObservationsWrapper(PyEnvironmentBaseWrapper):
  """Wraps an environment and flattens nested multi-dimensional observations.

  Example:
    The observation returned by the environment is a multi-dimensional sequence
    of items of varying lengths.

    timestep.observation_spec =
      {'position': ArraySpec(shape=(4,), dtype=float32),
       'target': ArraySpec(shape=(5,), dtype=float32)}

    timestep.observation =
      {'position':  [1,2,3,4], target': [5,6,7,8,9]}

    By packing the observation, we reduce the dimensions into a single dimension
    and concatenate the values of all the observations into one array.

    timestep.observation_spec = (
      'packed_observations': ArraySpec(shape=(9,), dtype=float32)

    timestep.observation = [1,2,3,4,5,6,7,8,9] # Array of len-9.


  Note: By packing observations into a single dimension, the specific ArraySpec
  structure of each observation (such as if min or max bounds are set) are lost.
  """

  def __init__(self,
               env: py_environment.PyEnvironment,
               observations_allowlist: Optional[Sequence[Text]] = None):
    """Initializes a wrapper to flatten environment observations.

    Args:
      env: A `py_environment.PyEnvironment` environment to wrap.
      observations_allowlist: A list of observation keys that want to be
        observed from the environment.  All other observations returned are
        filtered out.  If not provided, all observations will be kept.
        Additionally, if this is provided, the environment is expected to return
        a dictionary of observations.

    Raises:
      ValueError: If the current environment does not return a dictionary of
        observations and observations_allowlist is provided.
      ValueError: If the observation_allowlist keys are not found in the
        environment.
    """
    super(FlattenObservationsWrapper, self).__init__(env)

    # If observations allowlist is provided:
    #  Check that the environment returns a dictionary of observations.
    #  Check that the set of allowed keys is a found in the environment keys.
    if observations_allowlist is not None:
      if not isinstance(env.observation_spec(), dict):
        raise ValueError(
            'If you provide an observations allowlist, the current environment '
            'must return a dictionary of observations! The returned observation'
            ' spec is type %s.' % (type(env.observation_spec())))

      # Check that observation allowlist keys are valid observation keys.
      if not (set(observations_allowlist).issubset(
          env.observation_spec().keys())):
        raise ValueError(
            'The observation allowlist contains keys not found in the '
            'environment! Unknown keys: %s' % list(
                set(observations_allowlist).difference(
                    env.observation_spec().keys())))

    # Check that all observations have the same dtype. This dtype will be used
    # to create the flattened ArraySpec.
    env_dtypes = list(
        set([obs.dtype for obs in env.observation_spec().values()]))
    if len(env_dtypes) != 1:
      raise ValueError('The observation spec must all have the same dtypes! '
                       'Currently found dtypes: %s' % (env_dtypes))
    inferred_spec_dtype = env_dtypes[0]

    self._observation_spec_dtype = inferred_spec_dtype
    self._observations_allowlist = observations_allowlist
    # Update the observation spec in the environment.
    observations_spec = env.observation_spec()
    if self._observations_allowlist is not None:
      observations_spec = self._filter_observations(observations_spec)

    # Compute the observation length after flattening the observation items and
    # nested structure. Observation specs are not batched.
    observation_total_len = sum(
        int(np.prod(observation.shape))
        for observation in self._flatten_nested_observations(
            observations_spec, is_batched=False))

    # Update the observation spec as an array of one-dimension.
    self._flattened_observation_spec = array_spec.ArraySpec(
        shape=(observation_total_len,),
        dtype=self._observation_spec_dtype,
        name='packed_observations')

  def _filter_observations(self, observations):
    """Filters out unwanted observations from the environment.

    Args:
      observations: A nested dictionary of arrays corresponding to
      `observation_spec()`. This is the observation attribute in the
      TimeStep object returned by the environment.

    Returns:
      A nested dict of arrays corresponding to `observation_spec()` with only
        observation keys in the observation allowlist.
    """
    filter_out = set(observations.keys()).difference(
        self._observations_allowlist)
    # Remove unwanted keys from the observation list.
    for filter_key in filter_out:
      del observations[filter_key]
    return observations

  def _pack_and_filter_timestep_observation(self, timestep):
    """Pack and filter observations into a single dimension.

    Args:
      timestep: A `TimeStep` namedtuple containing:
        - step_type: A `StepType` value.
        - reward: Reward at this timestep.
        - discount: A discount in the range [0, 1].
        - observation: A NumPy array, or a nested dict, list or tuple of arrays
          corresponding to `observation_spec()`.

    Returns:
      A new `TimeStep` namedtuple that has filtered observations and packed into
        a single dimenison.
    """
    # We can't set attribute to the TimeStep tuple, so we make a copy of the
    # observations.
    observations = timestep.observation
    if self._observations_allowlist is not None:
      observations = self._filter_observations(observations)

    return ts.TimeStep(
        timestep.step_type, timestep.reward, timestep.discount,
        self._flatten_nested_observations(
            observations, is_batched=self._env.batched))

  def _flatten_nested_observations(self, observations, is_batched):
    """Flatten individual observations and then flatten the nested structure.

    Args:
      observations: A flattened NumPy array of shape corresponding to
        `observation_spec()` or an `observation_spec()`.
      is_batched: Whether or not the provided observation is batched.

    Returns:
      A concatenated and flattened NumPy array of observations.
    """

    def np_flatten(x):
      # Check if observations are batch, and if so keep the batch dimension and
      # flatten the all other dimensions into one.
      if is_batched:
        return np.reshape(x, [x.shape[0], -1])
      else:
        return np.reshape(x, [-1])

    # Flatten the individual observations if they are multi-dimensional and then
    # flatten the nested structure.
    flat_observations = [np_flatten(x) for x in tf.nest.flatten(observations)]
    axis = 1 if is_batched else 0
    return np.concatenate(flat_observations, axis=axis)

  def _step(self, action):
    """Steps the environment while packing the observations returned.

    Args:
      action: A NumPy array, or a nested dict, list or tuple of arrays
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A flattened NumPy array of shape corresponding to
         `observation_spec()`.
    """
    return self._pack_and_filter_timestep_observation(self._env.step(action))

  def _reset(self):
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` of `FIRST`.
        reward: `None`, indicating the reward is undefined.
        discount: `None`, indicating the discount is undefined.
        observation: A flattened NumPy array of shape corresponding to
         `observation_spec()`.
    """
    return self._pack_and_filter_timestep_observation(self._env.reset())

  def observation_spec(self) -> types.NestedArraySpec:
    """Defines the observations provided by the environment.

    Returns:
      An `ArraySpec` with a shape of the total length of observations kept.
    """
    return self._flattened_observation_spec


@six.add_metaclass(abc.ABCMeta)
class GoalReplayEnvWrapper(PyEnvironmentBaseWrapper):
  """Adds a goal to the observation, used for HER (Hindsight Experience Replay).

  Sources:
    [1] Hindsight Experience Replay. https://arxiv.org/abs/1707.01495.

  To use this wrapper, create an environment-specific version by inheriting this
  class.
  """

  def __init__(self, env: py_environment.PyEnvironment):
    """Initializes a wrapper to add a goal to the observation.

    Args:
      env: A `py_environment.PyEnvironment` environment to wrap.

    Raises:
      ValueError: If environment observation is not a dict
    """
    super(GoalReplayEnvWrapper, self).__init__(env)
    self._env = env
    self._goal = None

  @abc.abstractmethod
  def get_trajectory_with_goal(self,
                               trajectory: ts.TimeStep,
                               goal: types.NestedArray) -> ts.TimeStep:
    """Generates a new trajectory assuming the given goal was the actual target.

    One example is updating a "distance-to-goal" field in the observation. Note
    that relevant state information must be recovered or re-calculated from the
    given trajectory.

    Args:
      trajectory: An instance of `TimeStep`.
      goal: Environment specific goal

    Returns:
      Updated instance of `TimeStep`

    Raises:
      NotImplementedError: function should be implemented in child class.
    """
    pass

  @abc.abstractmethod
  def get_goal_from_trajectory(self,
                               trajectory: ts.TimeStep) -> types.NestedArray:
    """Extracts the goal from a given trajectory.

    Args:
      trajectory: An instance of `TimeStep`.

    Returns:
      Environment specific goal

    Raises:
      NotImplementedError: function should be implemented in child class.
    """
    pass

  def _reset(self, *args, **kwargs):
    """Resets the environment, updating the trajectory with goal."""
    trajectory = self._env.reset(*args, **kwargs)
    self._goal = self.get_goal_from_trajectory(trajectory)
    return self.get_trajectory_with_goal(trajectory, self._goal)

  def _step(self, *args, **kwargs):
    """Execute a step in the environment, updating the trajectory with goal."""
    trajectory = self._env.step(*args, **kwargs)
    return self.get_trajectory_with_goal(trajectory, self._goal)


@gin.configurable
class HistoryWrapper(PyEnvironmentBaseWrapper):
  """Adds observation and action history to the environment's observations."""

  def __init__(self,
               env: py_environment.PyEnvironment,
               history_length: int = 3,
               include_actions: bool = False,
               tile_first_step_obs: bool = False,
               ):
    """Initializes a HistoryWrapper.

    Args:
      env: Environment to wrap.
      history_length: Length of the history to attach.
      include_actions: Whether actions should be included in the history.
      tile_first_step_obs: If True the observation on reset is tiled to fill the
       history.
    """
    super(HistoryWrapper, self).__init__(env)
    self._history_length = history_length
    self._include_actions = include_actions
    self._tile_first_step_obs = tile_first_step_obs

    self._zero_observation = self._zeros_from_spec(env.observation_spec())
    self._zero_action = self._zeros_from_spec(env.action_spec())

    self._observation_history = collections.deque(maxlen=history_length)
    self._action_history = collections.deque(maxlen=history_length)

    self._observation_spec = self._get_observation_spec()

  def _get_observation_spec(self):

    def _update_shape(spec):
      return spec.replace(shape=(self._history_length,) + spec.shape)

    observation_spec = tf.nest.map_structure(_update_shape,
                                             self._env.observation_spec())

    if self._include_actions:
      action_spec = tf.nest.map_structure(_update_shape,
                                          self._env.action_spec())
      return {'observation': observation_spec, 'action': action_spec}
    else:
      return observation_spec

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def _zeros_from_spec(self, spec):

    def _zeros(spec):
      return np.zeros(spec.shape, dtype=spec.dtype)

    return tf.nest.map_structure(_zeros, spec)

  def _add_history(self, time_step, action):
    self._observation_history.append(time_step.observation)
    self._action_history.append(action)

    if self._include_actions:
      observation = {
          'observation':
              nest_utils.stack_nested_arrays(self._observation_history),
          'action':
              nest_utils.stack_nested_arrays(self._action_history)
      }
    else:
      observation = nest_utils.stack_nested_arrays(self._observation_history)
    return time_step._replace(observation=observation)

  def _reset(self):
    time_step = self._env.reset()

    if self._tile_first_step_obs:
      self._observation_history.extend([
          time_step.observation.copy() for _ in range(self._history_length - 1)
      ])
    else:
      self._observation_history.extend([self._zero_observation] *
                                       (self._history_length - 1))
    self._action_history.extend([self._zero_action] *
                                (self._history_length - 1))

    return self._add_history(time_step, self._zero_action)

  def _step(self, action):
    if self.current_time_step() is None or self.current_time_step().is_last():
      return self._reset()

    time_step = self._env.step(action)
    return self._add_history(time_step, action)


@gin.configurable
class OneHotActionWrapper(PyEnvironmentBaseWrapper):
  """Converts discrete action to one_hot format."""

  def __init__(self, env: py_environment.PyEnvironment):
    super(OneHotActionWrapper, self).__init__(env)

    def convert_to_one_hot(spec):
      """Convert spec to one_hot format."""
      if np.issubdtype(spec.dtype, np.integer):
        if len(spec.shape) > 1:
          raise ValueError('OneHotActionWrapper only supports single action!'
                           'action_spec: {}'.format(spec))

        num_actions = spec.maximum - spec.minimum + 1
        output_shape = spec.shape + (num_actions,)

        return array_spec.BoundedArraySpec(
            shape=output_shape,
            dtype=spec.dtype,
            minimum=0,
            maximum=1,
            name='one_hot_action_spec')
      else:
        return spec

    self._one_hot_action_spec = tf.nest.map_structure(
        convert_to_one_hot, self._env.action_spec())

  def action_spec(self) -> types.NestedArraySpec:
    return self._one_hot_action_spec

  def _step(self, action):

    def convert_back(action, inner_spec, spec):
      if action.shape != inner_spec.shape or action.dtype != inner_spec.dtype:
        raise ValueError('Action shape/dtype different from its definition in '
                         'the inner_spec. Action: {action}. Inner_spec: '
                         '{spec}.'.format(action=action, spec=spec))
      if np.issubdtype(action.dtype, np.integer):
        action = spec.minimum + np.argmax(action, axis=-1)
      return action

    action = tf.nest.map_structure(
        convert_back, action, self._one_hot_action_spec,
        self._env.action_spec())
    return self._env.step(action)


@gin.configurable
class ExtraDisabledActionsWrapper(PyEnvironmentBaseWrapper):
  r"""Adds extra unavailable actions.

  This wrapper introduces new actions to the environment, but these actions are
  always disabled via action masking. That is, if the original observation is
  `obs`, the new observation will be
  ```
  new_obs = (obs, [1, 1, ..., 1, 0, 0, 0, ..., 0])
                   \----v----/   \------v------/
                   num_actions    num_extra_actions
  ```
  where the second element of the above tuple is the mask. Note that the above
  example does not include the batch dimension.
  """

  def __init__(self, env: py_environment.PyEnvironment, num_extra_actions: int):
    """Initializes an instance of `ExtraDisabledActionsWrapper`.

    Args:
      env: The environment to wrap.
      num_extra_actions: The number of extra actions to add.
    """
    super(ExtraDisabledActionsWrapper, self).__init__(env)
    orig_action_spec = env.action_spec()
    self._action_spec = array_spec.BoundedArraySpec(
        shape=orig_action_spec.shape,
        dtype=orig_action_spec.dtype,
        minimum=orig_action_spec.minimum,
        maximum=orig_action_spec.maximum + num_extra_actions)
    mask_spec = array_spec.ArraySpec(
        shape=[self._action_spec.maximum - self._action_spec.minimum + 1],
        dtype=np.int64)
    self._masked_observation_spec = (env.observation_spec(), mask_spec)
    self._constant_mask = np.array(
        [[1] * (orig_action_spec.maximum - orig_action_spec.minimum + 1) +
         [0] * num_extra_actions] * self.batch_size)

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def observation_spec(self) -> types.NestedArraySpec:
    return self._masked_observation_spec

  def _add_mask_to_observation(self, timestep):
    return ts.TimeStep(
        timestep.step_type, timestep.reward, timestep.discount,
        (timestep.observation, self._constant_mask))

  def _step(self, action):
    return self._add_mask_to_observation(self._env.step(action))

  def _reset(self):
    return self._add_mask_to_observation(self._env.reset())
