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

"""Wrapper providing a PyEnvironmentBase adapter for Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gym
import gym.spaces
import numpy as np
import tensorflow as tf

from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


def spec_from_gym_space(space,
                        dtype_map=None,
                        simplify_box_bounds=True,
                        name=None):
  """Converts gym spaces into array specs.

  Gym does not properly define dtypes for spaces. By default all spaces set
  their type to float64 even though observations do not always return this type.
  See:
  https://github.com/openai/gym/issues/527

  To handle this we allow a dtype_map for setting default types for mapping
  spaces to specs.

  TODO(oars): Support using different dtypes for different parts of the
  observations. Not sure that we have a need for this yet.

  Args:
    space: gym.Space to turn into a spec.
    dtype_map: A dict from specs to dtypes to use as the default dtype.
    simplify_box_bounds: Whether to replace bounds of Box space that are arrays
      with identical values with one number and rely on broadcasting.
    name: Name of the spec.

  Returns:
    A BoundedArraySpec nest mirroring the given space structure.
  Raises:
    ValueError: If there is an unknown space type.
  """
  if dtype_map is None:
    dtype_map = {}

  # We try to simplify redundant arrays to make logging and debugging less
  # verbose and easier to read since the printed spec bounds may be large.
  def try_simplify_array_to_value(np_array):
    """If given numpy array has all the same values, returns that value."""
    first_value = np_array.item(0)
    if np.all(np_array == first_value):
      return np.array(first_value, dtype=np_array.dtype)
    else:
      return np_array

  def nested_spec(spec, child_name):
    """Returns the nested spec with a unique name."""
    nested_name = name + '/' + child_name if name else child_name
    return spec_from_gym_space(spec, dtype_map, simplify_box_bounds,
                               nested_name)

  if isinstance(space, gym.spaces.Discrete):
    # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
    # are inclusive on their bounds.
    maximum = space.n - 1
    # TODO(oars): change to use dtype in space once Gym is updated.
    dtype = dtype_map.get(gym.spaces.Discrete, np.int64)
    return specs.BoundedArraySpec(
        shape=(), dtype=dtype, minimum=0, maximum=maximum, name=name)
  elif isinstance(space, gym.spaces.MultiDiscrete):
    dtype = dtype_map.get(gym.spaces.MultiDiscrete, np.int32)
    maximum = try_simplify_array_to_value(
        np.asarray(space.nvec - 1, dtype=dtype))
    return specs.BoundedArraySpec(
        shape=space.shape, dtype=dtype, minimum=0, maximum=maximum, name=name)
  elif isinstance(space, gym.spaces.MultiBinary):
    dtype = dtype_map.get(gym.spaces.MultiBinary, np.int8)
    shape = (space.n,)
    return specs.BoundedArraySpec(
        shape=shape, dtype=dtype, minimum=0, maximum=1, name=name)
  elif isinstance(space, gym.spaces.Box):
    if hasattr(space, 'dtype') and gym.spaces.Box not in dtype_map:
      dtype = space.dtype
    else:
      dtype = dtype_map.get(gym.spaces.Box, np.float32)
    minimum = np.asarray(space.low, dtype=dtype)
    maximum = np.asarray(space.high, dtype=dtype)
    if simplify_box_bounds:
      minimum = try_simplify_array_to_value(minimum)
      maximum = try_simplify_array_to_value(maximum)
    return specs.BoundedArraySpec(
        shape=space.shape,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
        name=name)
  elif isinstance(space, gym.spaces.Tuple):
    return tuple(
        [nested_spec(s, 'tuple_%d' % i) for i, s in enumerate(space.spaces)])
  elif isinstance(space, gym.spaces.Dict):
    return collections.OrderedDict([
        (key, nested_spec(s, key)) for key, s in space.spaces.items()
    ])
  else:
    raise ValueError(
        'The gym space {} is currently not supported.'.format(space))


class GymWrapper(py_environment.PyEnvironment):
  """Base wrapper implementing PyEnvironmentBaseWrapper interface for Gym envs.

  Action and observation specs are automatically generated from the action and
  observation spaces. See base class for py_environment.Base details.
  """

  def __init__(self,
               gym_env,
               discount=1.0,
               spec_dtype_map=None,
               match_obs_space_dtype=True,
               auto_reset=True,
               simplify_box_bounds=True):
    super(GymWrapper, self).__init__()

    self._gym_env = gym_env
    self._discount = discount
    self._action_is_discrete = isinstance(self._gym_env.action_space,
                                          gym.spaces.Discrete)
    self._match_obs_space_dtype = match_obs_space_dtype
    # TODO(sfishman): Add test for auto_reset param.
    self._auto_reset = auto_reset
    self._observation_spec = spec_from_gym_space(
        self._gym_env.observation_space, spec_dtype_map, simplify_box_bounds,
        'observation')
    self._action_spec = spec_from_gym_space(self._gym_env.action_space,
                                            spec_dtype_map, simplify_box_bounds,
                                            'action')
    self._flat_obs_spec = tf.nest.flatten(self._observation_spec)
    self._info = None
    self._done = True

  @property
  def gym(self):
    return self._gym_env

  def __getattr__(self, name):
    """Forward all other calls to the base environment."""
    return getattr(self._gym_env, name)

  def get_info(self):
    """Returns the gym environment info returned on the last step."""
    return self._info

  def _reset(self):
    # TODO(oars): Upcoming update on gym adds **kwargs on reset. Update this to
    # support that.
    observation = self._gym_env.reset()
    self._info = None
    self._done = False

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)
    return ts.restart(observation)

  @property
  def done(self):
    return self._done

  def _step(self, action):
    # Automatically reset the environments on step if they need to be reset.
    if self._auto_reset and self._done:
      return self.reset()

    # TODO(oars): Figure out how tuple or dict actions will be generated by the
    # agents and if we can pass them through directly to gym.

    observation, reward, self._done, self._info = self._gym_env.step(action)

    if self._match_obs_space_dtype:
      observation = self._to_obs_space_dtype(observation)

    if self._done:
      return ts.termination(observation, reward)
    else:
      return ts.transition(observation, reward, self._discount)

  def _to_obs_space_dtype(self, observation):
    """Make sure observation matches the specified space.

    Observation spaces in gym didn't have a dtype for a long time. Now that they
    do there is a large number of environments that do not follow the dtype in
    the space definition. Since we use the space definition to create the
    tensorflow graph we need to make sure observations match the expected
    dtypes.

    Args:
      observation: Observation to match the dtype on.

    Returns:
      The observation with a dtype matching the observation spec.
    """
    # Make sure we handle cases where observations are provided as a list.
    flat_obs = nest.flatten_up_to(self._observation_spec, observation)

    matched_observations = []
    for spec, obs in zip(self._flat_obs_spec, flat_obs):
      matched_observations.append(np.asarray(obs, dtype=spec.dtype))
    return tf.nest.pack_sequence_as(self._observation_spec,
                                    matched_observations)

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def close(self):
    return self._gym_env.close()

  def seed(self, seed):
    return self._gym_env.seed(seed)

  def render(self, mode='rgb_array'):
    return self._gym_env.render(mode)
