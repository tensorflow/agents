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

"""Wrapper providing a PyEnvironmentBase adapter for Gymnasium environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Optional, Text, Union

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import tensorflow as tf
from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


def spec_from_gym_space(
    space: gym.Space,
    simplify_box_bounds: bool = True,
    name: Optional[Text] = None,
) -> Union[
    specs.BoundedArraySpec,
    specs.ArraySpec,
    tuple[specs.ArraySpec, ...],
    list[specs.ArraySpec],
    collections.OrderedDict[str, specs.ArraySpec],
]:
  """Converts gymnasium spaces into array specs, or a collection thereof.

  Please note:
    Unlike OpenAI's gym, Farama's gymnasium provides a dtype for
    each current implementation of spaces. dtype should be defined
    in all specific subclasses of gymnasium.Space even if it is still
    optional in the superclass.

  Args:
    space: gymnasium.Space to turn into a spec.
    simplify_box_bounds: Whether to replace bounds of Box space that are arrays
      with identical values with one number and rely on broadcasting.
    name: Name of the spec.

  Returns:
    A BoundedArraySpec or an ArraySpec nest mirroring the given space structure.
    The result can be a tuple, sequence or dict of specs for specific Spaces.
  Raises:
    ValueError: If there is an unknown space type.
  """

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
    return spec_from_gym_space(spec, simplify_box_bounds, nested_name)

  if isinstance(space, gym.spaces.Discrete):
    # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
    # are inclusive on their bounds.
    maximum = space.n - 1
    return specs.BoundedArraySpec(
        shape=(), dtype=np.int64, minimum=0, maximum=maximum, name=name
    )
  elif isinstance(space, gym.spaces.MultiDiscrete):
    dtype = np.integer
    maximum = try_simplify_array_to_value(
        np.asarray(space.nvec - 1, dtype=dtype)
    )
    return specs.BoundedArraySpec(
        shape=space.shape, dtype=dtype, minimum=0, maximum=maximum, name=name
    )
  elif isinstance(space, gym.spaces.MultiBinary):
    return specs.BoundedArraySpec(
        shape=space.shape, dtype=np.int8, minimum=0, maximum=1, name=name
    )
  elif isinstance(space, gym.spaces.Box):
    dtype = space.dtype
    minimum = np.asarray(space.low, dtype=dtype)
    maximum = np.asarray(space.high, dtype=dtype)
    if simplify_box_bounds:
      simple_minimum = try_simplify_array_to_value(minimum)
      simple_maximum = try_simplify_array_to_value(maximum)
      # Can only simplify if both bounds are simplified. Otherwise
      # broadcasting doesn't work from non-simplified to simplified.
      if simple_minimum.shape == simple_maximum.shape:
        minimum = simple_minimum
        maximum = simple_maximum
    return specs.BoundedArraySpec(
        shape=space.shape,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
        name=name,
    )
  elif isinstance(space, gym.spaces.Tuple):
    return tuple(
        [nested_spec(s, 'tuple_%d' % i) for i, s in enumerate(space.spaces)]
    )
  elif isinstance(space, gym.spaces.Dict):
    return collections.OrderedDict(
        [(key, nested_spec(s, key)) for key, s in space.spaces.items()]
    )
  elif isinstance(space, gym.spaces.Sequence):
    return list([nested_spec(space.feature_space, 'nested_space')])
  elif isinstance(space, gym.spaces.Graph):
    return (
        nested_spec(space.node_space, 'node_space'),
        nested_spec(space.edge_space, 'edge_space'),
    )
  elif isinstance(space, gym.spaces.Text):
    return specs.ArraySpec(shape=space.shape, dtype=tf.string, name=name)
  else:
    raise ValueError(
        'The gymnasium space {} is currently not supported.'.format(space)
    )


class GymnasiumWrapper(py_environment.PyEnvironment):
  """Base wrapper implementing PyEnvironmentBaseWrapper interface for Gymnasium envs.

  Action and observation specs are automatically generated from the action and
  observation spaces. See base class for py_environment.Base details.
  """

  def __init__(
      self,
      gym_env: gym.Env,
      discount: types.Float = 1.0,
      auto_reset: bool = True,
      simplify_box_bounds: bool = True,
  ):
    super(GymnasiumWrapper, self).__init__(auto_reset)

    self._gym_env = gym_env
    self._discount = discount
    self._action_is_discrete = isinstance(
        self._gym_env.action_space, gym.spaces.Discrete
    )
    self._observation_spec = spec_from_gym_space(
        self._gym_env.observation_space,
        simplify_box_bounds,
        'observation',
    )
    self._action_spec = spec_from_gym_space(
        self._gym_env.action_space,
        simplify_box_bounds,
        'action',
    )
    self._flat_obs_spec = tf.nest.flatten(self._observation_spec)
    self._info = None
    self._truncated = True

  @property
  def gym(self) -> gym.Env:
    return self._gym_env

  def __getattr__(self, name: Text) -> Any:
    """Forward all other calls to the base environment."""
    gym_env = super(GymnasiumWrapper, self).__getattribute__('_gym_env')
    return getattr(gym_env, name)

  def get_info(self) -> Any:
    """Returns the gym environment info returned on the last step."""
    return self._info

  def _reset(self):
    # Upcoming update on gym adds **kwargs on reset. Update this to
    # support that.
    observation, self._info = self._gym_env.reset()
    self._terminated = False
    self._truncated = False
    return ts.restart(observation)

  @property
  def terminated(self) -> bool:
    return self._terminated

  @property
  def truncated(self) -> bool:
    return self._truncated

  def _step(self, action):
    # Some environments (e.g. FrozenLake) use the action as a key to the
    # transition probability, so it has to be hashable. In the case of discrete
    # actions we have a numpy scalar (e.g array(2)) which is not hashable
    # in this case, we simply pull out the scalar value which will be hashable.
    if self._action_is_discrete and isinstance(action, np.ndarray):
      action = action.item()

    # Figure out how tuple or dict actions will be generated by the
    # agents and if we can pass them through directly to gym.
    observation, reward, self._terminated, self._truncated, self._info = (
        self._gym_env.step(action)
    )

    if self._terminated:
      return ts.termination(observation, reward)
    elif self._truncated:
      return ts.truncation(observation, reward, self._discount)
    else:
      return ts.transition(observation, reward, self._discount)

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def close(self) -> None:
    return self._gym_env.close()

  def seed(self, seed: types.Seed) -> types.Seed:
    np_random, seed = seeding.np_random(seed)
    self._gym_env.np_random = np_random
    return seed

  def render(self, mode: Text = 'rgb_array') -> Any:
    return (
        self._gym_env.render()
    )  # mode should be set for key "render_mode" in make()

  # pytype: disable=attribute-error
  def set_state(self, state: Any) -> None:
    return self._gym_env.set_state(state)

  def get_state(self) -> Any:
    return self._gym.get_state()

  # pytype: enable=attribute-error
