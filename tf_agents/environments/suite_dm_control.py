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

"""Suite for loading DeepMind Control Suite environments.

Follow these instructions to install it:

https://github.com/deepmind/dm_control#installation-and-requirements

"""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Sequence, Text

import gin
from tf_agents.environments import dm_control_wrapper
from tf_agents.environments import py_environment
from tf_agents.typing import types

_TRY_IMPORT = True  # pylint: disable=g-statement-before-imports

if _TRY_IMPORT:
  try:
    from dm_control import suite  # pylint: disable=g-import-not-at-top
    from dm_control.suite.wrappers import pixels  # pylint: disable=g-import-not-at-top
  except ImportError:
    suite = None
else:
  from dm_control import suite  # pylint: disable=g-import-not-at-top
  from dm_control.suite.wrappers import pixels  # pylint: disable=g-import-not-at-top


def is_available() -> bool:
  return suite is not None


def _load_env(domain_name: Text,
              task_name: Text,
              task_kwargs=None,
              environment_kwargs=None,
              visualize_reward: bool = False):
  """Loads a DM environment.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.

  Returns:
    The requested environment.

  Raises:
    ImportError: if dm_control module was not available.
  """

  if not is_available():
    raise ImportError('dm_control module is not available.')
  return suite.load(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward)


@gin.configurable
def load(
    domain_name: Text,
    task_name: Text,
    task_kwargs=None,
    environment_kwargs=None,
    visualize_reward: bool = False,
    render_kwargs=None,
    env_wrappers: Sequence[types.PyEnvWrapper] = ()
) -> py_environment.PyEnvironment:
  """Returns an environment from a domain name, task name and optional settings.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
    render_kwargs: Optional `dict` of keyword arguments for rendering.
    env_wrappers: Iterable with references to wrapper classes to use on the
      wrapped environment.

  Returns:
    The requested environment.

  Raises:
    ImportError: if dm_control module was not available.
  """
  dm_env = _load_env(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward)

  env = dm_control_wrapper.DmControlWrapper(dm_env, render_kwargs)

  for wrapper in env_wrappers:
    env = wrapper(env)

  return env


@gin.configurable
def load_pixels(
    domain_name: Text,
    task_name: Text,
    observation_key: Text = 'pixels',
    pixels_only: bool = True,
    task_kwargs=None,
    environment_kwargs=None,
    visualize_reward: bool = False,
    render_kwargs=None,
    env_wrappers: Sequence[types.PyEnvWrapper] = ()
) -> py_environment.PyEnvironment:
  """Returns an environment from a domain name, task name and optional settings.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    observation_key: Optional custom string specifying the pixel observation's
      key in the `OrderedDict` of observations. Defaults to 'pixels'.
    pixels_only: If True (default), the original set of 'state' observations
      returned by the wrapped environment will be discarded, and the
      `OrderedDict` of observations will only contain pixels. If False, the
      `OrderedDict` will contain the original observations as well as the pixel
      observations.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
    render_kwargs: Optional `dict` of keyword arguments for rendering.
    env_wrappers: Iterable with references to wrapper classes to use on the
      wrapped environment.

  Returns:
    The requested environment.

  Raises:
    ImportError: if dm_control module was not available.
  """
  dm_env = _load_env(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward)

  dm_env = pixels.Wrapper(
      dm_env,
      pixels_only=pixels_only,
      render_kwargs=render_kwargs,
      observation_key=observation_key)
  env = dm_control_wrapper.DmControlWrapper(dm_env, render_kwargs)

  for wrapper in env_wrappers:
    env = wrapper(env)

  return env
