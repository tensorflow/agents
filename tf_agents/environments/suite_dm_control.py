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

"""Suite for loading DeepMind Control Suite environments.

Follow these instructions to install it:

https://github.com/deepmind/dm_control#installation-and-requirements

"""

_TRY_IMPORT = True  # pylint: disable=g-statement-before-imports

if _TRY_IMPORT:
  try:
    from dm_control import suite  # pylint: disable=g-import-not-at-top
  except ImportError:
    suite = None
else:
  from dm_control import suite  # pylint: disable=g-import-not-at-top

import gin
from tf_agents.environments import dm_control_wrapper


def is_available():
  return suite is not None


@gin.configurable
def load(domain_name,
         task_name,
         task_kwargs=None,
         visualize_reward=False,
         render_kwargs=None,
         env_wrappers=()):
  """Returns an environment from a domain name, task name and optional settings.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
    render_kwargs: Optional `dict` of keyword arguments for rendering.
    env_wrappers: Iterable with references to wrapper classes to use on the
      gym_wrapped environment.

  Returns:
    The requested environment.

  Raises:
    ImportError: if dm_control module was not available.
  """
  if not is_available():
    raise ImportError("dm_control module is not available.")
  dm_env = suite.load(domain_name, task_name, task_kwargs, visualize_reward)
  env = dm_control_wrapper.DmControlWrapper(dm_env, render_kwargs)

  for wrapper in env_wrappers:
    env = wrapper(env)

  return env
