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

"""Common utilities for TF-Agents Environments."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Union

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec


def get_tf_env(
    environment: Union[py_environment.PyEnvironment,
                       tf_environment.TFEnvironment]
) -> tf_environment.TFEnvironment:
  """Ensures output is a tf_environment, wrapping py_environments if needed."""
  if environment is None:
    raise ValueError('`environment` cannot be None')
  if isinstance(environment, py_environment.PyEnvironment):
    tf_env = tf_py_environment.TFPyEnvironment(environment)
  elif isinstance(environment, tf_environment.TFEnvironment):
    tf_env = environment
  else:
    raise ValueError(
        '`environment` %s must be an instance of '
        '`tf_environment.TFEnvironment` or `py_environment.PyEnvironment`.' %
        environment)
  return tf_env


def validate_py_environment(environment: py_environment.PyEnvironment,
                            episodes: int = 5):
  """Validates the environment follows the defined specs."""
  time_step_spec = environment.time_step_spec()
  action_spec = environment.action_spec()

  random_policy = random_py_policy.RandomPyPolicy(
      time_step_spec=time_step_spec, action_spec=action_spec)

  episode_count = 0
  time_step = environment.reset()

  while episode_count < episodes:
    if not array_spec.check_arrays_nest(time_step, time_step_spec):
      raise ValueError(
          'Given `time_step`: %r does not match expected `time_step_spec`: %r' %
          (time_step, time_step_spec))

    action = random_policy.action(time_step).action
    time_step = environment.step(action)

    if time_step.is_last():
      episode_count += 1
      time_step = environment.reset()
