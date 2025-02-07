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
from __future__ import print_function

from typing import Optional, Union

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec
from tf_agents.typing import types


def get_tf_env(
    environment: Union[
        py_environment.PyEnvironment, tf_environment.TFEnvironment
    ]
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
        '`tf_environment.TFEnvironment` or `py_environment.PyEnvironment`.'
        % environment
    )
  return tf_env


def validate_py_environment(
    environment: py_environment.PyEnvironment,
    episodes: int = 5,
    observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
    action_constraint_suffix: Optional[str] = None,
):
  """Validates the environment follows the defined specs.

  Args:
    environment: The environment to test.
    episodes: The number of episodes to run a random policy over.
    observation_and_action_constraint_splitter: A function used to process
      observations with action constraints. These constraints can indicate,
      for example, a mask of valid/invalid actions for a given state of the
      environment. The function takes in a full observation and returns a
      tuple consisting of 1) the part of the observation intended as input to
      the network and 2) the constraint. An example
      `observation_and_action_constraint_splitter` could be as simple as: ```
      def observation_and_action_constraint_splitter(observation): return
      observation['network_input'], observation['constraint'] ``` *Note*: when
      using `observation_and_action_constraint_splitter`, make sure the
      provided `q_network` is compatible with the network-specific half of the
      output of the `observation_and_action_constraint_splitter`. In
      particular, `observation_and_action_constraint_splitter` will be called
      on the observation before passing to the network. If
      `observation_and_action_constraint_splitter` is None, action constraints
      are not applied.
    action_constraint_suffix: Alternative to
      observation_and_action_constraint_splitter. If provided, the observation
      is expected to contain action constraints with the given suffix
      corresponding to actions in the action spec. The policy will sample
      respecting the constraints for those actions.
  """
  time_step_spec = environment.time_step_spec()
  action_spec = environment.action_spec()
  random_policy = random_py_policy.RandomPyPolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      observation_and_action_constraint_splitter=(
          observation_and_action_constraint_splitter
      ),
      action_constraint_suffix=action_constraint_suffix,
  )

  if environment.batch_size is not None:
    batched_time_step_spec = array_spec.add_outer_dims_nest(
        time_step_spec, outer_dims=(environment.batch_size,)
    )
  else:
    batched_time_step_spec = time_step_spec

  episode_count = 0
  time_step = environment.reset()

  while episode_count < episodes:
    if not array_spec.check_arrays_nest(time_step, batched_time_step_spec):
      raise ValueError(
          'Given `time_step`: %r does not match expected `time_step_spec`: %r'
          % (time_step, batched_time_step_spec)
      )

    action = random_policy.action(time_step).action
    time_step = environment.step(action)

    episode_count += np.sum(time_step.is_last())
