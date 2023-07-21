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

"""Suite for loading MuJoCo Gym environments.

**NOTE**: Mujoco requires separated installation.

Follow the instructions at:

https://github.com/openai/mujoco-py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Optional, Sequence, Text

import gin
import gym
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.typing import types

_TRY_IMPORT = True  # pylint: disable=g-statement-before-imports

if _TRY_IMPORT:
  try:
    import mujoco_py  # pylint: disable=g-import-not-at-top
  except ImportError:
    mujoco_py = None
else:
  import mujoco_py  # pylint: disable=g-import-not-at-top


def is_available() -> bool:
  return mujoco_py is not None


@gin.configurable
def load(
    environment_name: Text,
    discount: types.Float = 1.0,
    max_episode_steps: Optional[types.Int] = None,
    gym_env_wrappers: Sequence[types.GymEnvWrapper] = (),
    env_wrappers: Sequence[types.PyEnvWrapper] = (),
    spec_dtype_map: Optional[Dict[gym.Space, np.dtype]] = None
) -> py_environment.PyEnvironment:
  """Loads the selected environment and wraps it with the specified wrappers.

  Note that by default a TimeLimit wrapper is used to limit episode lengths
  to the default benchmarks defined by the registered environments.

  Args:
    environment_name: Name for the environment to load.
    discount: Discount to use for the environment.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no timestep_limit set in the environment's spec.
    gym_env_wrappers: Iterable with references to wrapper classes to use
      directly on the gym environment.
    env_wrappers: Iterable with references to wrapper classes to use on the
      gym_wrapped environment.
    spec_dtype_map: A dict that maps gym specs to tf dtypes to use as the
      default dtype for the tensors. An easy way how to configure a custom
      mapping through Gin is to define a gin-configurable function that returns
      desired mapping and call it in your Gin config file, for example:
      `suite_gym.load.spec_dtype_map = @get_custom_mapping()`.

  Returns:
    A PyEnvironmentBase instance.
  """
  if spec_dtype_map is None:
    # Use float32 for Observations.
    spec_dtype_map = {gym.spaces.Box: np.float32}
  return suite_gym.load(environment_name, discount, max_episode_steps,
                        gym_env_wrappers, env_wrappers, spec_dtype_map)
