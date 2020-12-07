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

"""Suite for loading bsuite gym environments.

Osband et al., Behaviour Suite for Reinforcement Learning, 2019.
https://github.com/deepmind/bsuite/

Follow https://github.com/deepmind/bsuite#getting-started to install bsuite
"""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text
import gin
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym

_TRY_IMPORT = True  # pylint: disable=g-statement-before-imports

# pylint: disable=g-import-not-at-top
if _TRY_IMPORT:
  try:
    from bsuite import bsuite
    from bsuite.utils import gym_wrapper
  except ImportError:
    bsuite = None
else:
  from bsuite import bsuite
  from bsuite.utils import gym_wrapper
# pylint: enable=g-import-not-at-top


def is_available() -> bool:
  return bsuite is not None


@gin.configurable
def load(bsuite_id: Text,
         record: bool = True,
         save_path: Optional[Text] = None,
         logging_mode: Text = 'csv',
         overwrite: bool = False) -> py_environment.PyEnvironment:
  """Loads the selected environment.

  Args:
    bsuite_id: a bsuite_id specifies a bsuite experiment. For an example
      `bsuite_id` "deep_sea/7" will be 7th level of the "deep_sea" task.
    record: whether to log bsuite results.
    save_path: the directory to save bsuite results.
    logging_mode: which form of logging to use for bsuite results
      ['csv', 'sqlite', 'terminal'].
    overwrite: overwrite csv logging if found.

  Returns:
    A PyEnvironment instance.
  """
  if record:
    raw_env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=save_path,
        logging_mode=logging_mode,
        overwrite=overwrite)
  else:
    raw_env = bsuite.load_from_id(bsuite_id=bsuite_id)
  gym_env = gym_wrapper.GymFromDMEnv(raw_env)
  return suite_gym.wrap_env(gym_env)
