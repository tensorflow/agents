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

"""Tests for tf_agents.environments.py_environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.environments import random_py_environment
from tf_agents.specs import array_spec


class PyEnvironmentTest(tf.test.TestCase):

  def testResetSavesCurrentTimeStep(self):
    obs_spec = array_spec.BoundedArraySpec((1,), np.int32)
    action_spec = array_spec.BoundedArraySpec((1,), np.int32)

    random_env = random_py_environment.RandomPyEnvironment(
        observation_spec=obs_spec, action_spec=action_spec)

    time_step = random_env.reset()
    current_time_step = random_env.current_time_step()
    tf.nest.map_structure(self.assertAllEqual, time_step, current_time_step)

  def testStepSavesCurrentTimeStep(self):
    obs_spec = array_spec.BoundedArraySpec((1,), np.int32)
    action_spec = array_spec.BoundedArraySpec((1,), np.int32)

    random_env = random_py_environment.RandomPyEnvironment(
        observation_spec=obs_spec, action_spec=action_spec)

    random_env.reset()
    time_step = random_env.step(action=np.ones((1,)))
    current_time_step = random_env.current_time_step()
    tf.nest.map_structure(self.assertAllEqual, time_step, current_time_step)


if __name__ == '__main__':
  tf.test.main()
