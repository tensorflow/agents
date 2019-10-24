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

"""Tests for tf_agents.environments.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing.absltest import mock
import numpy as np
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


def get_mock_env(action_spec, observation_spec, step_return):
  env = mock.MagicMock()

  env.observation_spec = lambda: observation_spec
  time_step_spec = ts.time_step_spec(observation_spec)
  env.time_step_spec = lambda: time_step_spec
  env.action_spec = lambda: action_spec
  env.step = lambda: step_return
  env.step.reset = lambda: step_return
  return env


class UtilsTest(test_utils.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    self._action_spec = [
        array_spec.BoundedArraySpec((1,), np.int32, -10, 10),
    ]

    self._observation_spec = array_spec.BoundedArraySpec((1,), np.int32, -10,
                                                         10)

  def testValidateOk(self):
    env = get_mock_env(self._action_spec, self._observation_spec, None)
    rng = np.random.RandomState()

    sample_fn = lambda: array_spec.sample_spec_nest(env.observation_spec(), rng)

    def step(unused_time_step):
      if rng.rand() < 0.10:
        return ts.termination(sample_fn(), 0.0)
      else:
        return ts.transition(sample_fn(), 1.0)

    env.step = step
    env.reset = lambda: ts.restart(sample_fn())

    utils.validate_py_environment(env, episodes=2)

  def testValidateNotATimeStep(self):
    env = get_mock_env(self._action_spec, self._observation_spec, None)

    with self.assertRaises(ValueError):
      utils.validate_py_environment(env, episodes=1)

  def testValidateWrongDType(self):
    env = get_mock_env(self._action_spec, self._observation_spec,
                       ts.restart(np.array([0], dtype=np.int64)))

    with self.assertRaisesRegexp(ValueError, "does not match expected"):
      utils.validate_py_environment(env, episodes=1)

  def testValidateWrongShape(self):
    env = get_mock_env(self._action_spec, self._observation_spec,
                       ts.restart(np.array([0, 1], dtype=np.int32)))

    with self.assertRaisesRegexp(ValueError, "does not match expected"):
      utils.validate_py_environment(env, episodes=1)

  def testValidateWrongDTypeAndShape(self):
    env = get_mock_env(self._action_spec, self._observation_spec,
                       ts.restart(np.array([0, 1], dtype=np.int64)))

    with self.assertRaisesRegexp(ValueError, "does not match expected"):
      utils.validate_py_environment(env, episodes=1)

  def testValidateOutOfBounds(self):
    env = get_mock_env(self._action_spec, self._observation_spec,
                       ts.restart(np.array([-11], dtype=np.int32)))

    with self.assertRaisesRegexp(ValueError, "does not match expected"):
      utils.validate_py_environment(env, episodes=1)

  def testValidateBoundedSpecDistinctBounds(self):
    observation_spec = array_spec.BoundedArraySpec((3,), np.int32,
                                                   [-10, -5, -2], [10, 5, 2])
    env = get_mock_env(self._action_spec, observation_spec, None)
    rng = np.random.RandomState()
    sample_fn = lambda: array_spec.sample_spec_nest(env.observation_spec(), rng)

    def step(unused_time_step):
      if rng.rand() < 0.10:
        return ts.termination(sample_fn(), 0.0)
      else:
        return ts.transition(sample_fn(), 1.0)

    env.step = step
    env.reset = lambda: ts.restart(sample_fn())
    utils.validate_py_environment(env, episodes=1)


if __name__ == "__main__":
  test_utils.main()
