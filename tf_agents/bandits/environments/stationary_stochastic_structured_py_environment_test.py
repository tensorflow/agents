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

"""Tests for the Stationary Stochastic Structured Bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.environments import stationary_stochastic_structured_py_environment as ssspe
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec


def check_unbatched_time_step_spec(time_step, time_step_spec, batch_size):
  """Checks if time step conforms array spec, even if batched."""
  if batch_size is None:
    return array_spec.check_arrays_nest(time_step, time_step_spec)

  return array_spec.check_arrays_nest(
      time_step, array_spec.add_outer_dims_nest(time_step_spec, (batch_size,)))


class StationaryStochasticStructuredBanditPyEnvironmentTest(tf.test.TestCase):

  def test_with_random_policy(self):

    def _global_context_sampling_fn():
      abc = np.array(['a', 'b', 'c'])
      return {'global1': np.random.randint(-2, 3, [3, 4]),
              'global2': abc[np.random.randint(0, 2, [1])]}

    def _arm_context_sampling_fn():
      aabbcc = np.array(['aa', 'bb', 'cc'])
      return {'arm1': np.random.randint(-3, 4, [5]),
              'arm2': np.random.randint(-3, 4, [3, 1]),
              'arm3': aabbcc[np.random.randint(0, 2, [1])]}

    def _reward_fn(global_obs, arm_obs):
      return global_obs['global1'][2, 1] + arm_obs['arm1'][4]

    env = ssspe.StationaryStochasticStructuredPyEnvironment(
        _global_context_sampling_fn,
        _arm_context_sampling_fn,
        6,
        _reward_fn,
        batch_size=2)
    time_step_spec = env.time_step_spec()
    action_spec = array_spec.BoundedArraySpec(
        shape=(), minimum=0, maximum=5, dtype=np.int32)

    random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    for _ in range(5):
      time_step = env.reset()
      self.assertTrue(
          check_unbatched_time_step_spec(
              time_step=time_step,
              time_step_spec=time_step_spec,
              batch_size=env.batch_size))

      action = random_policy.action(time_step).action
      self.assertAllEqual(action.shape, [2])
      self.assertAllGreaterEqual(action, 0)
      self.assertAllLess(action, 6)
      time_step = env.step(action)
      self.assertEqual(time_step.reward.shape, (2,))


if __name__ == '__main__':
  tf.test.main()
