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

"""Tests for the Stationary Stochastic Per-Arm Bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.environments import stationary_stochastic_per_arm_py_environment as sspe
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec


def normal_with_sigma_1_sampler(mu):
  return np.random.normal(mu, 1)


def check_unbatched_time_step_spec(time_step, time_step_spec, batch_size):
  """Checks if time step conforms array spec, even if batched."""
  if batch_size is None:
    return array_spec.check_arrays_nest(time_step, time_step_spec)

  return array_spec.check_arrays_nest(
      time_step, array_spec.add_outer_dims_nest(time_step_spec, (batch_size,)))


class LinearNormalReward(object):

  def __init__(self, theta):
    self.theta = theta

  def __call__(self, x):
    mu = np.dot(x, self.theta)
    return np.random.normal(mu, 1)


class StationaryStochasticPerArmBanditPyEnvironmentTest(tf.test.TestCase,
                                                        parameterized.TestCase):

  def _check_arm_obs_spec(self, obs_spec, variable_action_method, num_actions,
                          arm_dim):
    if variable_action_method == bandit_spec_utils.VariableActionMethod.MASK:
      self.assertAllEqual(obs_spec[1].shape, [num_actions])
      obs_spec = obs_spec[0]

    if (variable_action_method ==
        bandit_spec_utils.VariableActionMethod.IN_BATCH_DIM):
      self.assertEqual(obs_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY].shape,
                       (1, arm_dim))
      return
    self.assertEqual(obs_spec[bandit_spec_utils.PER_ARM_FEATURE_KEY].shape,
                     (num_actions, arm_dim))
    if (variable_action_method ==
        bandit_spec_utils.VariableActionMethod.NUM_ACTIONS_FEATURE):
      self.assertEqual(
          obs_spec[bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY].shape, ())

  def test_with_uniform_context_and_normal_mu_reward(self):

    def _global_context_sampling_fn():
      return np.random.randint(-10, 10, [4])

    def _arm_context_sampling_fn():
      return np.random.randint(-2, 3, [5])

    reward_fn = LinearNormalReward([0, 1, 2, 3, 4, 5, 6, 7, 8])

    env = sspe.StationaryStochasticPerArmPyEnvironment(
        _global_context_sampling_fn,
        _arm_context_sampling_fn,
        6,
        reward_fn,
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

  @parameterized.parameters([
      (bandit_spec_utils.VariableActionMethod.MASK, 3),
      (bandit_spec_utils.VariableActionMethod.NUM_ACTIONS_FEATURE, 2),
      (bandit_spec_utils.VariableActionMethod.IN_BATCH_DIM, 1)
  ])
  def test_with_variable_num_actions(self, variable_action_method, batch_size):

    def _global_context_sampling_fn():
      return np.random.randint(-10, 10, [4])

    def _arm_context_sampling_fn():
      return np.random.randint(-2, 3, [5])

    def _num_actions_fn():
      return np.random.randint(5, 7)

    reward_fn = LinearNormalReward([0, 1, 2, 3, 4, 5, 6, 7, 8])

    env = sspe.StationaryStochasticPerArmPyEnvironment(
        _global_context_sampling_fn,
        _arm_context_sampling_fn,
        6,
        reward_fn,
        _num_actions_fn,
        batch_size=batch_size,
        variable_action_method=variable_action_method)

    time_step_spec = env.time_step_spec()
    self._check_arm_obs_spec(time_step_spec.observation,
                             variable_action_method, 6, 5)

    for _ in range(5):
      time_step = env.reset()
      actual_batch_size = time_step.step_type.shape[0]
      self.assertTrue(
          check_unbatched_time_step_spec(
              time_step=time_step,
              time_step_spec=time_step_spec,
              batch_size=actual_batch_size))

      action = np.random.randint(0, 4, [batch_size])
      time_step = env.step(action)


if __name__ == '__main__':
  tf.test.main()
