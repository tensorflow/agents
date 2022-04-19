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

"""Tests for tf_agents.train.utils.spec_utils."""

from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.environments import suite_gym
from tf_agents.train.utils import spec_utils
from tf_agents.utils import test_utils


class SpecUtilsTest(test_utils.TestCase):

  def test_get_tensor_specs(self):
    collect_env = suite_gym.load('CartPole-v0')
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))
    self.assertEqual(observation_spec.name, 'observation')
    self.assertEqual(action_spec.name, 'action')
    self.assertEqual(time_step_spec.observation.name, 'observation')
    self.assertEqual(time_step_spec.reward.name, 'reward')

  def test_get_collect_data_spec(self):
    env = suite_gym.load('CartPole-v0')
    policy = driver_test_utils.PyPolicyMock(env.time_step_spec(),
                                            env.action_spec())
    collect_spec = spec_utils.get_collect_data_spec_from_policy_and_env(env,
                                                                        policy)
    self.assertEqual(collect_spec.observation.name, 'observation')
    self.assertEqual(collect_spec.reward.name, 'reward')

if __name__ == '__main__':
  test_utils.main()
