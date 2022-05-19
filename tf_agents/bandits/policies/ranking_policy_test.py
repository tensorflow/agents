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

"""Tests for ranking_policy."""
from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.networks import global_and_arm_feature_network as arm_net
from tf_agents.bandits.policies import ranking_policy
from tf_agents.specs import bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class RankingPolicyTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(batch_size=1, num_items=20, num_slots=5),
                            dict(batch_size=3, num_items=15, num_slots=15),
                            dict(batch_size=30, num_items=115, num_slots=100))
  def testPolicy(self, batch_size, num_items, num_slots):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        7, 5, num_items)
    time_step_spec = ts.time_step_spec(obs_spec)
    network = arm_net.create_feed_forward_common_tower_network(
        obs_spec, [3], [4], [5])

    policy = ranking_policy.PenalizeCosineDistanceRankingPolicy(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network)
    observation = tensor_spec.sample_spec_nest(
        obs_spec, outer_dims=[batch_size], minimum=-1, maximum=1)
    time_spec = ts.restart(observation, batch_size=batch_size)
    action_step = policy.action(time_spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(action_step.action.shape, [batch_size, num_slots])

  def testTemperature(self):
    if not tf.executing_eagerly():
      self.skipTest('This test is only run in eager mode.')
    batch_size = 1
    num_items = 20
    num_slots = 4
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        7, 5, num_items)
    time_step_spec = ts.time_step_spec(obs_spec)
    network = arm_net.create_feed_forward_common_tower_network(
        obs_spec, [3], [4], [5])

    low_temp_policy = ranking_policy.NoPenaltyRankingPolicy(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network,
        logits_temperature=0.001)
    high_temp_policy = ranking_policy.NoPenaltyRankingPolicy(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network,
        logits_temperature=1000.)
    observation = tensor_spec.sample_spec_nest(
        obs_spec, outer_dims=[batch_size], minimum=-1, maximum=1)
    time_step = ts.restart(observation, batch_size=batch_size)
    low_temp_first_items = tf.stack(
        [low_temp_policy.action(time_step).action[0][0] for _ in range(30)])
    num_low_temp_items = tf.shape(tf.unique(low_temp_first_items)[0])[0]

    high_temp_first_items = tf.stack(
        [high_temp_policy.action(time_step).action[0][0] for _ in range(30)])
    num_high_temp_items = tf.shape(tf.unique(high_temp_first_items)[0])[0]
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # The high temperature policy is more random, so when called repeatedly, it
    # chooses more diverse items for the first slot. Hence, the number of unique
    # elements will be more.
    self.assertLess(num_low_temp_items, num_high_temp_items)

  @parameterized.parameters(dict(batch_size=1, num_items=20, num_slots=5),
                            dict(batch_size=3, num_items=15, num_slots=15))
  def testNumActionsPolicy(self, batch_size, num_items, num_slots):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        7,
        5,
        num_items,
        add_num_actions_feature=True)
    time_step_spec = ts.time_step_spec(obs_spec)
    network = arm_net.create_feed_forward_common_tower_network(
        obs_spec, [3], [4], [5])

    policy = ranking_policy.PenalizeCosineDistanceRankingPolicy(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network,
        penalty_mixture_coefficient=0.3)
    observation = tensor_spec.sample_spec_nest(
        obs_spec, outer_dims=[batch_size])
    time_spec = ts.restart(observation, batch_size=batch_size)
    action_step = policy.action(time_spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(action_step.action.shape, [batch_size, num_slots])

  @parameterized.parameters(dict(batch_size=1, num_items=20, num_slots=5),
                            dict(batch_size=3, num_items=15, num_slots=15))
  def testDescendingScorePolicy(self, batch_size, num_items, num_slots):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        7, 5, num_items)
    time_step_spec = ts.time_step_spec(obs_spec)
    network = arm_net.create_feed_forward_common_tower_network(
        obs_spec, [3], [4], [5])

    policy = ranking_policy.DescendingScoreRankingPolicy(
        num_items=num_items,
        num_slots=num_slots,
        time_step_spec=time_step_spec,
        network=network)
    observation = tensor_spec.sample_spec_nest(
        obs_spec, outer_dims=[batch_size], minimum=-1, maximum=1)
    time_spec = ts.restart(observation, batch_size=batch_size)
    action_step = policy.action(time_spec)
    self.assertAllEqual(action_step.action.shape, [batch_size, num_slots])
    # Check that the policy is deterministic.
    action_step_again = policy.action(time_spec)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllEqual(action_step.action, action_step_again.action)

if __name__ == '__main__':
  tf.test.main()
