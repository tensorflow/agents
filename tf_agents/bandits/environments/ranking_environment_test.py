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

"""Tests for the Ranking environment."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.environments import ranking_environment
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


class RankingPyEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([{
      'batch_size': 1,
      'global_dim': 4,
      'item_dim': 5,
      'num_items': 7,
      'num_slots': 5,
      'feedback_model': ranking_environment.FeedbackModel.CASCADING,
      'click_model': ranking_environment.ClickModel.GHOST_ACTIONS,
      'real_cascade': False,
  }, {
      'batch_size': 4,
      'global_dim': 5,
      'item_dim': 3,
      'num_items': 8,
      'num_slots': 6,
      'feedback_model': ranking_environment.FeedbackModel.CASCADING,
      'click_model': ranking_environment.ClickModel.DISTANCE_BASED,
      'real_cascade': True,
  }, {
      'batch_size': 8,
      'global_dim': 12,
      'item_dim': 4,
      'num_items': 23,
      'num_slots': 9,
      'feedback_model': ranking_environment.FeedbackModel.SCORE_VECTOR,
      'click_model': ranking_environment.ClickModel.DISTANCE_BASED,
      'real_cascade': False,

  }])
  def test_ranking_environment(self, batch_size, global_dim, item_dim,
                               num_items, num_slots, feedback_model,
                               click_model, real_cascade):

    def _global_sampling_fn():
      return np.random.randint(-10, 10, [global_dim])

    def _item_sampling_fn():
      return np.random.randint(-2, 3, [item_dim])

    scores_weight_matrix = (np.reshape(
        np.arange(global_dim * item_dim, dtype=float),
        newshape=[item_dim, global_dim]) - 10) / 5

    env = ranking_environment.RankingPyEnvironment(
        _global_sampling_fn,
        _item_sampling_fn,
        num_items=num_items,
        num_slots=num_slots,
        scores_weight_matrix=scores_weight_matrix,
        feedback_model=feedback_model,
        click_model=click_model,
        real_cascade=real_cascade,
        distance_threshold=10.0,
        batch_size=batch_size)
    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()

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
      self.assertAllEqual(action.shape, [batch_size, num_slots])
      self.assertAllGreaterEqual(action, 0)
      time_step = env.step(action)
      reward = time_step.reward
      if feedback_model == ranking_environment.FeedbackModel.CASCADING:
        self.assertAllEqual(reward['chosen_index'].shape, [batch_size])
        self.assertAllGreaterEqual(reward['chosen_index'], 0)
        self.assertAllEqual(reward['chosen_value'].shape, [batch_size])
      else:
        self.assertAllEqual(reward.shape, [batch_size, num_slots])

  def test_cascading_to_scorevector(self):
    batch_size = 5
    global_dim = 12
    item_dim = 4
    num_items = 23
    num_slots = 9
    def _global_sampling_fn():
      return np.random.randint(-10, 10, [global_dim])

    def _item_sampling_fn():
      return np.random.randint(-2, 3, [item_dim])
    scores_weight_matrix = (np.reshape(
        np.arange(global_dim * item_dim, dtype=float),
        newshape=[item_dim, global_dim]) - 10) / 5
    env = ranking_environment.RankingPyEnvironment(
        _global_sampling_fn,
        _item_sampling_fn,
        num_items=num_items,
        num_slots=num_slots,
        scores_weight_matrix=scores_weight_matrix,
        feedback_model=ranking_environment.FeedbackModel.SCORE_VECTOR,
        click_model=ranking_environment.ClickModel.DISTANCE_BASED,
        distance_threshold=10.0,
        batch_size=batch_size)

    chosen_items = np.array([0, 2, 9, 1, 2])
    chosen_values = np.array([6, 8, 4, 2, 3])
    score_vector = env._cascading_to_scorevector(chosen_items, chosen_values)
    self.assertAllEqual(score_vector.shape, [batch_size, num_slots])

    # The third row is all zeros because `chosen_item == 9` means no click.
    self.assertAllEqual(score_vector, [[6, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 8, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 2, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 3, 0, 0, 0, 0, 0, 0]])


class ExplicitBiasEnvironmentTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([{
      'batch_size': 1,
      'global_dim': 4,
      'item_dim': 5,
      'num_items': 7,
      'num_slots': 5,
  }, {
      'batch_size': 8,
      'global_dim': 12,
      'item_dim': 4,
      'num_items': 23,
      'num_slots': 9,
  }])
  def test_explicit_bias_environment(self, batch_size, global_dim, item_dim,
                                     num_items, num_slots):
    def _global_sampling_fn():
      return np.random.randint(-10, 10, [global_dim])

    def _item_sampling_fn():
      return np.random.randint(-2, 3, [item_dim])

    def _relevance_fn(global_obs, item_obs):
      min_dim = min(global_dim, item_dim)
      dot_prod = np.dot(global_obs[:min_dim],
                        item_obs[:min_dim]).astype(np.float32)
      return 1 / (1 + np.exp(-dot_prod))

    positional_biases = list(0.75 - np.arange(num_slots) / (2 * num_slots))
    env = ranking_environment.ExplicitPositionalBiasRankingEnvironment(
        _global_sampling_fn,
        _item_sampling_fn,
        _relevance_fn,
        num_items,
        positional_biases,
        batch_size,
    )

    time_step_spec = env.time_step_spec()
    action_spec = env.action_spec()

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
      self.assertAllEqual(action.shape, [batch_size, num_slots])
      self.assertAllGreaterEqual(action, 0)
      time_step = env.step(action)
      reward = time_step.reward
      self.assertAllEqual(reward.shape, [batch_size, num_slots])


if __name__ == '__main__':
  tf.test.main()
