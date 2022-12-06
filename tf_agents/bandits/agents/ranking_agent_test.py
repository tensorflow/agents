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

"""Tests for ranking_agent.py."""

from absl.testing import parameterized

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import ranking_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


def _get_initial_and_final_steps(observations, scores):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      scores,
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  return initial_step, final_step


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1),
      single_experience)


@parameterized.parameters([{
    'policy_type': ranking_agent.RankingPolicyType.COSINE_DISTANCE,
    'batch_size': 7,
    'global_dim': 2,
    'item_dim': 3,
    'num_items': 10,
    'num_slots': 5,
    'non_click_score': None,
    'loss': 'default'
}, {
    'policy_type': ranking_agent.RankingPolicyType.DESCENDING_SCORES,
    'batch_size': 1,
    'global_dim': 7,
    'item_dim': 5,
    'num_items': 21,
    'num_slots': 17,
    'non_click_score': -10,
    'loss': 'default'
}, {
    'policy_type': ranking_agent.RankingPolicyType.DESCENDING_SCORES,
    'batch_size': 2,
    'global_dim': 3,
    'item_dim': 4,
    'num_items': 13,
    'num_slots': 11,
    'non_click_score': 0,
    'loss': 'softmax_cross_entropy'
}])
class RankingAgentTest(test_utils.TestCase, parameterized.TestCase):

  def testTrainAgentCascadingFeedback(self, policy_type, batch_size, global_dim,
                                      item_dim, num_items, num_slots,
                                      non_click_score, loss):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        global_dim, item_dim, num_items)
    scoring_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    reward_spec = {
        ranking_agent.CHOSEN_INDEX:
            tensor_spec.BoundedTensorSpec(
                shape=[], dtype=tf.int32, minimum=0, maximum=num_slots),
        ranking_agent.CHOSEN_VALUE:
            tensor_spec.TensorSpec(shape=(), dtype=tf.float32)
    }
    time_step_spec = ts.time_step_spec(obs_spec, reward_spec=reward_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(num_slots,), minimum=0, maximum=num_items - 1, dtype=tf.int32)
    agent = ranking_agent.RankingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        scoring_network=scoring_net,
        policy_type=policy_type,
        feedback_model=ranking_agent.FeedbackModel.CASCADING,
        non_click_score=non_click_score,
        optimizer=optimizer)
    global_obs = tf.reshape(
        tf.range(batch_size * global_dim, dtype=tf.float32),
        [batch_size, global_dim])
    item_obs = tf.reshape(
        tf.range(batch_size * num_slots * item_dim, dtype=tf.float32),
        [batch_size, num_slots, item_dim])
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY: global_obs,
        bandit_spec_utils.PER_ARM_FEATURE_KEY: item_obs,
    }
    scores = {
        ranking_agent.CHOSEN_INDEX: tf.range(batch_size),
        ranking_agent.CHOSEN_VALUE: tf.range(batch_size, dtype=tf.float32)
    }
    initial_step, final_step = _get_initial_and_final_steps(
        observations, scores)
    action_step = policy_step.PolicyStep(
        action=(),
        info=policy_utilities.PolicyInfo(
            predicted_rewards_mean=tf.reshape(
                tf.range(batch_size * num_slots, dtype=tf.float32),
                shape=[batch_size, 1, num_slots])))
    experience = _get_experience(initial_step, action_step, final_step)
    loss_info = agent.train(experience, None)
    self.assertAllEqual(loss_info.loss.shape, ())

  def testTrainAgentScoreFeedback(self, policy_type, batch_size, global_dim,
                                  item_dim, num_items, num_slots,
                                  non_click_score, loss):
    if not tf.executing_eagerly():
      self.skipTest('Only works in eager mode.')
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        global_dim, item_dim, num_items)
    scoring_net = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    reward_spec = tensor_spec.TensorSpec(shape=(num_slots,), dtype=tf.float32)
    time_step_spec = ts.time_step_spec(obs_spec, reward_spec=reward_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(num_slots,), minimum=0, maximum=num_items - 1, dtype=tf.int32)
    if non_click_score is not None:
      with self.assertRaisesRegex(ValueError, 'Parameter `non_click_score`'):
        agent = ranking_agent.RankingAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            scoring_network=scoring_net,
            policy_type=policy_type,
            feedback_model=ranking_agent.FeedbackModel.SCORE_VECTOR,
            non_click_score=non_click_score,
            optimizer=optimizer)
      non_click_score = None

    def loss_fn(logits, labels, reduction):
      del reduction
      return tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

    agent = ranking_agent.RankingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        scoring_network=scoring_net,
        policy_type=policy_type,
        error_loss_fn=(loss_fn if loss == 'softmax_cross_entropy' else
                       tf.compat.v1.losses.mean_squared_error),
        feedback_model=ranking_agent.FeedbackModel.SCORE_VECTOR,
        non_click_score=non_click_score,
        optimizer=optimizer)
    global_obs = tf.reshape(
        tf.range(batch_size * global_dim, dtype=tf.float32),
        [batch_size, global_dim])
    item_obs = tf.reshape(
        tf.range(batch_size * num_slots * item_dim, dtype=tf.float32),
        [batch_size, num_slots, item_dim])
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY: global_obs,
        bandit_spec_utils.PER_ARM_FEATURE_KEY: item_obs,
    }
    scores = tf.reshape(
        tf.range(batch_size * num_slots, dtype=tf.float32),
        shape=[batch_size, num_slots])
    initial_step, final_step = _get_initial_and_final_steps(
        observations, scores)
    action_step = policy_step.PolicyStep(
        action=(),
        info=policy_utilities.PolicyInfo(
            predicted_rewards_mean=tf.reshape(
                tf.range(batch_size * num_slots, dtype=tf.float32),
                shape=[batch_size, num_slots])))
    experience = _get_experience(initial_step, action_step, final_step)
    weights = tf.range(batch_size, dtype=tf.float32)
    agent.train(experience, weights)


if __name__ == '__main__':
  tf.test.main()
