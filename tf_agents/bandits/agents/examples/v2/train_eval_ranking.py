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

"""End-to-end test for ranking."""

import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import ranking_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import ranking_environment
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.environments import tf_py_environment
from tf_agents.specs import bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('policy_type', 'cosine_distance',
                    'The type of policy used.')
flags.DEFINE_string('feedback_model', 'cascading', 'The feedback model. Only '
                    '`cascading` is implemented at the moment.')
flags.DEFINE_string('click_model', 'ghost_actions', 'The way diversity is '
                    'encouraged. Possible values are `ghost_actions` and '
                    '`distance_based`.')
flags.DEFINE_float('distance_threshold', 10.0, 'If the diversity model is '
                   '`distance_based`, this is the distance threshold.')
flags.DEFINE_string('env_type', 'base', 'The environment used. Possible values'
                    ' are `base` and `exp_pos_bias`.')

FLAGS = flags.FLAGS

# Environment and driver parameters.

BATCH_SIZE = 128
NUM_ITEMS = 1001
NUM_SLOTS = 5
GLOBAL_DIM = 50
ITEM_DIM = 40

TRAINING_LOOPS = 2000
STEPS_PER_LOOP = 2

LR = 0.05


def main(unused_argv):

  def _global_sampling_fn():
    return np.random.randint(-1, 1, [GLOBAL_DIM]).astype(np.float32)

  def _item_sampling_fn():
    unnormalized = np.random.randint(-2, 3, [ITEM_DIM]).astype(np.float32)
    return unnormalized / np.linalg.norm(unnormalized)

  def _relevance_fn(global_obs, item_obs):
    min_dim = min(GLOBAL_DIM, ITEM_DIM)
    dot_prod = np.dot(global_obs[:min_dim],
                      item_obs[:min_dim]).astype(np.float32)
    return 1 / (1 + np.exp(-dot_prod))

  if FLAGS.env_type == 'exp_pos_bias':
    positional_biases = list(1.0 / np.arange(1, NUM_SLOTS + 1)**1.3)
    env = ranking_environment.ExplicitPositionalBiasRankingEnvironment(
        _global_sampling_fn,
        _item_sampling_fn,
        _relevance_fn,
        NUM_ITEMS,
        positional_biases,
        batch_size=BATCH_SIZE)
    feedback_model = ranking_agent.FeedbackModel.SCORE_VECTOR
  elif FLAGS.env_type == 'base':
    # Inner product with the excess dimensions ignored.
    scores_weight_matrix = np.eye(ITEM_DIM, GLOBAL_DIM, dtype=np.float32)

    feedback_model = ranking_agent.FeedbackModel.SCORE_VECTOR
    if FLAGS.feedback_model == 'cascading':
      feedback_model = ranking_agent.FeedbackModel.CASCADING
    else:
      raise NotImplementedError('Feedback model {} not implemented'.format(
          FLAGS.feedback_model))
    if FLAGS.click_model == 'ghost_actions':
      click_model = ranking_environment.ClickModel.GHOST_ACTIONS
    elif FLAGS.click_model == 'distance_based':
      click_model = ranking_environment.ClickModel.DISTANCE_BASED
    else:
      raise NotImplementedError('Diversity mode {} not implemented'.format(
          FLAGS.click_mode))

    env = ranking_environment.RankingPyEnvironment(
        _global_sampling_fn,
        _item_sampling_fn,
        num_items=NUM_ITEMS,
        num_slots=NUM_SLOTS,
        scores_weight_matrix=scores_weight_matrix,
        # TODO(b/247995883): Merge the two feedback model enums from the agent
        # and the enviroment.
        feedback_model=feedback_model.value,
        click_model=click_model,
        distance_threshold=FLAGS.distance_threshold,
        batch_size=BATCH_SIZE)

  environment = tf_py_environment.TFPyEnvironment(env)

  obs_spec = environment.observation_spec()
  network = (
      global_and_arm_feature_network.create_feed_forward_common_tower_network(
          obs_spec, (40, 10), (40, 10), (20, 10)))
  if FLAGS.policy_type == 'cosine_distance':
    policy_type = ranking_agent.RankingPolicyType.COSINE_DISTANCE
  elif FLAGS.policy_type == 'no_penalty':
    policy_type = ranking_agent.RankingPolicyType.NO_PENALTY
  else:
    raise NotImplementedError('Policy type {} is not implemented'.format(
        FLAGS.policy_type))
  agent = ranking_agent.RankingAgent(
      time_step_spec=environment.time_step_spec(),
      action_spec=environment.action_spec(),
      scoring_network=network,
      optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
      policy_type=policy_type,
      feedback_model=feedback_model,
      summarize_grads_and_vars=True)

  def order_items_from_action_fn(orig_trajectory):
    """Puts the features of the selected items in the recommendation order.

    This function is used to make sure that at training the item observation is
    filled with features of items selected by the policy, in the order of the
    selection. Features of unselected items are discarded.

    Args:
      orig_trajectory: The trajectory as output by the policy

    Returns:
      The modified trajectory that contains slotted item features.
    """
    item_obs = orig_trajectory.observation[
        bandit_spec_utils.PER_ARM_FEATURE_KEY]
    action = orig_trajectory.action
    if isinstance(
        orig_trajectory.observation[bandit_spec_utils.PER_ARM_FEATURE_KEY],
        tensor_spec.TensorSpec):
      dtype = orig_trajectory.observation[
          bandit_spec_utils.PER_ARM_FEATURE_KEY].dtype
      shape = [
          NUM_SLOTS, orig_trajectory.observation[
              bandit_spec_utils.PER_ARM_FEATURE_KEY].shape[-1]
      ]
      new_observation = {
          bandit_spec_utils.GLOBAL_FEATURE_KEY:
              orig_trajectory.observation[bandit_spec_utils.GLOBAL_FEATURE_KEY],
          bandit_spec_utils.PER_ARM_FEATURE_KEY:
              tensor_spec.TensorSpec(dtype=dtype, shape=shape)
      }
    else:
      slotted_items = tf.gather(item_obs, action, batch_dims=1)
      new_observation = {
          bandit_spec_utils.GLOBAL_FEATURE_KEY:
              orig_trajectory.observation[bandit_spec_utils.GLOBAL_FEATURE_KEY],
          bandit_spec_utils.PER_ARM_FEATURE_KEY:
              slotted_items
      }
    return trajectory.Trajectory(
        step_type=orig_trajectory.step_type,
        observation=new_observation,
        action=(),
        policy_info=(),
        next_step_type=orig_trajectory.next_step_type,
        reward=orig_trajectory.reward,
        discount=orig_trajectory.discount)

  trainer.train(
      root_dir=FLAGS.root_dir,
      agent=agent,
      environment=environment,
      training_loops=TRAINING_LOOPS,
      steps_per_loop=STEPS_PER_LOOP,
      training_data_spec_transformation_fn=order_items_from_action_fn,
      save_policy=False)


if __name__ == '__main__':
  app.run(main)
