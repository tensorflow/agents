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

"""End-to-end example for bandit training under the MovieLens bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import dropout_thompson_sampling_agent as dropout_ts_agent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent as eps_greedy_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import movielens_per_arm_py_environment
from tf_agents.bandits.environments import movielens_py_environment
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_path', '', 'Location of the movielens dataset.')
flags.DEFINE_enum(
    'agent', 'LinUCB', ['LinUCB', 'LinTS', 'epsGreedy', 'DropoutTS'],
    'Which agent to use. Possible values: `LinUCB`, `LinTS`, `epsGreedy`,'
    ' `DropoutTS`.')
flags.DEFINE_bool('per_arm', False, 'Whether to  use the per arm version of the'
                  ' movielens environment.')

FLAGS = flags.FLAGS

BATCH_SIZE = 8
TRAINING_LOOPS = 20000
STEPS_PER_LOOP = 2

RANK_K = 20
NUM_ACTIONS = 20

# LinUCB agent constants.

AGENT_ALPHA = 10.0

# epsilon Greedy constants.

EPSILON = 0.05
LAYERS = (50, 50, 50)
LR = 0.005

# Dropout TS constants.
DROPOUT_RATE = 0.2


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  data_path = FLAGS.data_path
  if not data_path:
    raise ValueError('Please specify the location of the data file.')
  if FLAGS.per_arm:
    env = movielens_per_arm_py_environment.MovieLensPerArmPyEnvironment(
        data_path, RANK_K, BATCH_SIZE, num_actions=NUM_ACTIONS)
  else:
    env = movielens_py_environment.MovieLensPyEnvironment(
        data_path, RANK_K, BATCH_SIZE, num_movies=NUM_ACTIONS)
  environment = tf_py_environment.TFPyEnvironment(env)

  optimal_reward_fn = functools.partial(
      environment_utilities.compute_optimal_reward_with_movielens_environment,
      environment=environment)

  optimal_action_fn = functools.partial(
      environment_utilities.compute_optimal_action_with_movielens_environment,
      environment=environment)

  if FLAGS.agent == 'LinUCB':
    agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        tikhonov_weight=0.001,
        alpha=AGENT_ALPHA,
        dtype=tf.float32,
        accepts_per_arm_features=FLAGS.per_arm)
  elif FLAGS.agent == 'LinTS':
    agent = lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        dtype=tf.float32,
        accepts_per_arm_features=FLAGS.per_arm)
  elif FLAGS.agent == 'epsGreedy':
    if FLAGS.per_arm:
      network = (
          global_and_arm_feature_network
          .create_feed_forward_dot_product_network(
              environment.time_step_spec().observation,
              global_layers=LAYERS,
              arm_layers=LAYERS))
    else:
      network = q_network.QNetwork(
          input_tensor_spec=environment.time_step_spec().observation,
          action_spec=environment.action_spec(),
          fc_layer_params=LAYERS)
    agent = eps_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        reward_network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
        epsilon=EPSILON,
        emit_policy_info='predicted_rewards_mean',
        info_fields_to_inherit_from_greedy=['predicted_rewards_mean'])
  elif FLAGS.agent == 'DropoutTS':
    train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    def dropout_fn():
      return tf.math.maximum(
          tf.math.reciprocal_no_nan(1.01 +
                                    tf.cast(train_step_counter, tf.float32)),
          0.0003)

    agent = dropout_ts_agent.DropoutThompsonSamplingAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        dropout_rate=dropout_fn,
        network_layers=LAYERS,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR))

  regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
  suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
      optimal_action_fn)

  trainer.train(
      root_dir=FLAGS.root_dir,
      agent=agent,
      environment=environment,
      training_loops=TRAINING_LOOPS,
      steps_per_loop=STEPS_PER_LOOP,
      additional_metrics=[regret_metric, suboptimal_arms_metric])


if __name__ == '__main__':
  app.run(main)
