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

"""End-to-end test for bandit training under stationary linear environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import stationary_stochastic_per_arm_py_environment as sspe
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.environments import tf_py_environment

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'LinUCB', ['LinUCB', 'LinTS', 'epsGreedy', 'NeuralLinUCB'],
    'Which agent to use. Possible values: `LinUCB`, `LinTS`, `epsGreedy`, and '
    '`NeuralLinUCB`.'
)

flags.DEFINE_enum(
    'network', 'commontower', ['commontower', 'dotproduct'],
    'Which network architecture to use for the eps-Greedy agent. '
    'Possible values are `commontower` and `dotproduct`.')

flags.DEFINE_bool('drop_arm_obs', False, 'Whether to wipe the arm observations '
                  'from the trajectories.')

flags.DEFINE_bool('add_trivial_mask', False, 'Whether to add action masking '
                  'that still allows all actions, for testing purposes.')

FLAGS = flags.FLAGS

# Environment and driver parameters.

BATCH_SIZE = 16
NUM_ACTIONS = 7
HIDDEN_PARAM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TRAINING_LOOPS = 2000
STEPS_PER_LOOP = 2

# Parameters for linear agents (LinUCB and LinTS).

AGENT_ALPHA = 0.1

# Parameters for neural agents (NeuralEpsGreedy and NerualLinUCB).

EPSILON = 0.01
LR = 0.05

# Parameters for NeuralLinUCB. ENCODING_DIM is the output dimension of the
# encoding network. This output will be used by either a linear reward layer and
# epsilon greedy exploration, or by a LinUCB logic, depending on the number of
# training steps executed so far. If the number of steps is less than or equal
# to EPS_PHASE_STEPS, epsilon greedy is used, otherwise LinUCB.

ENCODING_DIM = 9
EPS_PHASE_STEPS = 1000


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  class LinearNormalReward(object):

    def __init__(self, theta):
      self.theta = theta

    def __call__(self, x):
      mu = np.dot(x, self.theta)
      return np.random.normal(mu, 1)

  def _global_context_sampling_fn():
    return np.random.randint(-10, 10, [4]).astype(np.float32)

  def _arm_context_sampling_fn():
    return np.random.randint(-2, 3, [5]).astype(np.float32)

  reward_fn = LinearNormalReward(HIDDEN_PARAM)

  observation_and_action_constraint_splitter = None
  num_actions_fn = None
  if FLAGS.add_trivial_mask:
    num_actions_fn = lambda: NUM_ACTIONS
    observation_and_action_constraint_splitter = lambda x: (x[0], x[1])

  env = sspe.StationaryStochasticPerArmPyEnvironment(
      _global_context_sampling_fn,
      _arm_context_sampling_fn,
      NUM_ACTIONS,
      reward_fn,
      num_actions_fn,
      batch_size=BATCH_SIZE)
  environment = tf_py_environment.TFPyEnvironment(env)

  if FLAGS.agent == 'LinUCB':
    agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        alpha=AGENT_ALPHA,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=True,
        dtype=tf.float32)
  elif FLAGS.agent == 'LinTS':
    agent = lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        alpha=AGENT_ALPHA,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=True,
        dtype=tf.float32)
  elif FLAGS.agent == 'epsGreedy':
    obs_spec = environment.observation_spec()
    if FLAGS.add_trivial_mask:
      obs_spec = obs_spec[0]
    if FLAGS.network == 'commontower':
      network = (
          global_and_arm_feature_network
          .create_feed_forward_common_tower_network(obs_spec, (40, 30),
                                                    (30, 40), (40, 20)))
    elif FLAGS.network == 'dotproduct':
      network = (
          global_and_arm_feature_network
          .create_feed_forward_dot_product_network(obs_spec, (4, 3, 6),
                                                   (3, 4, 6)))
    agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        reward_network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
        epsilon=EPSILON,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=True,
        emit_policy_info=policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN)
  elif FLAGS.agent == 'NeuralLinUCB':
    obs_spec = environment.observation_spec()
    if FLAGS.add_trivial_mask:
      obs_spec = obs_spec[0]
    network = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (40, 30), (30, 40), (40, 20), ENCODING_DIM))
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        encoding_network=network,
        encoding_network_num_train_steps=EPS_PHASE_STEPS,
        encoding_dim=ENCODING_DIM,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
        alpha=1.0,
        gamma=1.0,
        epsilon_greedy=EPSILON,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=True,
        debug_summaries=True,
        summarize_grads_and_vars=True,
        emit_policy_info=policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN)

  def _all_rewards(observation, hidden_param):
    """Outputs rewards for all actions, given an observation."""
    if observation_and_action_constraint_splitter is not None:
      observation = observation_and_action_constraint_splitter(observation)[0]
    hidden_param = tf.cast(hidden_param, dtype=tf.float32)
    global_obs = observation[bandit_spec_utils.GLOBAL_FEATURE_KEY]
    per_arm_obs = observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]
    num_actions = tf.shape(per_arm_obs)[1]
    tiled_global = tf.tile(
        tf.expand_dims(global_obs, axis=1), [1, num_actions, 1])
    concatenated = tf.concat([tiled_global, per_arm_obs], axis=-1)
    rewards = tf.linalg.matvec(concatenated, hidden_param)
    return rewards

  def optimal_reward(observation, hidden_param):
    return tf.reduce_max(_all_rewards(observation, hidden_param), axis=1)

  def optimal_action(observation, hidden_param):
    return tf.argmax(
        _all_rewards(observation, hidden_param), axis=1, output_type=tf.int32)

  optimal_reward_fn = functools.partial(
      optimal_reward, hidden_param=HIDDEN_PARAM)
  optimal_action_fn = functools.partial(
      optimal_action, hidden_param=HIDDEN_PARAM)
  regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
  suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
      optimal_action_fn)

  if FLAGS.drop_arm_obs:
    drop_arm_feature_fn = functools.partial(
        bandit_spec_utils.drop_arm_observation,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter))
  else:
    drop_arm_feature_fn = None
  trainer.train(
      root_dir=FLAGS.root_dir,
      agent=agent,
      environment=environment,
      training_loops=TRAINING_LOOPS,
      steps_per_loop=STEPS_PER_LOOP,
      additional_metrics=[regret_metric, suboptimal_arms_metric],
      training_data_spec_transformation_fn=drop_arm_feature_fn)


if __name__ == '__main__':
  app.run(main)
