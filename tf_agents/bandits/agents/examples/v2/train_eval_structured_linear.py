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

"""End-to-end test for bandit training under structured linear environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent as eps_greedy_agent
from tf_agents.bandits.agents import utils
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import stationary_stochastic_py_environment as sspe
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'LinUCB', ['LinUCB', 'LinTS', 'epsGreedy'],
    'Which agent to use. Possible values are `LinUCB` and `LinTS`.')

FLAGS = flags.FLAGS

BATCH_SIZE = 8
CONTEXT_DIM = 100
NUM_ACTIONS = 100
REWARD_NOISE_VARIANCE = 0.0001
TRAINING_LOOPS = 2000
STEPS_PER_LOOP = 4
AGENT_ALPHA = 0.1
NN_LEARNING_RATE = 1e-2
REWARD_NETWORK_LAYER_PARAMS = (50, 50, 50)
EPSILON = 0.1


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  with tf.device('/CPU:0'):  # due to b/128333994
    action_reward_fns = (
        environment_utilities.structured_linear_reward_fn_generator(
            CONTEXT_DIM, NUM_ACTIONS, REWARD_NOISE_VARIANCE))

    env = sspe.StationaryStochasticPyEnvironment(
        functools.partial(
            environment_utilities.context_sampling_fn,
            batch_size=BATCH_SIZE,
            context_dim=CONTEXT_DIM),
        action_reward_fns,
        batch_size=BATCH_SIZE)
    environment = tf_py_environment.TFPyEnvironment(env)

    optimal_reward_fn = functools.partial(
        environment_utilities.tf_compute_optimal_reward,
        per_action_reward_fns=action_reward_fns)

    optimal_action_fn = functools.partial(
        environment_utilities.tf_compute_optimal_action,
        per_action_reward_fns=action_reward_fns)

    if FLAGS.agent == 'LinUCB':
      agent = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
    elif FLAGS.agent == 'epsGreedy':
      laplacian_matrix = utils.build_laplacian_over_ordinal_integer_actions(
          environment.action_spec())

      network = q_network.QNetwork(
          input_tensor_spec=environment.time_step_spec().observation,
          action_spec=environment.action_spec(),
          fc_layer_params=REWARD_NETWORK_LAYER_PARAMS)
      agent = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=NN_LEARNING_RATE),
          epsilon=EPSILON,
          laplacian_matrix=laplacian_matrix,
          laplacian_smoothing_weight=0.01)
    elif FLAGS.agent == 'LinTS':
      agent = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32)

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
