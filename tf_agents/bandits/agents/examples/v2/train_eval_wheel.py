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

"""End-to-end test for bandit training under the wheel bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import exp3_mixture_agent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent as eps_greedy_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import wheel_py_environment
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.bandits.policies import policy_utilities
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'LinUCB', ['LinUCB', 'LinTS', 'epsGreedy', 'random', 'Mix'],
    'Which agent to use. Possible values: `LinUCB`, `LinTS`, `epsGreedy`, '
    '`random`, `Mix`.')

FLAGS = flags.FLAGS

BATCH_SIZE = 8
TRAINING_LOOPS = 20000
STEPS_PER_LOOP = 2

DELTA = 0.5
MU_BASE = [0.05, 0.01, 0.011, 0.009, 0.012]
STD_BASE = [0.001] * 5
MU_HIGH = 0.5
STD_HIGH = 0.001


# LinUCB agent constants.

AGENT_ALPHA = 10.0

# epsilon Greedy constants.

EPSILON = 0.05
LAYERS = (50, 50, 50)
LR = 0.001


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  with tf.device('/CPU:0'):  # due to b/128333994
    env = wheel_py_environment.WheelPyEnvironment(DELTA, MU_BASE, STD_BASE,
                                                  MU_HIGH, STD_HIGH, BATCH_SIZE)
    environment = tf_py_environment.TFPyEnvironment(env)

    optimal_reward_fn = functools.partial(
        environment_utilities.tf_wheel_bandit_compute_optimal_reward,
        delta=DELTA,
        mu_inside=MU_BASE[0],
        mu_high=MU_HIGH)
    optimal_action_fn = functools.partial(
        environment_utilities.tf_wheel_bandit_compute_optimal_action,
        delta=DELTA)
    network = q_network.QNetwork(
        input_tensor_spec=environment.time_step_spec().observation,
        action_spec=environment.action_spec(),
        fc_layer_params=(LAYERS))

    if FLAGS.agent == 'LinUCB':
      agent = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
    elif FLAGS.agent == 'LinTS':
      agent = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
    elif FLAGS.agent == 'epsGreedy':
      agent = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          epsilon=EPSILON)
    elif FLAGS.agent == 'random':
      agent = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          epsilon=1.)
    elif FLAGS.agent == 'Mix':
      emit_policy_info = (policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,)
      agent_epsgreedy = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          emit_policy_info=emit_policy_info,
          epsilon=EPSILON)
      agent_linucb = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          emit_policy_info=emit_policy_info,
          dtype=tf.float32)
      agent_random = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          emit_policy_info=emit_policy_info,
          epsilon=1.)
      agent_halfrandom = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          emit_policy_info=emit_policy_info,
          epsilon=0.5)
      agent = exp3_mixture_agent.Exp3MixtureAgent(
          (agent_epsgreedy, agent_linucb, agent_random, agent_halfrandom))

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
