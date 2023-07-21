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

"""End-to-end test for bandits against the 'covertype' environment.

Forest Cover type dataset in the UCI Machine Learning Repository can be found at
https://archive.ics.uci.edu/ml/datasets/covertype.

We turn this 7-class classification problem to a bandit problem with 7 actions.
The reward is 1 if the right class was chosen, 0 otherwise.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import app
from absl import flags

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent as eps_greedy_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import classification_environment as ce
from tf_agents.bandits.environments import dataset_utilities
from tf_agents.bandits.environments import environment_utilities as env_util
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.networks import q_network


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'epsGreedy', ['LinUCB', 'LinTS', 'epsGreedy'],
    'Which agent to use. Possible values are `LinUCB` and `LinTS`.')
flags.DEFINE_string(
    'covertype_csv', '',
    'Location of the csv file containing the covertype dataset.')

FLAGS = flags.FLAGS
tfd = tfp.distributions


BATCH_SIZE = 8
TRAINING_LOOPS = 15000
STEPS_PER_LOOP = 2
AGENT_ALPHA = 10.0

EPSILON = 0.01
LAYERS = (300, 200, 100, 100, 50, 50)
LR = 0.002


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  with tf.device('/CPU:0'):  # due to b/128333994

    covertype_dataset = dataset_utilities.convert_covertype_dataset(
        FLAGS.covertype_csv)
    covertype_reward_distribution = tfd.Independent(
        tfd.Deterministic(tf.eye(7)), reinterpreted_batch_ndims=2)
    environment = ce.ClassificationBanditEnvironment(
        covertype_dataset, covertype_reward_distribution, BATCH_SIZE)

    optimal_reward_fn = functools.partial(
        env_util.compute_optimal_reward_with_classification_environment,
        environment=environment)

    optimal_action_fn = functools.partial(
        env_util.compute_optimal_action_with_classification_environment,
        environment=environment)

    if FLAGS.agent == 'LinUCB':
      agent = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          emit_log_probability=False,
          dtype=tf.float32)
    elif FLAGS.agent == 'LinTS':
      agent = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
    elif FLAGS.agent == 'epsGreedy':
      network = q_network.QNetwork(
          input_tensor_spec=environment.time_step_spec().observation,
          action_spec=environment.action_spec(),
          fc_layer_params=LAYERS)
      agent = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          epsilon=EPSILON)

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
