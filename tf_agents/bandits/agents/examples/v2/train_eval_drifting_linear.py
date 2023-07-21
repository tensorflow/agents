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

"""End-to-end test for bandits against a drifting linear environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import drifting_linear_environment as dle
from tf_agents.bandits.environments import non_stationary_stochastic_environment as nse
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'LinUCB', ['LinUCB', 'LinTS'],
    'Which agent to use. Possible values are `LinUCB` and `LinTS`.')

FLAGS = flags.FLAGS
tfd = tfp.distributions


CONTEXT_DIM = 15
NUM_ACTIONS = 5
REWARD_NOISE_VARIANCE = 0.01
DRIFT_VARIANCE = 0.01
DRIFT_MEAN = 0.01
BATCH_SIZE = 8
TRAINING_LOOPS = 200
STEPS_PER_LOOP = 2
AGENT_ALPHA = 10.0


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  with tf.device('/CPU:0'):  # due to b/128333994
    observation_shape = [CONTEXT_DIM]
    overall_shape = [BATCH_SIZE] + observation_shape
    observation_distribution = tfd.Normal(
        loc=tf.zeros(overall_shape), scale=tf.ones(overall_shape))
    action_shape = [NUM_ACTIONS]
    observation_to_reward_shape = observation_shape + action_shape
    observation_to_reward_distribution = tfd.Normal(
        loc=tf.zeros(observation_to_reward_shape),
        scale=tf.ones(observation_to_reward_shape))
    drift_distribution = tfd.Normal(loc=DRIFT_MEAN, scale=DRIFT_VARIANCE)
    additive_reward_distribution = tfd.Normal(
        loc=tf.zeros(action_shape),
        scale=(REWARD_NOISE_VARIANCE * tf.ones(action_shape)))
    environment_dynamics = dle.DriftingLinearDynamics(
        observation_distribution,
        observation_to_reward_distribution,
        drift_distribution,
        additive_reward_distribution)
    environment = nse.NonStationaryStochasticEnvironment(environment_dynamics)

    if FLAGS.agent == 'LinUCB':
      agent = lin_ucb_agent.LinearUCBAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          gamma=0.95,
          emit_log_probability=False,
          dtype=tf.float32)
    elif FLAGS.agent == 'LinTS':
      agent = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          gamma=0.95,
          dtype=tf.float32)

    regret_metric = tf_bandit_metrics.RegretMetric(
        environment.environment_dynamics.compute_optimal_reward)
    suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
        environment.environment_dynamics.compute_optimal_action)

    trainer.train(
        root_dir=FLAGS.root_dir,
        agent=agent,
        environment=environment,
        training_loops=TRAINING_LOOPS,
        steps_per_loop=STEPS_PER_LOOP,
        additional_metrics=[regret_metric, suboptimal_arms_metric])


if __name__ == '__main__':
  app.run(main)
