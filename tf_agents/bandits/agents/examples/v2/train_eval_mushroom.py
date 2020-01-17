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

"""End-to-end test for bandits against a mushroom environment.
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
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import classification_environment as ce
from tf_agents.bandits.environments import environment_utilities as env_util
from tf_agents.bandits.environments import mushroom_environment_utilities
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum(
    'agent', 'LinUCB', ['LinUCB', 'LinTS'],
    'Which agent to use. Possible values are `LinUCB` and `LinTS`.')
flags.DEFINE_string(
    'mushroom_csv', '',
    'Location of the csv file containing the mushroom dataset.')

FLAGS = flags.FLAGS
tfd = tfp.distributions


BATCH_SIZE = 8
TRAINING_LOOPS = 200
STEPS_PER_LOOP = 2
AGENT_ALPHA = 10.0


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  with tf.device('/CPU:0'):  # due to b/128333994

    mushroom_reward_distribution = (
        mushroom_environment_utilities.mushroom_reward_distribution(
            r_noeat=0.0, r_eat_safe=5.0, r_eat_poison_bad=-35.0,
            r_eat_poison_good=5.0, prob_poison_bad=0.5))
    mushroom_dataset = (
        mushroom_environment_utilities.convert_mushroom_csv_to_tf_dataset(
            FLAGS.mushroom_csv))
    environment = ce.ClassificationBanditEnvironment(
        mushroom_dataset, mushroom_reward_distribution, BATCH_SIZE)

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
