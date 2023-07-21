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

"""End-to-end test for bandit training under Bernoulli environments."""


import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import bernoulli_thompson_sampling_agent as bern_ts_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.environments import bernoulli_py_environment as bern_env
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.environments import tf_py_environment

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_enum('agent', 'BernTS', ['BernTS'], 'Which agent to use.')

FLAGS = flags.FLAGS

BATCH_SIZE = 8
TRAINING_LOOPS = 1000
STEPS_PER_LOOP = 2


def main(unused_argv):
  tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.

  means = [0.1, 0.2, 0.3, 0.45, 0.5]
  env = bern_env.BernoulliPyEnvironment(
      means=means,
      batch_size=BATCH_SIZE)
  environment = tf_py_environment.TFPyEnvironment(env)

  def optimal_reward_fn(unused_observation):
    return np.max(means)

  def optimal_action_fn(unused_observation):
    return np.int32(np.argmax(means))

  if FLAGS.agent == 'BernTS':
    agent = bern_ts_agent.BernoulliThompsonSamplingAgent(
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec(),
        dtype=tf.float64,
        batch_size=BATCH_SIZE)
  else:
    raise ValueError('Only BernoulliTS is supported for now.')

  regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
  suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
      optimal_action_fn)

  trainer.train(
      root_dir=FLAGS.root_dir,
      agent=agent,
      environment=environment,
      training_loops=TRAINING_LOOPS,
      steps_per_loop=STEPS_PER_LOOP,
      additional_metrics=[regret_metric, suboptimal_arms_metric],
      save_policy=False)


if __name__ == '__main__':
  app.run(main)
