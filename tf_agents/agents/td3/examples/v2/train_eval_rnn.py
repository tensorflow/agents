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

r"""Train and Eval TD3.

To run:

```bash
tensorboard --logdir $HOME/tmp/td3_rnn/dm/CartPole-Balance/ --port 2223 &

python tf_agents/agents/td3/examples/v2/train_eval_rnn.py \
  --root_dir=$HOME/tmp/td3_rnn/dm/CartPole-Balance/ \
  --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_dm_control
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    env_name='cartpole',
    task_name='balance',
    observations_whitelist='position',
    num_iterations=100000,
    actor_fc_layers=(400, 300),
    actor_output_fc_layers=(100,),
    actor_lstm_size=(40,),
    critic_obs_fc_layers=(400,),
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(300,),
    critic_output_fc_layers=(100,),
    critic_lstm_size=(40,),
    # Params for collect
    initial_collect_episodes=1,
    collect_episodes_per_iteration=1,
    replay_buffer_capacity=100000,
    exploration_noise_std=0.1,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=200,
    batch_size=64,
    actor_update_period=2,
    train_sequence_length=10,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    dqda_clipping=None,
    td_errors_loss_fn=None,
    gamma=0.995,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for checkpoints, summaries, and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):

  """A simple train and eval for TD3."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    if observations_whitelist is not None:
      env_wrappers = [
          functools.partial(
              wrappers.FlattenObservationsWrapper,
              observations_whitelist=[observations_whitelist])
      ]
    else:
      env_wrappers = []

    tf_env = tf_py_environment.TFPyEnvironment(
        suite_dm_control.load(env_name, task_name, env_wrappers=env_wrappers))
    eval_tf_env = tf_py_environment.TFPyEnvironment(
        suite_dm_control.load(env_name, task_name, env_wrappers=env_wrappers))

    actor_net = actor_rnn_network.ActorRnnNetwork(
        tf_env.time_step_spec().observation,
        tf_env.action_spec(),
        input_fc_layer_params=actor_fc_layers,
        lstm_size=actor_lstm_size,
        output_fc_layer_params=actor_output_fc_layers)

    critic_net_input_specs = (tf_env.time_step_spec().observation,
                              tf_env.action_spec())

    critic_net = critic_rnn_network.CriticRnnNetwork(
        critic_net_input_specs,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        lstm_size=critic_lstm_size,
        output_fc_layer_params=critic_output_fc_layers,
    )

    tf_agent = td3_agent.Td3Agent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        exploration_noise_std=exploration_noise_std,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        actor_update_period=actor_update_period,
        dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
    )
    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=initial_collect_episodes)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    if use_tf_functions:
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data.
    logging.info(
        'Initializing replay buffer by collecting experience for %d episodes '
        'with a random policy.', initial_collect_episodes)
    initial_collect_driver.run()

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    if eval_metrics_callback is not None:
      eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [BxTx...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=train_sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    for _ in range(num_iterations):
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      for _ in range(train_steps_per_iteration):
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)
      time_acc += time.time() - start_time

      if global_step.numpy() % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step.numpy(),
                     train_loss.loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step.numpy() % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)

    return train_loss


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
