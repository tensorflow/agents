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

# Lint as: python2, python3
r"""Train and Eval RNN SAC.

To run:

```bash
tensorboard --logdir $HOME/tmp/sac_rnn/dm/CartPole-Balance/ --port 2223 &

python tf_agents/agents/sac/examples/v2:train_eval_rnn --\
  --root_dir=$HOME/tmp/sac_rnn/dm/CartPole-Balance/ \
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
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_dm_control
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    env_name='cartpole',
    task_name='balance',
    observations_whitelist='position',
    eval_env_name=None,
    num_iterations=1000000,
    # Params for networks.
    actor_fc_layers=(400, 300),
    actor_output_fc_layers=(100,),
    actor_lstm_size=(40,),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(300,),
    critic_output_fc_layers=(100,),
    critic_lstm_size=(40,),
    num_parallel_environments=1,
    # Params for collect
    initial_collect_episodes=1,
    collect_episodes_per_iteration=1,
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    critic_learning_rate=3e-4,
    train_sequence_length=20,
    actor_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=0.1,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=50000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  """A simple train and eval for RNN SAC on DM control."""
  root_dir = os.path.expanduser(root_dir)

  summary_writer = tf.compat.v2.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

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

    env_load_fn = functools.partial(suite_dm_control.load,
                                    task_name=task_name,
                                    env_wrappers=env_wrappers)

    if num_parallel_environments == 1:
      py_env = env_load_fn(env_name)
    else:
      py_env = parallel_py_environment.ParallelPyEnvironment(
          [lambda: env_load_fn(env_name)]*num_parallel_environments)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_env_name = eval_env_name or env_name
    eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(eval_env_name))

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=actor_fc_layers,
        lstm_size=actor_lstm_size,
        output_fc_layer_params=actor_output_fc_layers,
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    critic_net = critic_rnn_network.CriticRnnNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        lstm_size=critic_lstm_size,
        output_fc_layer_params=critic_output_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    env_steps = tf_metrics.EnvironmentSteps(prefix='Train')
    average_return = tf_metrics.AverageReturnMetric(
        prefix='Train',
        buffer_size=num_eval_episodes,
        batch_size=tf_env.batch_size)
    train_metrics = [
        tf_metrics.NumberOfEpisodes(prefix='Train'),
        env_steps,
        average_return,
        tf_metrics.AverageEpisodeLengthMetric(
            prefix='Train',
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size),
    ]

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy

    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'train'),
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_episodes=initial_collect_episodes)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    if use_tf_functions:
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data.
    if env_steps.result() == 0 or replay_buffer.num_frames() == 0:
      logging.info(
          'Initializing replay buffer by collecting experience for %d episodes '
          'with a random policy.', initial_collect_episodes)
      initial_collect_driver.run()

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=env_steps.result(),
        summary_writer=summary_writer,
        summary_prefix='Eval',
    )
    if eval_metrics_callback is not None:
      eval_metrics_callback(results, env_steps.result())
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    time_acc = 0
    env_steps_before = env_steps.result().numpy()

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
      # Reduce filter_fn over full trajectory sampled. The sequence is kept only
      # if all elements except for the last one pass the filter. This is to
      # allow training on terminal steps.
      return tf.reduce_all(~trajectories.is_boundary()[:-1])
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=train_sequence_length+1).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(5)
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    for _ in range(num_iterations):
      start_time = time.time()
      start_env_steps = env_steps.result()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      episode_steps = env_steps.result() - start_env_steps
      # TODO(b/152648849)
      for _ in range(episode_steps):
        for _ in range(train_steps_per_iteration):
          train_step()
        time_acc += time.time() - start_time

        if global_step.numpy() % log_interval == 0:
          logging.info('env steps = %d, average return = %f',
                       env_steps.result(), average_return.result())
          env_steps_per_sec = (env_steps.result().numpy() -
                               env_steps_before) / time_acc
          logging.info('%.3f env steps/sec', env_steps_per_sec)
          tf.compat.v2.summary.scalar(
              name='env_steps_per_sec',
              data=env_steps_per_sec,
              step=env_steps.result())
          time_acc = 0
          env_steps_before = env_steps.result().numpy()

        for train_metric in train_metrics:
          train_metric.tf_summaries(train_step=env_steps.result())

        if global_step.numpy() % eval_interval == 0:
          results = metric_utils.eager_compute(
              eval_metrics,
              eval_tf_env,
              eval_policy,
              num_episodes=num_eval_episodes,
              train_step=env_steps.result(),
              summary_writer=summary_writer,
              summary_prefix='Eval',
          )
          if eval_metrics_callback is not None:
            eval_metrics_callback(results, env_steps.numpy())
          metric_utils.log_metrics(eval_metrics)

        global_step_val = global_step.numpy()
        if global_step_val % train_checkpoint_interval == 0:
          train_checkpointer.save(global_step=global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
          policy_checkpointer.save(global_step=global_step_val)

        if global_step_val % rb_checkpoint_interval == 0:
          rb_checkpointer.save(global_step=global_step_val)


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(FLAGS.root_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
