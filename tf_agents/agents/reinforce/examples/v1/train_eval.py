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

r"""Train and Eval REINFORCE.

To run:

```bash
tensorboard --logdir $HOME/tmp/reinforce_v1/gym/CartPole-v0/ --port 2223 &

python tf_agents/agents/reinforce/examples/v1/train_eval.py \
  --root_dir=$HOME/tmp/reinforce_v1/gym/CartPole-v0/ \
  --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.policies import py_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 1000,
                     'Total number train/eval iterations to perform.')
FLAGS = flags.FLAGS


def train_eval(
    root_dir,
    env_name='CartPole-v0',
    num_iterations=1000,
    # TODO(b/127576522): rename to policy_fc_layers.
    actor_fc_layers=(100,),
    value_net_fc_layers=(100,),
    use_value_network=False,
    # Params for collect
    collect_episodes_per_iteration=2,
    replay_buffer_capacity=2000,
    # Params for train
    learning_rate=1e-3,
    gamma=0.9,
    gradient_clipping=None,
    normalize_returns=True,
    value_estimation_loss_coef=0.2,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=100,
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=100,
    policy_checkpoint_interval=100,
    rb_checkpoint_interval=200,
    log_interval=100,
    summary_interval=100,
    summaries_flush_secs=1,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  """A simple train and eval for Reinforce."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    eval_py_env = suite_gym.load(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    # TODO(b/127870767): Handle distributions without gin.
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.time_step_spec().observation,
        tf_env.action_spec(),
        fc_layer_params=actor_fc_layers)

    if use_value_network:
      value_net = value_network.ValueNetwork(
          tf_env.time_step_spec().observation,
          fc_layer_params=value_net_fc_layers)

    tf_agent = reinforce_agent.ReinforceAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        value_network=value_net if use_value_network else None,
        value_estimation_loss_coef=value_estimation_loss_coef,
        gamma=gamma,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        normalize_returns=normalize_returns,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    collect_policy = tf_agent.collect_policy

    collect_op = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration).run()

    experience = replay_buffer.gather_all()
    train_op = tf_agent.train(experience)
    clear_rb_op = replay_buffer.clear()

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=tf_agent.policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    summary_ops = []
    for train_metric in train_metrics:
      summary_ops.append(train_metric.tf_summaries(
          train_step=global_step, step_metrics=train_metrics[:2]))

    with eval_summary_writer.as_default(), \
         tf.compat.v2.summary.record_if(True):
      for eval_metric in eval_metrics:
        eval_metric.tf_summaries(train_step=global_step)

    init_agent_op = tf_agent.initialize()

    with tf.compat.v1.Session() as sess:
      # Initialize the graph.
      train_checkpointer.initialize_or_restore(sess)
      rb_checkpointer.initialize_or_restore(sess)
      # TODO(b/126239733): Remove once Periodically can be saved.
      common.initialize_uninitialized_variables(sess)

      sess.run(init_agent_op)
      sess.run(train_summary_writer.init())
      sess.run(eval_summary_writer.init())

      # Compute evaluation metrics.
      global_step_call = sess.make_callable(global_step)
      global_step_val = global_step_call()
      metric_utils.compute_summaries(
          eval_metrics,
          eval_py_env,
          eval_py_policy,
          num_episodes=num_eval_episodes,
          global_step=global_step_val,
          callback=eval_metrics_callback,
      )

      collect_call = sess.make_callable(collect_op)
      train_step_call = sess.make_callable([train_op, summary_ops])
      clear_rb_call = sess.make_callable(clear_rb_op)

      timed_at_step = global_step_call()
      time_acc = 0
      steps_per_second_ph = tf.compat.v1.placeholder(
          tf.float32, shape=(), name='steps_per_sec_ph')
      steps_per_second_summary = tf.compat.v2.summary.scalar(
          name='global_steps_per_sec', data=steps_per_second_ph,
          step=global_step)

      for _ in range(num_iterations):
        start_time = time.time()
        collect_call()
        total_loss, _ = train_step_call()
        clear_rb_call()
        time_acc += time.time() - start_time
        global_step_val = global_step_call()

        if global_step_val % log_interval == 0:
          logging.info('step = %d, loss = %f', global_step_val, total_loss.loss)
          steps_per_sec = (global_step_val - timed_at_step) / time_acc
          logging.info('%.3f steps/sec', steps_per_sec)
          sess.run(
              steps_per_second_summary,
              feed_dict={steps_per_second_ph: steps_per_sec})
          timed_at_step = global_step_val
          time_acc = 0

        if global_step_val % train_checkpoint_interval == 0:
          train_checkpointer.save(global_step=global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
          policy_checkpointer.save(global_step=global_step_val)

        if global_step_val % rb_checkpoint_interval == 0:
          rb_checkpointer.save(global_step=global_step_val)

        if global_step_val % eval_interval == 0:
          metric_utils.compute_summaries(
              eval_metrics,
              eval_py_env,
              eval_py_policy,
              num_episodes=num_eval_episodes,
              global_step=global_step_val,
              callback=eval_metrics_callback,
          )


def main(_):
  tf.compat.v1.enable_resource_variables()
  logging.set_verbosity(logging.INFO)
  train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
