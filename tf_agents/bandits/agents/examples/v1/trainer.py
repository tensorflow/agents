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

r"""Generic TF1 trainer for TF-Agents bandits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import logging

from gin.tf import utils
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common


def build_replay_buffer(agent, batch_size, steps_per_loop):
  """Return a `TFUniformReplayBuffer` for the given `agent`."""
  buf = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.policy.trajectory_spec,
      batch_size=batch_size,
      max_length=steps_per_loop)
  return buf


def train(
    root_dir,
    agent,
    environment,
    training_loops,
    steps_per_loop=1,
    additional_metrics=(),
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=10,
    policy_checkpoint_interval=10,
    log_interval=10,
    summary_interval=10):
  """A training driver."""

  if not common.resource_variables_enabled():
    raise RuntimeError(common.MISSING_RESOURCE_VARIABLES_ERROR)

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir)
  train_summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=environment.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=environment.batch_size),
    ] + list(additional_metrics)

    # Add to replay buffer and other agent specific observers.
    replay_buffer = build_replay_buffer(agent, environment.batch_size,
                                        steps_per_loop)
    agent_observers = [replay_buffer.add_batch] + train_metrics

    driver = dynamic_step_driver.DynamicStepDriver(
        env=environment,
        policy=agent.policy,
        num_steps=steps_per_loop * environment.batch_size,
        observers=agent_observers)

    collect_op, _ = driver.run()
    batch_size = driver.env.batch_size
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=steps_per_loop,
        single_deterministic_pass=True)
    trajectories, unused_info = tf.data.experimental.get_single_element(dataset)
    train_op = agent.train(experience=trajectories)
    clear_replay_op = replay_buffer.clear()

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        max_to_keep=1,
        agent=agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        max_to_keep=None,
        policy=agent.policy,
        global_step=global_step)

    summary_ops = []
    for train_metric in train_metrics:
      summary_ops.append(
          train_metric.tf_summaries(
              train_step=global_step, step_metrics=train_metrics[:2]))

    init_agent_op = agent.initialize()

    config_saver = utils.GinConfigSaverHook(train_dir, summarize_config=True)
    config_saver.begin()

    with tf.compat.v1.Session() as sess:
      # Initialize the graph.
      train_checkpointer.initialize_or_restore(sess)
      common.initialize_uninitialized_variables(sess)

      config_saver.after_create_session(sess)

      global_step_call = sess.make_callable(global_step)
      global_step_val = global_step_call()

      sess.run(train_summary_writer.init())
      sess.run(collect_op)

      if global_step_val == 0:
        # Save an initial checkpoint so the evaluator runs for global_step=0.
        policy_checkpointer.save(global_step=global_step_val)
        sess.run(init_agent_op)

      collect_call = sess.make_callable(collect_op)
      train_step_call = sess.make_callable([train_op, summary_ops])
      clear_replay_call = sess.make_callable(clear_replay_op)

      timed_at_step = global_step_val
      time_acc = 0
      steps_per_second_ph = tf.compat.v1.placeholder(
          tf.float32, shape=(), name='steps_per_sec_ph')
      steps_per_second_summary = tf.compat.v2.summary.scalar(
          name='global_steps_per_sec',
          data=steps_per_second_ph,
          step=global_step)

      for _ in range(training_loops):
        # Collect and train.
        start_time = time.time()
        collect_call()
        total_loss, _ = train_step_call()
        clear_replay_call()
        global_step_val = global_step_call()

        time_acc += time.time() - start_time

        total_loss = total_loss.loss

        if global_step_val % log_interval == 0:
          logging.info('step = %d, loss = %f', global_step_val, total_loss)
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
