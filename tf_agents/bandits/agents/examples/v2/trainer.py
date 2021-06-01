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

r"""Generic TF-Agents training function for bandits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

tf = tf.compat.v2

AGENT_CHECKPOINT_NAME = 'agent'
STEP_CHECKPOINT_NAME = 'step'
CHECKPOINT_FILE_PREFIX = 'ckpt'


def get_replay_buffer(data_spec,
                      batch_size,
                      steps_per_loop):
  """Return a `TFUniformReplayBuffer` for the given `agent`."""
  buf = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=data_spec,
      batch_size=batch_size,
      max_length=steps_per_loop)
  return buf


def set_expected_shape(experience, num_steps):
  def set_time_dim(input_tensor, steps):
    tensor_shape = input_tensor.shape.as_list()
    tensor_shape[1] = steps
    input_tensor.set_shape(tensor_shape)
  tf.nest.map_structure(lambda t: set_time_dim(t, num_steps), experience)


def get_training_loop_fn(driver, replay_buffer, agent, steps):
  """Returns a `tf.function` that runs the driver and training loops.

  Args:
    driver: an instance of `Driver`.
    replay_buffer: an instance of `ReplayBuffer`.
    agent: an instance of `TFAgent`.
    steps: an integer indicating how many driver steps should be
      executed and presented to the trainer during each training loop.
  """
  def training_loop():
    """Returns a `tf.function` that runs the training loop."""
    driver.run()
    batch_size = driver.env.batch_size
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=steps,
        single_deterministic_pass=True)
    experience, unused_info = tf.data.experimental.get_single_element(dataset)
    set_expected_shape(experience, steps)
    loss_info = agent.train(experience)
    replay_buffer.clear()
    return loss_info
  return training_loop


def restore_and_get_checkpoint_manager(root_dir, agent, metrics, step_metric):
  """Restores from `root_dir` and returns a function that writes checkpoints."""
  trackable_objects = {metric.name: metric for metric in metrics}
  trackable_objects[AGENT_CHECKPOINT_NAME] = agent
  trackable_objects[STEP_CHECKPOINT_NAME] = step_metric
  checkpoint = tf.train.Checkpoint(**trackable_objects)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                  directory=root_dir,
                                                  max_to_keep=5)
  latest = checkpoint_manager.latest_checkpoint
  if latest is not None:
    logging.info('Restoring checkpoint from %s.', latest)
    checkpoint.restore(latest)
    logging.info('Successfully restored to step %s.', step_metric.result())
  else:
    logging.info('Did not find a pre-existing checkpoint. '
                 'Starting from scratch.')
  return checkpoint_manager


def train(root_dir,
          agent,
          environment,
          training_loops,
          steps_per_loop,
          additional_metrics=(),
          training_data_spec_transformation_fn=None):
  """Perform `training_loops` iterations of training.

  Checkpoint results.

  If one or more baseline_reward_fns are provided, the regret is computed
  against each one of them. Here is example baseline_reward_fn:

  def baseline_reward_fn(observation, per_action_reward_fns):
   rewards = ... # compute reward for each arm
   optimal_action_reward = ... # take the maximum reward
   return optimal_action_reward

  Args:
    root_dir: path to the directory where checkpoints and metrics will be
      written.
    agent: an instance of `TFAgent`.
    environment: an instance of `TFEnvironment`.
    training_loops: an integer indicating how many training loops should be run.
    steps_per_loop: an integer indicating how many driver steps should be
      executed and presented to the trainer during each training loop.
    additional_metrics: Tuple of metric objects to log, in addition to default
      metrics `NumberOfEpisodes`, `AverageReturnMetric`, and
      `AverageEpisodeLengthMetric`.
    training_data_spec_transformation_fn: Optional function that transforms the
    data items before they get to the replay buffer.
  """

  # TODO(b/127641485): create evaluation loop with configurable metrics.
  if training_data_spec_transformation_fn is None:
    data_spec = agent.policy.trajectory_spec
  else:
    data_spec = training_data_spec_transformation_fn(
        agent.policy.trajectory_spec)
  replay_buffer = get_replay_buffer(data_spec, environment.batch_size,
                                    steps_per_loop)

  # `step_metric` records the number of individual rounds of bandit interaction;
  # that is, (number of trajectories) * batch_size.
  step_metric = tf_metrics.EnvironmentSteps()
  metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
  ] + list(additional_metrics)

  if isinstance(environment.reward_spec(), dict):
    metrics += [tf_metrics.AverageReturnMultiMetric(
        reward_spec=environment.reward_spec(),
        batch_size=environment.batch_size)]
  else:
    metrics += [
        tf_metrics.AverageReturnMetric(batch_size=environment.batch_size)]

  if training_data_spec_transformation_fn is not None:
    add_batch_fn = lambda data: replay_buffer.add_batch(  # pylint: disable=g-long-lambda
        training_data_spec_transformation_fn(data))
  else:
    add_batch_fn = replay_buffer.add_batch

  observers = [add_batch_fn, step_metric] + metrics

  driver = dynamic_step_driver.DynamicStepDriver(
      env=environment,
      policy=agent.collect_policy,
      num_steps=steps_per_loop * environment.batch_size,
      observers=observers)

  training_loop = get_training_loop_fn(
      driver, replay_buffer, agent, steps_per_loop)
  checkpoint_manager = restore_and_get_checkpoint_manager(
      root_dir, agent, metrics, step_metric)
  train_step_counter = tf.compat.v1.train.get_or_create_global_step()
  saver = policy_saver.PolicySaver(agent.policy, train_step=train_step_counter)

  summary_writer = tf.summary.create_file_writer(root_dir)
  summary_writer.set_as_default()
  for i in range(training_loops):
    training_loop()
    metric_utils.log_metrics(metrics)
    for metric in metrics:
      metric.tf_summaries(train_step=step_metric.result())
    checkpoint_manager.save()
    if i % 100 == 0:
      saver.save(os.path.join(root_dir, 'policy_%d' % step_metric.result()))
