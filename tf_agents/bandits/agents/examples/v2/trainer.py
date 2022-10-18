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

"""Generic TF-Agents training function for bandits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.replay_buffers import bandit_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import export_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver

tf = tf.compat.v2

AGENT_CHECKPOINT_NAME = 'agent'
STEP_CHECKPOINT_NAME = 'step'
CHECKPOINT_FILE_PREFIX = 'ckpt'


def export_metrics(step, metrics):
  """Exports metrics."""
  metric_utils.log_metrics(metrics)
  export_utils.export_metrics(step=step, metrics=metrics)
  for metric in metrics:
    metric.tf_summaries(train_step=step)


def export_loss_info(step, loss_info):
  """Exports loss info."""
  export_utils.export_metrics(step=step, metrics=[], loss_info=loss_info)


def get_replay_buffer(data_spec, batch_size, steps_per_loop,
                      async_steps_per_loop):
  """Return a `TFUniformReplayBuffer` for the given `agent`."""
  return bandit_replay_buffer.BanditReplayBuffer(
      data_spec=data_spec,
      batch_size=batch_size,
      max_length=steps_per_loop * async_steps_per_loop)


def set_expected_shape(experience, num_steps):
  """Sets expected shape."""
  def set_time_dim(input_tensor, steps):
    tensor_shape = input_tensor.shape.as_list()
    if len(tensor_shape) < 2:
      raise ValueError(
          'input_tensor is expected to be of rank-2, but found otherwise: '
          f'input_tensor={input_tensor}, tensor_shape={tensor_shape}')
    tensor_shape[1] = steps
    input_tensor.set_shape(tensor_shape)
  tf.nest.map_structure(lambda t: set_time_dim(t, num_steps), experience)


def get_training_loop(driver, replay_buffer, agent, steps,
                      async_steps_per_loop):
  """Returns a `tf.function` that runs the driver and training loops.

  Args:
    driver: an instance of `Driver`.
    replay_buffer: an instance of `ReplayBuffer`.
    agent: an instance of `TFAgent`.
    steps: an integer indicating how many driver steps should be
      executed and presented to the trainer during each training loop.
    async_steps_per_loop: an integer. In each training loop, the driver runs
      this many times, and then the agent gets asynchronously trained over this
      many batches sampled from the replay buffer.
  """

  def training_loop(train_step, metrics):
    """Returns a function that runs a single training loop and logs metrics."""
    for batch_id in range(async_steps_per_loop):
      driver.run()
      export_metrics(
          step=train_step * async_steps_per_loop + batch_id, metrics=metrics)
    batch_size = driver.env.batch_size
    dataset_it = iter(
        replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=steps,
            single_deterministic_pass=True))
    for batch_id in range(async_steps_per_loop):
      experience, unused_buffer_info = dataset_it.get_next()
      set_expected_shape(experience, steps)
      loss_info = agent.train(experience)
      export_loss_info(
          step=train_step * async_steps_per_loop + batch_id,
          loss_info=loss_info)

    replay_buffer.clear()

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
          async_steps_per_loop=None,
          additional_metrics=(),
          get_replay_buffer_fn=None,
          get_training_loop_fn=None,
          training_data_spec_transformation_fn=None,
          save_policy=True):
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
      executed in a single driver run.
    async_steps_per_loop: an optional integer for simulating offline or
      asynchronous training: In each training loop iteration, the driver runs
      this many times, each executing `steps_per_loop` driver steps, and then
      the agent gets asynchronously trained over this many batches sampled from
      the replay buffer. When unset or set to 1, the function performs
      synchronous training, where the agent gets trained on a single batch
      immediately after the driver runs.
    additional_metrics: Tuple of metric objects to log, in addition to default
      metrics `NumberOfEpisodes`, `AverageReturnMetric`, and
      `AverageEpisodeLengthMetric`.
    get_replay_buffer_fn: An optional function that creates a replay buffer by
      taking a data_spec, batch size, the number of driver steps per loop, and
      the number of asynchronous training steps per loop. Note that the returned
      replay buffer will be passed to `get_training_loop_fn` below to generate a
      traininig loop function. If `None`, the `get_replay_buffer` function
      defined in this module will be used.
    get_training_loop_fn: An optional function that constructs the traininig
      loop function executing a single train step. This function takes a driver,
      a replay buffer, an agent, the number of driver steps per loop, and the
      number of asynchronous training steps per loop. If `None`, the
      `get_training_loop` function defined in this module will be used.
    training_data_spec_transformation_fn: Optional function that transforms the
      data items before they get to the replay buffer.
    save_policy: (bool) whether to save the policy or not.
  """

  # TODO(b/127641485): create evaluation loop with configurable metrics.
  if training_data_spec_transformation_fn is None:
    data_spec = agent.policy.trajectory_spec
  else:
    data_spec = training_data_spec_transformation_fn(
        agent.policy.trajectory_spec)
  if async_steps_per_loop is None:
    async_steps_per_loop = 1
  if get_replay_buffer_fn is None:
    get_replay_buffer_fn = get_replay_buffer
  replay_buffer = get_replay_buffer_fn(data_spec, environment.batch_size,
                                       steps_per_loop, async_steps_per_loop)

  # `step_metric` records the number of individual rounds of bandit interaction;
  # that is, (number of trajectories) * batch_size.
  step_metric = tf_metrics.EnvironmentSteps()
  metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.AverageEpisodeLengthMetric(batch_size=environment.batch_size)
  ] + list(additional_metrics)

  # If the reward anything else than a single scalar, we're adding multimetric
  # average reward.
  if isinstance(environment.reward_spec(),
                dict) or environment.reward_spec().shape != tf.TensorShape(()):
    metrics += [tf_metrics.AverageReturnMultiMetric(
        reward_spec=environment.reward_spec(),
        batch_size=environment.batch_size)]
  if not isinstance(environment.reward_spec(), dict):
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

  if get_training_loop_fn is None:
    get_training_loop_fn = get_training_loop
  training_loop = get_training_loop_fn(driver, replay_buffer, agent,
                                       steps_per_loop, async_steps_per_loop)
  checkpoint_manager = restore_and_get_checkpoint_manager(
      root_dir, agent, metrics, step_metric)
  train_step_counter = tf.compat.v1.train.get_or_create_global_step()
  if save_policy:
    saver = policy_saver.PolicySaver(
        agent.policy, train_step=train_step_counter)

  summary_writer = tf.summary.create_file_writer(root_dir)
  summary_writer.set_as_default()

  for i in range(training_loops):
    training_loop(train_step=i, metrics=metrics)
    checkpoint_manager.save()
    if save_policy & (i % 100 == 0):
      saver.save(os.path.join(root_dir, 'policy_%d' % step_metric.result()))
