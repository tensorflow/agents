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

# Lint as: python3
"""Learner implementation for Agents. Refer to the examples dir."""

import os

import gin
import tensorflow.compat.v2 as tf

from tf_agents.experimental.train import interval_trigger
from tf_agents.utils import common

TRAIN_DIR = 'train'
POLICY_SAVED_MODEL_DIR = 'policies'
COLLECT_POLICY_SAVED_MODEL_DIR = 'collect_policy'
GREEDY_POLICY_SAVED_MODEL_DIR = 'greedy_policy'
RAW_POLICY_SAVED_MODEL_DIR = 'policy'
POLICY_CHECKPOINT_DIR = 'checkpoints'


@gin.configurable
class Learner(tf.Module):
  """Manages all the learning details needed when training an agent.

  These include:
    * Using distribution strategies correctly
    * Summaries
    * Checkpoints
    * Minimizing entering/exiting TF context:
        Especially in the case of TPUs scheduling a single TPU program to
        perform multiple train steps is critical for performance.
    * Generalizes the train call to be done correctly across CPU, GPU, or TPU
      executions managed by DistributionStrategies. This uses `strategy.run` and
      then makes sure to do a reduce operation over the `LossInfo` returned by
      the agent.
  """

  def __init__(self,
               root_dir,
               train_step,
               agent,
               experience_dataset_fn=None,
               after_train_step_fn=None,
               triggers=None,
               checkpoint_interval=100000,
               summary_interval=1000,
               max_checkpoints_to_keep=3,
               use_kwargs_in_agent_train=False,
               strategy=None):
    """Initializes a Learner instance.

    Args:
      root_dir: Main directory path where checkpoints, saved_models, and
        summaries will be written to.
      train_step: a scalar tf.int64 `tf.Variable` which will keep track of the
        number of train steps. This is used for artifacts created like
        summaries, or outputs in the root_dir.
      agent: `tf_agent.TFAgent` instance to train with.
      experience_dataset_fn: a function that will create an instance of a
        tf.data.Dataset used to sample experience for training. Required for
        using the Learner as is. Optional for subclass learners which take a new
        iterator each time when `learner.run` is called.
      after_train_step_fn: (Optional) callable of the form `fn(sample, loss)`
        which can be used for example to update priorities in a replay buffer
        where sample is pulled from the `experience_iterator` and loss is a
        `LossInfo` named tuple returned from the agent. This is called after
        every train step.
      triggers: List of callables of the form `trigger(train_step)`. After every
        `run` call every trigger is called with the current `train_step` value
        as an np scalar.
      checkpoint_interval: Number of train steps in between checkpoints. Note
        these are placed into triggers and so a check to generate a checkpoint
        only occurs after every `run` call. Set to -1 to disable.  This only
        takes care of the checkpointing the training process.  Policies must be
        explicitly exported through triggers
      summary_interval: Number of train steps in between summaries. Note these
        are placed into triggers and so a check to generate a checkpoint only
        occurs after every `run` call.
      max_checkpoints_to_keep: Maximum number of checkpoints to keep around.
        These are used to recover from pre-emptions when training.
      use_kwargs_in_agent_train: If True the experience from the replay buffer
        is passed into the agent as kwargs. This requires samples from the RB to
        be of the form `dict(experience=experience, kwarg1=kwarg1, ...)`. This
        is useful if you have an agent with a custom argspec.
      strategy: (Optional) `tf.distribute.Strategy` to use during training.
    """
    self._train_dir = os.path.join(root_dir, TRAIN_DIR)
    self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
        self._train_dir, flush_millis=10000)

    self.train_step = train_step
    self._agent = agent
    self.use_kwargs_in_agent_train = use_kwargs_in_agent_train
    self.strategy = strategy or tf.distribute.get_strategy()

    if experience_dataset_fn:
      with self.strategy.scope():
        dataset = self.strategy.experimental_distribute_datasets_from_function(
            lambda _: experience_dataset_fn())
        self._experience_iterator = iter(dataset)

    self.after_train_step_fn = after_train_step_fn
    self.triggers = triggers or []

    # Prevent autograph from going into the agent.
    self._agent.train = tf.autograph.experimental.do_not_convert(agent.train)

    with self.strategy.scope():
      agent.initialize()

    checkpoint_dir = os.path.join(self._train_dir, POLICY_CHECKPOINT_DIR)
    self._checkpointer = common.Checkpointer(
        checkpoint_dir,
        max_to_keep=max_checkpoints_to_keep,
        agent=self._agent,
        train_step=self.train_step)
    self._checkpointer.initialize_or_restore()  # pytype: disable=attribute-error

    self.triggers.append(self._get_checkpoint_trigger(checkpoint_interval))
    self.summary_interval = tf.constant(summary_interval, dtype=tf.int64)

  @property
  def train_step_numpy(self):
    """The current train_step.

    Returns:
      The current `train_step`. Note this will return a scalar numpy array which
      holds the `train_step` value when this was called.
    """
    return self.train_step.numpy()

  def _get_checkpoint_trigger(self, checkpoint_interval):
    if checkpoint_interval <= 0:
      return lambda _, force_trigger=False: None

    save_fn = lambda: self._checkpointer.save(self.train_step)
    return interval_trigger.IntervalTrigger(checkpoint_interval, save_fn)

  def run(self, iterations=1, iterator=None):
    """Runs `iterations` iterations of training.

    Args:
      iterations: Number of train iterations to perform per call to run. The
        iterations will be evaluated in a tf.while loop created by autograph.
        Final aggregated losses will be returned.
      iterator: The iterator to the dataset to use for training. If not
        specified, `self._experience_iterator` is used.

    Returns:
      The total loss computed before running the final step.
    """

    def _summary_record_if():
      return tf.math.equal(
          self.train_step % tf.constant(self.summary_interval), 0)

    with self.train_summary_writer.as_default(), \
         common.soft_device_placement(), \
         tf.compat.v2.summary.record_if(_summary_record_if), \
         self.strategy.scope():
      iterator = iterator or self._experience_iterator
      loss_info = self._train(iterations, iterator)

      train_step_val = self.train_step.numpy()
      for trigger in self.triggers:
        trigger(train_step_val)

      return loss_info

  # Use tf.config.experimental_run_functions_eagerly(True) if you want to
  # disable use of tf.function.
  @common.function(autograph=True)
  def _train(self, iterations, iterator):
    assert iterations >= 1, (
        'Iterations must be greater or equal to 1, was %d' % iterations)
    # Call run explicitly once to get loss info shape for autograph. Because the
    # for loop below will get converted to a `tf.while_loop` by autograph we
    # need the shape of loss info to be well defined.
    loss_info = self.single_train_step(iterator)

    for _ in tf.range(iterations - 1):
      loss_info = self.single_train_step(iterator)

    def _reduce_loss(loss):
      return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

    # We assume all data can be reduced in the loss_info. This means no
    # string dtypes are currently allowed as LossInfo Fields.
    reduced_loss_info = tf.nest.map_structure(_reduce_loss, loss_info)
    return reduced_loss_info

  def single_train_step(self, iterator):
    (experience, sample_info) = next(iterator)

    if self.use_kwargs_in_agent_train:
      loss_info = self.strategy.run(self._agent.train, kwargs=experience)
    else:
      loss_info = self.strategy.run(self._agent.train, args=(experience,))

    if self.after_train_step_fn:
      self.after_train_step_fn((experience, sample_info), loss_info)

    return loss_info
