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
"""PPO Learner implementation."""

import gin
import tensorflow.compat.v2 as tf

from tf_agents.experimental.train import learner
from tf_agents.networks import utils
from tf_agents.utils import common


@gin.configurable
class PPOLearner(object):
  """Manages all the learning details needed when training an PPO.

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
               max_num_sequences=None,
               minibatch_size=None,
               shuffle_buffer_size=None,
               after_train_strategy_step_fn=None,
               triggers=None,
               checkpoint_interval=100000,
               summary_interval=1000,
               use_kwargs_in_agent_train=False,
               strategy=None):
    """Initializes a PPOLearner instance.

    Args:
      root_dir: Main directory path where checkpoints, saved_models, and
        summaries will be written to.
      train_step: a scalar tf.int64 `tf.Variable` which will keep track of the
        number of train steps. This is used for artifacts created like
        summaries, or outputs in the root_dir.
      agent: `tf_agent.TFAgent` instance to train with.
      max_num_sequences: The max number of sequences to read from the input
        dataset in `run`. Defaults to None, in which case `run` will terminate
        when reach the end of the dataset (for instance when the rate limiter
        times out).
      minibatch_size: The minibatch size. The dataset used for training is
        shaped [minibatch_size, 1, ...].
      shuffle_buffer_size: The buffer size for shuffling the trajectories before
        splitting them into mini batches. Only required when mini batch
        learning is enabled (minibatch_size is set). Otherwise it is ignored.
        Commonly set to a number 1-3x the episode length of your environment.
      after_train_strategy_step_fn: (Optional) callable of the form
        `fn(sample, loss)` which can be used for example to update priorities in
        a replay buffer where sample is pulled from the `experience_iterator`
        and loss is a `LossInfo` named tuple returned from the agent. This is
        called after every train step. It runs using `strategy.run(...)`.
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
      use_kwargs_in_agent_train: If True the experience from the replay buffer
        is passed into the agent as kwargs. This requires samples from the RB to
        be of the form `dict(experience=experience, kwarg1=kwarg1, ...)`. This
        is useful if you have an agent with a custom argspec.
      strategy: (Optional) `tf.distribute.Strategy` to use during training.
    """
    if minibatch_size is not None and shuffle_buffer_size is None:
      raise ValueError(
          'shuffle_buffer_size must be provided if minibatch_size is not None.'
      )

    if agent.update_normalizers_in_train:
      raise ValueError(
          'agent.update_normalizers_in_train should be set to False when '
          'PPOLearner is used.'
      )

    self._agent = agent
    self._max_num_sequences = max_num_sequences
    self._minibatch_size = minibatch_size
    self._shuffle_buffer_size = shuffle_buffer_size

    self._generic_learner = learner.Learner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn=None,
        after_train_strategy_step_fn=after_train_strategy_step_fn,
        triggers=triggers,
        checkpoint_interval=checkpoint_interval,
        summary_interval=summary_interval,
        use_kwargs_in_agent_train=use_kwargs_in_agent_train,
        strategy=strategy)

  def run(self, iterations, dataset):
    """Runs training until dataset timesout, or when num sequences is reached.

    Args:
      iterations: Number of iterations/epochs to repeat over the collected
        sequences. (Schulman,2017) sets this to 10 for Mujoco, 15 for Roboschool
         and 3 for Atari.
      dataset: A 'tf.Dataset' where each sample is shaped
        [sample_batch_size, sequence_length, ...], commonly the output from
        'reverb_replay_buffer.as_dataset(sample_batch_size, preprocess_fn)'.

    Returns:
      The total loss computed before running the final step.
    """
    # TODO(b/160802425): Verify this setup works with distributed.
    if self._max_num_sequences:
      dataset = dataset.take(self._max_num_sequences)
    cached_dataset = dataset.cache()
    self._update_advantage_normalizer(cached_dataset)

    new_dataset = cached_dataset.repeat(iterations)
    if self._minibatch_size:

      def squash_dataset_element(sequence, info):
        return tf.nest.map_structure(
            utils.BatchSquash(2).flatten, (sequence, info))

      # We unbatch the dataset shaped [B, T, ...] to a new dataset that contains
      # individual elements.
      # Note that we unbatch across the time dimension, which could result in
      # mini batches that contain subsets from more than one sequences. The PPO
      # agent can handle mini batches across episode boundaries.
      new_dataset = new_dataset.map(squash_dataset_element).unbatch()
      new_dataset = new_dataset.shuffle(self._shuffle_buffer_size)
      new_dataset = new_dataset.batch(1, drop_remainder=True)
      new_dataset = new_dataset.batch(self._minibatch_size, drop_remainder=True)

    # TODO(b/161133726): use learner.run once it supports None iterations.
    def _summary_record_if():
      return tf.math.equal(
          self._generic_learner.train_step %
          tf.constant(self._generic_learner.summary_interval), 0)

    with self._generic_learner.train_summary_writer.as_default(), \
     common.soft_device_placement(), \
     tf.compat.v2.summary.record_if(_summary_record_if), \
     self._generic_learner.strategy.scope():
      loss_info = self.multi_train_step(iter(new_dataset))

      train_step_val = self._generic_learner.train_step_numpy
      for trigger in self._generic_learner.triggers:
        trigger(train_step_val)

    self._update_normalizers(cached_dataset)

    return loss_info

  @common.function(autograph=True)
  def multi_train_step(self, iterator):
    experience, sample_info = next(iterator)

    loss_info = self.single_train_step(experience, sample_info)
    for experience, sample_info in iterator:
      loss_info = self.single_train_step(experience, sample_info)
    return loss_info

  @common.function(autograph=False)
  def single_train_step(self, experience, sample_info):
    """Train a single (mini) batch of Trajectories."""
    if self._generic_learner.use_kwargs_in_agent_train:
      loss_info = self._generic_learner.strategy.run(
          self._agent.train, kwargs=experience)
    else:
      loss_info = self._generic_learner.strategy.run(
          self._agent.train, args=(experience,))

    if self._generic_learner.after_train_strategy_step_fn:
      if self.use_kwargs_in_agent_train:
        self.strategy.run(
            self._generic_learner.after_train_strategy_step_fn,
            kwargs=dict(
                experience=(experience, sample_info), loss_info=loss_info))
      else:
        self.strategy.run(
            self._generic_learner.after_train_strategy_step_fn,
            args=((experience, sample_info), loss_info))

    return loss_info

  @common.function(autograph=True)
  def _update_normalizers(self, dataset):
    iterator = iter(dataset)
    traj, _ = next(iterator)
    self._agent.update_observation_normalizer(traj.observation)
    self._agent.update_reward_normalizer(traj.reward)

    for traj, _ in iterator:
      self._agent.update_observation_normalizer(traj.observation)
      self._agent.update_reward_normalizer(traj.reward)

  @common.function(autograph=True)
  def _update_advantage_normalizer(self, dataset):
    self._agent._reset_advantage_normalizer()  # pylint: disable=protected-access

    iterator = iter(dataset)
    traj, _ = next(iterator)
    self._agent._update_advantage_normalizer(traj.policy_info['advantage'])  # pylint: disable=protected-access

    for traj, _ in iterator:
      self._agent._update_advantage_normalizer(traj.policy_info['advantage'])  # pylint: disable=protected-access

  @property
  def train_step_numpy(self):
    """The current train_step.

    Returns:
      The current `train_step`. Note this will return a scalar numpy array which
      holds the `train_step` value when this was called.
    """
    return self._generic_learner.train_step_numpy
