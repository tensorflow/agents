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

# Lint as: python3
"""PPO Learner implementation."""
from typing import Callable, Optional, Text

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


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
               root_dir: Text,
               train_step: tf.Variable,
               agent: ppo_agent.PPOAgent,
               experience_dataset_fn: Callable[..., tf.data.Dataset],
               normalization_dataset_fn: Callable[..., tf.data.Dataset],
               num_batches: int,
               num_epochs: int = 1,
               minibatch_size: Optional[int] = None,
               shuffle_buffer_size: Optional[int] = None,
               after_train_strategy_step_fn: Optional[Callable[
                   [types.NestedTensor, tf_agent.LossInfo], None]] = None,
               triggers: Optional[Callable[..., None]] = None,
               checkpoint_interval: int = 100000,
               summary_interval: int = 1000,
               use_kwargs_in_agent_train: bool = False,
               strategy: Optional[tf.distribute.Strategy] = None):
    """Initializes a PPOLearner instance.

    ```python
    agent = ppo_agent.PPOAgent(...,
      compute_value_and_advantage_in_train=False,
      # Skips updating normalizers in the agent, as it's handled in the learner.
      update_normalizers_in_train=False)

    # train_replay_buffer and normalization_replay_buffer point to two Reverb
    # tables that are synchronized. Sampling is done in a FIFO fashion.
    def experience_dataset_fn():
      return train_replay_buffer.as_dataset(sample_batch_size,
        sequence_preprocess_fn=agent.preprocess_sequence)
    def normalization_dataset_fn():
      return normalization_replay_buffer.as_dataset(sample_batch_size,
        sequence_preprocess_fn=agent.preprocess_sequence)

    learner = PPOLearner(..., agent, experience_dataset_fn,
      normalization_dataset_fn)
    learner.run()
    ```

    Args:
      root_dir: Main directory path where checkpoints, saved_models, and
        summaries will be written to.
      train_step: a scalar tf.int64 `tf.Variable` which will keep track of the
        number of train steps. This is used for artifacts created like
        summaries, or outputs in the root_dir.
      agent: `ppo_agent.PPOAgent` instance to train with. Note that
        update_normalizers_in_train should be set to `False`, otherwise a
        ValueError will be raised. We do not update normalizers in the agent
        again because we already update it in the learner. When mini batching is
        enabled, compute_value_and_advantage_in_train should be set to False,
        and preprocessing should be done as part of the data pipeline as part of
        `replay_buffer.as_dataset`.
      experience_dataset_fn: a function that will create an instance of a
        tf.data.Dataset used to sample experience for training. Each element
        in the dataset is a (Trajectory, SampleInfo) pair.
      normalization_dataset_fn: a function that will create an instance of a
        tf.data.Dataset used for normalization. This dataset is often from a
        separate reverb table that is synchronized with the table used in
        experience_dataset_fn. Each element in the dataset is a (Trajectory,
        SampleInfo) pair.
      num_batches: The number of batches to sample for training and
        normalization. If fewer than this amount of batches exists in the
        dataset, the learner will wait for more data to be added, or until the
        reverb timeout is reached.
      num_epochs: The number of iterations to go through the same sequences.
      minibatch_size: The minibatch size. The dataset used for training is
        shaped `[minibatch_size, 1, ...]`. If None, full sequences will be fed
        into the agent. Please set this parameter to None for RNN networks which
        requires full sequences.
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
        only occurs after every `run` call. Set to -1 to disable (this is not
        recommended, because it means that if the pipeline gets preempted, all
        previous progress is lost). This only takes care of the checkpointing
        the training process.  Policies must be explicitly exported through
        triggers.
      summary_interval: Number of train steps in between summaries. Note these
        are placed into triggers and so a check to generate a checkpoint only
        occurs after every `run` call.
      use_kwargs_in_agent_train: If True the experience from the replay buffer
        is passed into the agent as kwargs. This requires samples from the RB to
        be of the form `dict(experience=experience, kwarg1=kwarg1, ...)`. This
        is useful if you have an agent with a custom argspec.
      strategy: (Optional) `tf.distribute.Strategy` to use during training.

    Raises:
      ValueError:mini batching is enabled, but shuffle_buffer_size isn't
        provided.
      ValueError: minibatch_size is passed in for RNN networks. RNNs require
        full sequences.
      ValueError:mini batching is enabled, but
        agent._compute_value_and_advantage_in_train is set to `True`.
      ValueError: agent.update_normalizers_in_train or is set to `True`. The
        learner already updates the normalizers, so no need to update again in
        the agent.
    """
    if minibatch_size and shuffle_buffer_size is None:
      raise ValueError(
          'shuffle_buffer_size must be provided if minibatch_size is not None.')

    if minibatch_size and (agent._actor_net.state_spec or
                           agent._value_net.state_spec):
      raise ValueError('minibatch_size must be set to None for RNN networks.')

    if minibatch_size and agent._compute_value_and_advantage_in_train:
      raise ValueError(
          'agent.compute_value_and_advantage_in_train should be set to False '
          'when mini batching is used.')

    if agent.update_normalizers_in_train:
      raise ValueError(
          'agent.update_normalizers_in_train should be set to False when '
          'PPOLearner is used.'
      )

    strategy = strategy or tf.distribute.get_strategy()
    self._agent = agent
    self._minibatch_size = minibatch_size
    self._shuffle_buffer_size = shuffle_buffer_size
    self._num_epochs = num_epochs
    self._experience_dataset_fn = experience_dataset_fn
    self._normalization_dataset_fn = normalization_dataset_fn
    self._num_batches = num_batches

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

    self.num_replicas = strategy.num_replicas_in_sync
    self._create_datasets(strategy)
    self.num_frames_for_training = tf.Variable(0, dtype=tf.int32)

  def _create_datasets(self, strategy):
    """Create the training dataset and iterator."""

    def _make_dataset(_):
      train_dataset = self._experience_dataset_fn().take(self._num_batches)

      # We take the current batches, repeat for `num_epochs` times and exhaust
      # this data in the current learner run. The next time learner runs, new
      # batches of data will be sampled, cached and repeated.
      # This is enabled by the `Counter().flat_map()` trick below.
      train_dataset = train_dataset.cache().repeat(self._num_epochs)

      if self._minibatch_size:

        def squash_dataset_element(sequence, info):
          return tf.nest.map_structure(
              utils.BatchSquash(2).flatten, (sequence, info))

        # We unbatch the dataset shaped [B, T, ...] to a new dataset that
        # contains individual elements.
        # Note that we unbatch across the time dimension, which could result
        # in mini batches that contain subsets from more than one sequences.
        # PPO agent can handle mini batches across episode boundaries.
        train_dataset = train_dataset.map(squash_dataset_element).unbatch()
        train_dataset = train_dataset.shuffle(self._shuffle_buffer_size)
        train_dataset = train_dataset.batch(1, drop_remainder=True)
        train_dataset = train_dataset.batch(
            self._minibatch_size, drop_remainder=True)

      return train_dataset

    def make_dataset(_):
      return tf.data.experimental.Counter().flat_map(_make_dataset)

    with strategy.scope():

      if strategy.num_replicas_in_sync > 1:
        self._train_dataset = (
            strategy.distribute_datasets_from_function(make_dataset))
      else:
        self._train_dataset = make_dataset(0)
      self._train_iterator = iter(self._train_dataset)

  def run(self):
    """Train `num_batches` batches repeating for `num_epochs` of iterations.

    Returns:
      The total loss computed before running the final step.
    """
    self._normalization_iterator = iter(self._normalization_dataset_fn())
    num_frames = self._update_normalizers(self._normalization_iterator)
    self.num_frames_for_training.assign(num_frames)

    if self._minibatch_size:
      num_total_batches = int(self.num_frames_for_training.numpy() /
                              self._minibatch_size) * self._num_epochs
    else:
      num_total_batches = self._num_batches * self._num_epochs

    iterations = int(num_total_batches / self.num_replicas)
    loss_info = self._generic_learner.run(iterations, self._train_iterator)

    return loss_info

  @common.function(autograph=True)
  def _update_normalizers(self, iterator):
    """Update the normalizers and count the total number of frames."""

    reward_spec = tensor_spec.TensorSpec(shape=[], dtype=tf.float32)
    def _update(traj):
      self._agent.update_observation_normalizer(traj.observation)
      self._agent.update_reward_normalizer(traj.reward)
      if traj.reward.shape:

        outer_shape = nest_utils.get_outer_shape(traj.reward, reward_spec)
        batch_size = outer_shape[0]
        if len(outer_shape) > 1:
          batch_size *= outer_shape[1]
      else:
        batch_size = 1
      return batch_size

    num_frames = 0
    traj, _ = next(iterator)
    num_frames += _update(traj)

    for _ in tf.range(1, self._num_batches):
      traj, _ = next(iterator)
      num_frames += _update(traj)

    return num_frames

  @property
  def train_step_numpy(self):
    """The current train_step.

    Returns:
      The current `train_step`. Note this will return a scalar numpy array which
      holds the `train_step` value when this was called.
    """
    return self._generic_learner.train_step_numpy
