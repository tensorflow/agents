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

"""Utilities to load data for CQL."""
import functools
from typing import Any, Callable, MutableSequence, Optional, Text, Tuple
import numpy as np
import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import example_encoding
from tf_agents.utils import example_encoding_dataset

DecoderFnType = Callable[[types.Tensor], types.NestedTensor]


def updated_sample(sample: Any, reward_shift: float,
                   action_clipping: Optional[Tuple[float, float]],
                   use_trajectories: bool):
  """Create a sample with reward_shift and action_clipping."""

  def _clip_actions(actions):
    return tf.clip_by_value(
        actions,
        clip_value_min=action_clipping[0],
        clip_value_max=action_clipping[1])

  if use_trajectories:
    # Update trajectory.
    shifted_reward = sample.reward + reward_shift
    if action_clipping:
      return sample._replace(
          action=tf.nest.map_structure(_clip_actions, sample.action),
          reward=shifted_reward)
    else:
      return sample._replace(reward=shifted_reward)
  else:
    # Update transition.
    next_time_step = sample.next_time_step
    next_time_step = ts.TimeStep(
        step_type=next_time_step.step_type,
        reward=next_time_step.reward + reward_shift,
        discount=next_time_step.discount,
        observation=next_time_step.observation)
    action_step = sample.action_step
    if action_clipping:
      action_step = action_step._replace(
          action=tf.nest.map_structure(_clip_actions, action_step.action))
    return trajectory.Transition(
        time_step=sample.time_step,
        action_step=action_step,
        next_time_step=next_time_step)


def create_single_tf_record_dataset(
    filename: Text,
    load_buffer_size: int = 0,
    shuffle_buffer_size: int = 10000,
    num_parallel_reads: Optional[int] = None,
    decoder: Optional[DecoderFnType] = None,
    reward_shift: float = 0.0,
    action_clipping: Optional[Tuple[float, float]] = None,
    use_trajectories: bool = True,
):
  """Create a TF dataset for a single TFRecord file.

  Args:
    filename: Path to a single TFRecord file.
    load_buffer_size: Number of bytes in the read buffer. 0 means no buffering.
    shuffle_buffer_size: Size of the buffer for shuffling items within a single
      TFRecord file.
    num_parallel_reads: Optional, number of parallel reads in the TFRecord
      dataset. If not specified, no parallelism.
    decoder: Optional, a custom decoder to use rather than using the default
      spec path.
    reward_shift: Value to add to the reward.
    action_clipping: Optional, (minimum, maximum) values to clip actions.
    use_trajectories: Whether to use trajectories. If false, use transitions.

  Returns:
    A TF.Dataset of experiences.
  """
  dataset = example_encoding_dataset.load_tfrecord_dataset(
      filename,
      buffer_size=load_buffer_size,
      as_experience=use_trajectories,
      as_trajectories=use_trajectories,
      add_batch_dim=False,
      num_parallel_reads=num_parallel_reads,
      decoder=decoder,
  )

  def _sample_to_experience(sample):
    dummy_info = ()
    return updated_sample(sample, reward_shift, action_clipping,
                          use_trajectories), dummy_info

  dataset = dataset.map(
      _sample_to_experience,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def create_tf_record_dataset(
    filenames: MutableSequence[Text],
    batch_size: int,
    shuffle_buffer_size_per_record: int = 100,
    shuffle_buffer_size: int = 100,
    load_buffer_size: int = 100000000,
    num_shards: int = 50,
    cycle_length: int = tf.data.experimental.AUTOTUNE,
    block_length: int = 10,
    num_parallel_reads: Optional[int] = None,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    num_prefetch: int = 10,
    strategy: Optional[tf.distribute.Strategy] = None,
    reward_shift: float = 0.0,
    action_clipping: Optional[Tuple[float, float]] = None,
    use_trajectories: bool = True,
):
  """Create a TF dataset from a list of filenames.

  A dataset is created for each record file and these are interleaved together
  to create the final dataset.

  Args:
    filenames: List of filenames of a TFRecord dataset containing TF Examples.
    batch_size: The batch size of tensors in the returned dataset.
    shuffle_buffer_size_per_record: The buffer size used for shuffling within a
      Record file.
    shuffle_buffer_size: The shuffle buffer size for the interleaved dataset.
    load_buffer_size: Number of bytes in the read buffer. 0 means no buffering.
    num_shards: The number of shards, each consisting of 1 or more record file
      datasets, that are then interleaved together.
    cycle_length: The number of input elements processed concurrently while
      interleaving.
    block_length: The number of consecutive elements to produce from each input
      element before cycling to another input element.
    num_parallel_reads: Optional, number of parallel reads in the TFRecord
      dataset. If not specified, len(filenames) will be used.
    num_parallel_calls: Number of parallel calls for interleave.
    num_prefetch: Number of batches to prefetch.
    strategy: Optional, `tf.distribute.Strategy` being used in training.
    reward_shift: Value to add to the reward.
    action_clipping: Optional, (minimum, maximum) values to clip actions.
    use_trajectories: Whether to use trajectories. If false, use transitions.

  Returns:
    A TF.Dataset containing a batch of nested Tensors.
  """
  initial_len = len(filenames)
  remainder = initial_len % num_shards
  for _ in range(num_shards - remainder):
    filenames.append(filenames[np.random.randint(low=0, high=initial_len)])
  filenames = np.array(filenames)
  np.random.shuffle(filenames)
  filenames = np.array_split(filenames, num_shards)

  record_file_ds = tf.data.Dataset.from_tensor_slices(filenames)
  record_file_ds = record_file_ds.repeat().shuffle(len(filenames))

  spec_path = filenames[0][0] + '.spec'
  record_spec = example_encoding_dataset.parse_encoded_spec_from_file(spec_path)
  decoder = example_encoding.get_example_decoder(record_spec)

  example_ds = record_file_ds.interleave(
      functools.partial(
          create_single_tf_record_dataset,
          load_buffer_size=load_buffer_size,
          shuffle_buffer_size=shuffle_buffer_size_per_record,
          num_parallel_reads=num_parallel_reads,
          decoder=decoder,
          reward_shift=reward_shift,
          action_clipping=action_clipping,
          use_trajectories=use_trajectories
      ),
      cycle_length=cycle_length,
      block_length=block_length,
      num_parallel_calls=num_parallel_calls,
  )
  example_ds = example_ds.shuffle(shuffle_buffer_size)

  use_tpu = isinstance(
      strategy,
      (tf.distribute.experimental.TPUStrategy, tf.distribute.TPUStrategy))
  example_ds = example_ds.batch(
      batch_size, drop_remainder=use_tpu).prefetch(num_prefetch)
  return example_ds
