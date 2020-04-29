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

"""TF-Agents Replay Buffer API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import common

from tensorflow.python.data.util import nest as data_nest  # pylint:disable=g-direct-tensorflow-import  # TF internal
from tensorflow.python.util import deprecation   # pylint:disable=g-direct-tensorflow-import  # TF internal


class ReplayBuffer(tf.Module):
  """Abstract base class for TF-Agents replay buffer.

  In eager mode, methods modify the buffer or return values directly. In graph
  mode, methods return ops that do so when executed.
  """

  def __init__(self, data_spec, capacity, stateful_dataset=False):
    """Initializes the replay buffer.

    Args:
      data_spec: A spec or a list/tuple/nest of specs describing a single item
        that can be stored in this buffer
      capacity: number of elements that the replay buffer can hold.
      stateful_dataset: whether the dataset contains stateful ops or not.
    """
    super(ReplayBuffer, self).__init__()
    common.check_tf1_allowed()
    self._data_spec = data_spec
    self._capacity = capacity
    self._stateful_dataset = stateful_dataset

  @property
  def data_spec(self):
    """Returns the spec for items in the replay buffer."""
    return self._data_spec

  @property
  def capacity(self):
    """Returns the capacity of the replay buffer."""
    return self._capacity

  @property
  def stateful_dataset(self):
    """Returns whether the dataset of the replay buffer has stateful ops."""
    return self._stateful_dataset

  def num_frames(self):
    """Returns the number of frames in the replay buffer."""
    return self._num_frames()

  def add_batch(self, items):
    """Adds a batch of items to the replay buffer.

    Args:
      items: An item or list/tuple/nest of items to be added to the replay
        buffer. `items` must match the data_spec of this class, with a
        batch_size dimension added to the beginning of each tensor/array.

    Returns:
      Adds `items` to the replay buffer.
    """
    return self._add_batch(items)

  @deprecation.deprecated(
      date=None,
      instructions=(
          'Use `as_dataset(..., single_deterministic_pass=False) instead.'
      ))
  def get_next(self, sample_batch_size=None, num_steps=None, time_stacked=True):
    """Returns an item or batch of items from the buffer.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. If None (default), a single item is returned
        which matches the data_spec of this class (without a batch dimension).
        Otherwise, a batch of sample_batch_size items is returned, where each
        tensor in items will have its first dimension equal to sample_batch_size
        and the rest of the dimensions match the corresponding data_spec. See
        examples below.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. If None (default), in non-episodic replay buffers, a batch of
        single items is returned. In episodic buffers, full episodes are
        returned (note that sample_batch_size must be None in that case).
        Otherwise, a batch of sub-episodes is returned, where a sub-episode is a
        sequence of consecutive items in the replay_buffer. The returned tensors
        will have first dimension equal to sample_batch_size (if
        sample_batch_size is not None), subsequent dimension equal to num_steps,
        if time_stacked=True and remaining dimensions which match the data_spec
        of this class. See examples below.
      time_stacked: (Optional.) Boolean, when true and num_steps > 1 it returns
        the items stacked on the time dimension. See examples below for details.
      Examples of tensor shapes returned: (B = batch size, T = timestep, D =
        data spec)  get_next(sample_batch_size=None, num_steps=None,
        time_stacked=True)
          return shape (non-episodic): [D]
          return shape (episodic): [T, D] (T = full length of the episode)
            get_next(sample_batch_size=B, num_steps=None, time_stacked=True)
          return shape (non-episodic): [B, D]
          return shape (episodic): Not supported get_next(sample_batch_size=B,
            num_steps=T, time_stacked=True)
          return shape: [B, T, D] get_next(sample_batch_size=None, num_steps=T,
            time_stacked=False)
          return shape: ([D], [D], ..) T tensors in the tuple
            get_next(sample_batch_size=B, num_steps=T, time_stacked=False)
          return shape: ([B, D], [B, D], ..) T tensors in the tuple

    Returns:
      A 2-tuple containing:
        - An item or sequence of (optionally batched and stacked) items.
        - Auxiliary info for the items (i.e. ids, probs).
    """
    return self._get_next(sample_batch_size, num_steps, time_stacked)

  def as_dataset(self,
                 sample_batch_size=None,
                 num_steps=None,
                 num_parallel_calls=None,
                 sequence_preprocess_fn=None,
                 single_deterministic_pass=False):
    """Creates and returns a dataset that returns entries from the buffer.

    A single entry from the dataset is the result of the following pipeline:

      * Sample sequences from the underlying data store
      * (optionally) Process them with `sequence_preprocess_fn`,
      * (optionally) Split them into subsequences of length `num_steps`
      * (optionally) Batch them into batches of size `sample_batch_size`.

    In practice, this pipeline is executed in parallel as much as possible
    if `num_parallel_calls != 1`.

    Some additional notes:

    If `num_steps is None`, different replay buffers will behave differently.
    For example, `TFUniformReplayBuffer` will return single time steps without
    a time dimension.  In contrast, e.g., `EpisodicReplayBuffer` will return
    full sequences (since each sequence may be an episode of unknown length,
    the outermost shape dimension will be `None`).

    If `sample_batch_size is None`, no batching is performed; and there is no
    outer batch dimension in the returned Dataset entries.  This setting
    is useful with variable episode lengths using e.g. `EpisodicReplayBuffer`,
    because it allows the user to get full episodes back, and use `tf.data`
    to build padded or truncated batches themselves.

    If `single_determinsitic_pass == True`, the replay buffer will make
    every attempt to ensure every time step is visited once and exactly once
    in a deterministic manner (though true determinism depends on the
    underlying data store).  Additional work may be done to ensure minibatches
    do not have multiple rows from the same episode.  In some cases, this
    may mean arguments like `num_parallel_calls` are ignored.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. If None (default), a single item is returned
        which matches the data_spec of this class (without a batch dimension).
        Otherwise, a batch of sample_batch_size items is returned, where each
        tensor in items will have its first dimension equal to sample_batch_size
        and the rest of the dimensions match the corresponding data_spec.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. If None (default), a batch of single items is returned.
        Otherwise, a batch of sub-episodes is returned, where a sub-episode is a
        sequence of consecutive items in the replay_buffer. The returned tensors
        will have first dimension equal to sample_batch_size (if
        sample_batch_size is not None), subsequent dimension equal to num_steps,
        and remaining dimensions which match the data_spec of this class.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process in parallel. If not
        specified, elements will be processed sequentially.
      sequence_preprocess_fn: (Optional) fn for preprocessing the collected
        data before it is split into subsequences of length `num_steps`.
        Defined in `TFAgent.preprocess_sequence`.  Defaults to pass through.
      single_deterministic_pass: Python boolean.  If `True`, the dataset will
        return a single deterministic pass through its underlying data.

        **NOTE**: If the buffer is modified while a Dataset iterator is
        iterating over this data, the iterator may miss any new data or
        otherwise have subtly invalid data.

    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:

        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).

    Raises:
      NotImplementedError: If a non-default argument value is not supported.
      ValueError: If the data spec contains lists that must be converted to
        tuples.
    """
    # data_tf.nest.flatten does not flatten python lists, nest.flatten does.
    if tf.nest.flatten(self._data_spec) != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  {}'.format(
              tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

    if single_deterministic_pass:
      ds = self._single_deterministic_pass_dataset(
          sample_batch_size=sample_batch_size,
          num_steps=num_steps,
          sequence_preprocess_fn=sequence_preprocess_fn,
          num_parallel_calls=num_parallel_calls)
    else:
      ds = self._as_dataset(
          sample_batch_size=sample_batch_size,
          num_steps=num_steps,
          sequence_preprocess_fn=sequence_preprocess_fn,
          num_parallel_calls=num_parallel_calls)

    if self._stateful_dataset:
      options = tf.data.Options()
      if hasattr(options, 'experimental_allow_stateful'):
        options.experimental_allow_stateful = True
        ds = ds.with_options(options)
    return ds

  @deprecation.deprecated(
      date=None,
      instructions=(
          'Use `as_dataset(..., single_deterministic_pass=True)` instead.'
      ))
  def gather_all(self):
    """Returns all the items in buffer.

    Returns:
      Returns all the items currently in the buffer. Returns a tensor
      of shape [B, T, ...] where B = batch size, T = timesteps,
      and the remaining shape is the shape spec of the items in the buffer.
    """
    return self._gather_all()

  def clear(self):
    """Resets the contents of replay buffer.

    Returns:
      Clears the replay buffer contents.
    """
    return self._clear()

  # Subclasses must implement these methods.
  @abc.abstractmethod
  def _num_frames(self):
    """Returns the number of frames in the replay buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _add_batch(self, items):
    """Adds a batch of items to the replay buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _get_next(self, sample_batch_size, num_steps, time_stacked):
    """Returns an item or batch of items from the buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _as_dataset(self,
                  sample_batch_size,
                  num_steps,
                  sequence_preprocess_fn,
                  num_parallel_calls):
    """Creates and returns a dataset that returns entries from the buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _single_deterministic_pass_dataset(self,
                                         sample_batch_size,
                                         num_steps,
                                         sequence_preprocess_fn,
                                         num_parallel_calls):
    """Creates and returns a dataset that returns entries from the buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _gather_all(self):
    """Returns all the items in buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _clear(self):
    """Clears the replay buffer."""
    raise NotImplementedError
