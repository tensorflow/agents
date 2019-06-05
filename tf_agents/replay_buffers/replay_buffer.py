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
import tensorflow as tf


class ReplayBuffer(tf.Module):
  """Abstract base class for TF-Agents replay buffer.

  In eager mode, methods modify the buffer or return values directly. In graph
  mode, methods return ops that do so when executed.
  """

  def __init__(self, data_spec, capacity):
    """Initializes the replay buffer.

    Args:
      data_spec: A spec or a list/tuple/nest of specs describing
        a single item that can be stored in this buffer
      capacity: number of elements that the replay buffer can hold.
    """
    super(ReplayBuffer, self).__init__()
    self._data_spec = data_spec
    self._capacity = capacity

  @property
  def data_spec(self):
    """Returns the spec for items in the replay buffer."""
    return self._data_spec

  @property
  def capacity(self):
    """Returns the capacity of the replay buffer."""
    return self._capacity

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

  def get_next(self,
               sample_batch_size=None,
               num_steps=None,
               time_stacked=True):
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

      Examples of tensor shapes returned:
        (B = batch size, T = timestep, D = data spec)

        get_next(sample_batch_size=None, num_steps=None, time_stacked=True)
          return shape (non-episodic): [D]
          return shape (episodic): [T, D] (T = full length of the episode)
        get_next(sample_batch_size=B, num_steps=None, time_stacked=True)
          return shape (non-episodic): [B, D]
          return shape (episodic): Not supported
        get_next(sample_batch_size=B, num_steps=T, time_stacked=True)
          return shape: [B, T, D]
        get_next(sample_batch_size=None, num_steps=T, time_stacked=False)
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
                 single_deterministic_pass=False):
    """Creates and returns a dataset that returns entries from the buffer.

    A single entry from the dataset is equivalent to one output from
    `get_next(sample_batch_size=sample_batch_size, num_steps=num_steps)`.

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
      single_deterministic_pass: Python boolean.  If `True`, the dataset
        will return a single deterministic pass through its underlying data.
        **NOTE**: If the buffer is modified while a Dataset iterator is
        iterating over this data, the iterator may miss any new data or
        otherwise have subtly invalid data.

    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).

    Raises:
      NotImplementedError: If a non-default argument value is not supported.
    """
    if single_deterministic_pass:
      return self._single_deterministic_pass_dataset(
          sample_batch_size=sample_batch_size,
          num_steps=num_steps,
          num_parallel_calls=num_parallel_calls)
    else:
      return self._as_dataset(
          sample_batch_size=sample_batch_size,
          num_steps=num_steps,
          num_parallel_calls=num_parallel_calls)

  def gather_all(self):
    """Returns all the items in buffer.

    **NOTE** This method will soon be deprecated in favor of
    `as_dataset(..., single_deterministic_pass=True)`.

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
  def _add_batch(self, items):
    """Adds a batch of items to the replay buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    """Returns an item or batch of items from the buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    """Creates and returns a dataset that returns entries from the buffer."""
    raise NotImplementedError

  @abc.abstractmethod
  def _single_deterministic_pass_dataset(self,
                                         sample_batch_size=None,
                                         num_steps=None,
                                         num_parallel_calls=None):
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
