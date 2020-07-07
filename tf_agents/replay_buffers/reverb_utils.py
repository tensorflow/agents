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
"""Utilities for working with a reverb replay buffer."""


from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import numbers

from absl import logging
from six.moves import zip


class ReverbAddEpisodeObserver(object):
  """Observer for writing episodes to the Reverb replay buffer."""

  def __init__(self,
               py_client,
               table_name,
               max_sequence_length,
               priority,
               bypass_partial_episodes=False):
    """Creates an instance of the ReverbAddEpisodeObserver.

    **Note**: This observer is designed to work with py_drivers only, and does
    not support batches.

    TODO(b/158865335): Optionally truncate long episodes and add to buffer.

    Args:
      py_client: Python client for the reverb replay server.
      table_name: The table name where samples will be written to.
      max_sequence_length: An integer. `max_sequence_length` used
        to write to the replay buffer tables. This defines the size of the
        internal buffer controlling the `upper` limit of the number of timesteps
        which can be referenced in a single prioritized item. Note that this is
        the maximum number of trajectories across all the cached episodes that
        you are writing into the replay buffer (e.g. `number_of_episodes`).
        `max_sequence_length` is not a limit of how many timesteps or
        items that can be inserted into the replay buffer. Note that,
        since `max_sequence_length` controls the size of internal buffer, it is
        suggested not to set this value to a very large number. If the number of
        steps in an episode is more than `max_sequence_length`, only items up to
        `max_sequence_length` is written into the table.
      priority: Initial priority for the table.
      bypass_partial_episodes: If `False` (default) and an episode length is
        greater than `max_sequence_length`, a `ValueError` is raised. If set to
        `True`, the episodes with length more than `max_sequence_length` do not
        cause a `ValueError`. These episodes are bypassed (will NOT be written
        into the replay buffer) and an error message is shown to the user.
        Note that in this case (`bypass_partial_episodes=True`), the steps for
        episodes with length more than `max_sequence_length` are wasted and
        thrown away. This decision is made to guarantee that the replay buffer
        always has FULL episodes. Note that, `max_sequence_length` is just an
        upper bound.

    Raises:
      ValueError: If `table_name` is not a string.
      ValueError: If `priority` is not numeric.
      ValueError: If max_sequence_length is not positive.
    """
    if not isinstance(table_name, str):
      raise ValueError("`table_name` must be a string.")
    if not isinstance(priority, numbers.Number):
      raise ValueError("`priority` must be a numeric value.")
    if max_sequence_length <= 0:
      raise ValueError(
          "`max_sequence_length` must be an integer greater equal one.")

    self._table_name = table_name
    self._max_sequence_length = max_sequence_length
    self._priority = priority

    self._py_client = py_client
    self._writer = py_client.writer(
        max_sequence_length=self._max_sequence_length)
    self._cached_steps = 0
    self._bypass_partial_episodes = bypass_partial_episodes
    self._overflow_episode = False

  def update_priority(self, priority):
    """Update the table priority.

    Args:
      priority: Updates the priority of the observer.

    ValueError: If priority is not numeric.
    """
    if not isinstance(priority, numbers.Number):
      raise ValueError("`priority` must be a numeric value.")
    self._priority = priority

  def __call__(self, trajectory):
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.

    Raises:
      ValueError: If `bypass_partial_episodes` == False and episode length
        is > `max_sequence_length`.
    """
    if (self._cached_steps >= self._max_sequence_length and
        not self._overflow_episode):
      # The reason for moving forward even if there is an overflowed episode is
      # to capture the boundary of the episode and reset `_cached_steps`.
      self._overflow_episode = True
      if self._bypass_partial_episodes:
        logging.error(
            "The number of trajectories within the same episode exceeds "
            "`max_sequence_length`. This episode is bypassed and will NOT "
            "be written into the replay buffer. Consider increasing the "
            "`max_sequence_length`.")
      else:
        raise ValueError(
            "The number of trajectories within the same episode "
            "exceeds `max_sequence_length`. Consider increasing the "
            "`max_sequence_length` or set `bypass_partial_episodes` to true "
            "to bypass the episodes with length more than "
            "`max_sequence_length`.")

    if not self._overflow_episode:
      self._writer.append(trajectory)
      self._cached_steps += 1

    if trajectory.is_boundary():
      self._write_cached_steps()

  def _write_cached_steps(self):
    """Writes the cached steps into the writer.

    **Note**: The method resets the number of episodes and steps after writing
      the data into the replay buffer.
    """
    if not self._overflow_episode:
      self._writer.create_item(
          table=self._table_name,
          num_timesteps=self._cached_steps,
          priority=self._priority)
    self.reset()
    self._overflow_episode = False

  def reset(self):
    """Resets the state of the observer.

    The observed data (appended to the writer) will be written to RB after
    calling reset. Note that, each write creates a separate entry in the
    replay buffer.
    """
    self.close()
    self.open()

  def open(self):
    """Open the writer of the observer."""
    if self._writer is None:
      self._writer = self._py_client.writer(
          max_sequence_length=self._max_sequence_length)
      self._cached_steps = 0

  def close(self):
    """Closes the writer of the observer.

    **Note**: Using the observer after closing it is not supported.
    """

    if self._writer is not None:
      self._writer.close()
      self._writer = None


class ReverbAddTrajectoryObserver(object):
  """Stateful observer for writing to the Reverb replay.

  b/158373731: Simplify the observer to only support step insertion.
  """

  def __init__(self,
               py_client,
               table_names,
               # TODO(b/160653590): Only allow single sequence length.
               sequence_lengths,
               stride_lengths=None,
               priority=5,
               allow_multi_episode_sequences=False):
    """Creates an instance of the ReverbAddTrajectoryObserver.

    If multiple table_names and sequence lengths are provided data will only be
    stored once but be available for sampling with multiple sequence lengths
    from the respective reverb tables.

    **Note**: This observer is designed to work with py_drivers only, and does
    not support batches.

    Args:
      py_client: Python client for the reverb replay server.
      table_names: A list or tuple  of table names where samples will be written
        to.
      sequence_lengths: List or tuple of integer sequence_lengths used to write
        to the given tables. Must be the same length as the given table_names.
        Note that setting this to other than 1 (the default) can cause final
        transitions to not be written to the RB. This can easily break learning
        for certain kinds of environments.
      stride_lengths: List or tuple of integer strides for the sliding window
        for overlapping sequences.
      priority: Initial priority for new samples in the RB.
      allow_multi_episode_sequences: Allows sequences to go over episode
        boundaries. **NOTE**: Samples generated when data is collected with this
        flag set to True will contain episode boundaries which need to be
        handled by the user.

    Raises:
      ValueError: If table_names or sequence_lengths are not lists or their
      lengths are not equal.
    """
    if not isinstance(table_names, (list, tuple)):
      raise ValueError("`table_names` must be a list or a tuple.")
    if not isinstance(sequence_lengths, (list, tuple)):
      raise ValueError("`sequence_lengths` must be a list or a tuple.")
    if stride_lengths:
      if not isinstance(stride_lengths, (list, tuple)):
        raise ValueError("`stride_lengths` must be a list or a tuple.")
      if len(table_names) != len(stride_lengths):
        raise ValueError(
            "Length of table_names and stride_lengths must be equal. "
            "Got: {} and {}".format(len(table_names), len(stride_lengths)))
    if len(table_names) != len(sequence_lengths):
      raise ValueError(
          "Length of table_names and sequence_lengths must be equal. "
          "Got: {} and {}".format(len(table_names), len(sequence_lengths)))

    self._table_names = table_names
    self._sequence_lengths = sequence_lengths
    self._stride_lengths = stride_lengths or [1] * len(sequence_lengths)
    self._priority = priority
    self._allow_multi_episode_sequences = allow_multi_episode_sequences

    self._py_client = py_client
    # TODO(b/153700282): Use a single writer with max_sequence_length=max(...)
    # once Reverb Dataset with emit_timesteps=True returns properly shaped
    # sequences.
    self._writers = [
        py_client.writer(
            max_sequence_length=s_len) for s_len in sequence_lengths]
    self._cached_steps = 0

  def __call__(self, trajectory, force_is_boundary=None):
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
      force_is_boundary: Forces the indication of the trajectory being boundary.
        Useful if a flattened trajectory is provided.
    """
    for writer in self._writers:
      writer.append(trajectory)
    self._cached_steps += 1

    self._write_cached_steps()

    # Reset the client on boundary transitions.
    if not self._allow_multi_episode_sequences:
      if self._is_boundary(trajectory, force_is_boundary):
        self.close()
        self.open()

  def _write_cached_steps(self):
    """Writes the cached steps into the writer.

    **Note**: The method does *not* clear the cache after writing.
    """

    def write_item(table_name, writer, sequence_length, stride_length):
      if (self._cached_steps >= sequence_length and
          (self._cached_steps - sequence_length) % stride_length == 0):
        writer.create_item(
            table=table_name,
            num_timesteps=sequence_length,
            priority=self._priority)

    for table_name, writer, sequence_length, stride_length in zip(
        self._table_names, self._writers, self._sequence_lengths,
        self._stride_lengths):
      write_item(table_name, writer, sequence_length, stride_length)

  def _is_boundary(self, trajectory, force_is_boundary):
    if force_is_boundary is not None:
      # Assumed the trajectory is flat, so being boundary is supllied as a side
      # input.
      return force_is_boundary
    # Assumed trajectory is is a true `Trajectory` object, so use the
    # corresponding method to decide if this is boundary trajectory.
    return trajectory.is_boundary()

  def reset(self):
    """Resets the state of the observer.

    No data observed before the reset will be pushed to the RB.
    """
    self.close()
    self.open()

  def open(self):
    """Open the writer of the observer."""
    if self._writers is None:
      self._writers = [
          self._py_client.writer(max_sequence_length=s_len)
          for s_len in self._sequence_lengths
      ]
      self._cached_steps = 0

  def close(self):
    """Closes the writer of the observer.

    **Note**: Using the observer after closing it is not supported.
    """

    if self._writers is not None:
      for writer in self._writers:
        writer.close()
      self._writers = None
