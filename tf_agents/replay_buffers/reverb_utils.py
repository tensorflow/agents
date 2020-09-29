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

from typing import Text, Union, Sequence

from absl import logging

from tf_agents.typing import types
from tf_agents.utils import lazy_loader

# Lazy loading since not all users have the reverb package installed.
reverb = lazy_loader.LazyLoader("reverb", globals(), "reverb")


class ReverbAddEpisodeObserver(object):
  """Observer for writing episodes to the Reverb replay buffer."""

  def __init__(self,
               py_client: types.ReverbClient,
               table_name: Union[Text, Sequence[Text]],
               max_sequence_length: int,
               priority: Union[float, int] = 1,
               bypass_partial_episodes: bool = False):
    """Creates an instance of the ReverbAddEpisodeObserver.

    **Note**: This observer is designed to work with py_drivers only, and does
    not support batches.

    TODO(b/158865335): Optionally truncate long episodes and add to buffer.

    Args:
      py_client: Python client for the reverb replay server.
      table_name: The table name(s) where samples will be written to.
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
      ValueError: If `priority` is not numeric.
      ValueError: If max_sequence_length is not positive.
    """
    if max_sequence_length <= 0:
      raise ValueError(
          "`max_sequence_length` must be an integer greater equal one.")

    if isinstance(table_name, Text):
      self._table_names = [table_name]
    else:
      self._table_names = table_name

    self._max_sequence_length = max_sequence_length
    self._priority = priority

    self._py_client = py_client
    self._writer = py_client.writer(
        max_sequence_length=self._max_sequence_length)
    self._cached_steps = 0
    self._bypass_partial_episodes = bypass_partial_episodes
    self._overflow_episode = False

  def update_priority(self, priority: Union[float, int]):
    """Update the table priority.

    Args:
      priority: Updates the priority of the observer.

    ValueError: If priority is not numeric.
    """
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
      self.write_cached_steps()

  def write_cached_steps(self):
    """Writes the cached steps into the writer.

    **Note**: The method resets the number of episodes and steps after writing
      the data into the replay buffer.
    """
    if not self._overflow_episode:
      for table_name in self._table_names:
        self._writer.create_item(
            table=table_name,
            num_timesteps=self._cached_steps,
            priority=self._priority)
    self.reset()

  def reset(self):
    """Resets the state of the observer.

    Note that the data cached in the writer will NOT get automatically written
    into the Reverb table. If you wish to write the cached partial episode as
    a new sequences, call `write_cached_steps` instead.
    """
    self.close()
    self.open()
    self._overflow_episode = False

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
  """Stateful observer for writing to the Reverb replay."""

  def __init__(self,
               py_client: types.ReverbClient,
               table_name: Union[Text, Sequence[Text]],
               sequence_length: int,
               stride_length: int = 1,
               priority: Union[float, int] = 1):
    """Creates an instance of the ReverbAddTrajectoryObserver.

    If multiple table_names and sequence lengths are provided data will only be
    stored once but be available for sampling with multiple sequence lengths
    from the respective reverb tables.

    **Note**: This observer is designed to work with py_drivers only, and does
    not support batches.

    Args:
      py_client: Python client for the reverb replay server.
      table_name: The table name(s) where samples will be written to.
      sequence_length: The sequence_length used to write
        to the given table.
      stride_length: The integer stride for the sliding window for overlapping
        sequences.  The default value of `1` creates an item for every
        window.  Using `L = sequence_length` this means items are created for
        times `{0, 1, .., L-1}, {1, 2, .., L}, ...`.  In contrast,
        `stride_length = L` will create an item only for disjoint windows
        `{0, 1, ..., L-1}, {L, ..., 2 * L - 1}, ...`.
      priority: Initial priority for new samples in the RB.
    """
    if isinstance(table_name, Text):
      self._table_names = [table_name]
    else:
      self._table_names = table_name
    self._sequence_length = sequence_length
    self._stride_length = stride_length
    self._priority = priority

    self._py_client = py_client
    # TODO(b/153700282): Use a single writer with max_sequence_length=max(...)
    # once Reverb Dataset with emit_timesteps=True returns properly shaped
    # sequences.
    self._writer = py_client.writer(max_sequence_length=sequence_length)
    self._cached_steps = 0

  def __call__(self, trajectory):
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
    """
    self._writer.append(trajectory)
    self._cached_steps += 1

    self._write_cached_steps()

    # Reset the client on boundary transitions.
    if trajectory.is_boundary():
      self.reset()

  def _write_cached_steps(self):
    """Writes the cached steps into the writer.

    **Note**: The method does *not* clear the cache after writing.
    """

    def write_item(writer, sequence_length, stride_length):
      if (self._cached_steps >= sequence_length and
          (self._cached_steps - sequence_length) % stride_length == 0):
        for table_name in self._table_names:
          writer.create_item(
              table=table_name,
              num_timesteps=sequence_length,
              priority=self._priority)

    write_item(
        self._writer,
        self._sequence_length,
        self._stride_length)

  def reset(self):
    """Resets the state of the observer.

    No data observed before the reset will be pushed to the RB.
    """
    self.close()
    self.open()

  def open(self):
    """Open the writer of the observer."""
    if self._writer is None:
      self._writer = self._py_client.writer(
          max_sequence_length=self._sequence_length)
      self._cached_steps = 0

  def close(self):
    """Closes the writer of the observer.

    **Note**: Using the observer after closing it is not supported.
    """
    self._writer.close()
    self._writer = None


class ReverbTrajectorySequenceObserver(ReverbAddTrajectoryObserver):
  """Reverb trajectory sequence observer.

  This is equivalent to ReverbAddTrajectoryObserver but sequences are not cut
  when a boundary trajectory is seen. This allows for sequences to be sampled
  with boundaries anywhere in the sequence rather than just at the end.

  Consider using this observer when you want to create training experience that
  can encompass any subsequence of the observed trajectories.

  **Note**: Counting of steps in drivers does not include boundary steps. To
  guarantee only 1 item is pushed to the replay when collecting n steps with a
  `sequence_length` of n make sure to set the `stride_length`
  """

  def __call__(self, trajectory):
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
    """
    self._writer.append(trajectory)
    self._cached_steps += 1

    self._write_cached_steps()

