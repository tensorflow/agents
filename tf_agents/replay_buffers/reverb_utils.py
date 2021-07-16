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
"""Utilities for working with a reverb replay buffer."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Text, Union, Sequence

from absl import logging
import tensorflow as tf

from tf_agents.trajectories import trajectory as trajectory_lib

from tf_agents.typing import types
from tf_agents.utils import lazy_loader

# Lazy loading since not all users have the reverb package installed.
reverb = lazy_loader.LazyLoader("reverb", globals(), "reverb")


class ReverbAddEpisodeObserver(object):
  """Observer for writing episodes to Reverb.

  This observer should be called at every step. It does not support batched
  trajectories. The steps are cached and written at the end of the episode.

  At the end of each episode, an item is written to Reverb. Each item is the
  trajectory containing an episode, including a boundary step in the end.
  Therefore, the sequence lengths of the items may vary. If you want a fixed
  sequence length, use `ReverbAddTrajectoryObserver` instead.

  Unfinished episodes remain in the cache and do not get written until
  `reset(write_cached_steps=True)` is called.

  TODO(b/176261664): Note that this observer only supports batch_size=1 from the
  consumer, if your episodes have variable lengths.
  """

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
    self._writer_has_data = False

  def update_priority(self, priority: Union[float, int]) -> None:
    """Update the table priority.

    Args:
      priority: Updates the priority of the observer.

    ValueError: If priority is not numeric.
    """
    self._priority = priority

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Cache the single step trajectory to be written into Reverb.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.

    Raises:
      ValueError: If `bypass_partial_episodes` == False and episode length
        is > `max_sequence_length`.
    """
    # TODO(b/176494855): Raise an error if an invalid trajectory is passed in.
    # Currently, invalid `traj` value (mid->first, last->last) is not specially
    # handled and is treated as a normal mid->mid step.
    if (self._cached_steps >= self._max_sequence_length and
        not self._overflow_episode):
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

    # At the end of the overflowing episode, drop the cached incomplete episode
    # and reset the writer.
    if self._overflow_episode and trajectory.is_boundary():
      self.reset(write_cached_steps=False)
      return

    if not self._overflow_episode:
      self._writer.append(trajectory)
      self._writer_has_data = True
      self._cached_steps += 1

      # At the end of an episode, write the item to Reverb and clear the cache.
      if trajectory.is_boundary():
        self.reset(write_cached_steps=True)

  def _write_cached_steps(self) -> None:
    """Writes the cached steps into the writer.

    **Note**: The method does not clear the cache.
    """
    # Only writes to Reverb when the writer has cached trajectories.
    if self._writer_has_data:
      for table_name in self._table_names:
        self._writer.create_item(
            table=table_name,
            num_timesteps=self._cached_steps,
            priority=self._priority)
      self._writer_has_data = False
    else:
      logging.info("Skipped writing to Reverb because the writer is empty.")

  def flush(self):
    """Ensures that items are pushed to the service.

    Note: The items are not always immediately pushed. This method is often
    needed when `rate_limiter_timeout_ms` is set for the replay buffer.
    By calling this method before the `learner.run()`, we ensure that there is
    enough data to be consumed.
    """
    self._writer.flush()

  def reset(self, write_cached_steps: bool = True) -> None:
    """Resets the state of the observer.

    **Note**: Reset should be called only after all collection has finished
    in a standard workflow. No need to manually call reset between episodes.

    Args:
      write_cached_steps: By default, if there is remaining data in the cache,
        write them to Reverb before clearing the cache. If `write_cached_steps`
        is `False`, throw away the cached data instead.
    """
    if write_cached_steps:
      self._write_cached_steps()

    self._cached_steps = 0
    self._overflow_episode = False

    self.close()
    self.open()

  def open(self) -> None:
    """Open the writer of the observer. This is a no-op if it's already open."""
    if self._writer is None:
      self._writer = self._py_client.writer(
          max_sequence_length=self._max_sequence_length)

  def close(self) -> None:
    """Closes the writer of the observer.

    **Note**: Using the observer after it is closed (and not reopened) is not
      supported.
    """

    if self._writer is not None:
      # self._writer.end_episode()
      self._writer.flush()
      self._writer.close()
      self._writer = None
      self._writer_has_data = False


class ReverbAddTrajectoryObserver(object):
  """Stateful observer for writing fixed length trajectories to Reverb.

  This observer should be called at every environment step. It does not support
  batched trajectories.

  Steps are cached until `sequence_length` steps are gathered. At which point an
  item is created. From there on a new item is created every `stride_length`
  observer calls.

  If an episode terminates before enough steps are cached, the data is discarded
  unless `pad_end_of_episodes` is set.
  """

  def __init__(self,
               py_client: types.ReverbClient,
               table_name: Union[Text, Sequence[Text]],
               sequence_length: int,
               stride_length: int = 1,
               priority: Union[float, int] = 1,
               pad_end_of_episodes: bool = False,
               tile_end_of_episodes: bool = False):
    """Creates an instance of the ReverbAddTrajectoryObserver.

    If multiple table_names and sequence lengths are provided data will only be
    stored once but be available for sampling with multiple sequence lengths
    from the respective reverb tables.

    **Note**: This observer is designed to work with py_drivers only, and does
    not support batches.

    Args:
      py_client: Python client for the reverb replay server.
      table_name: The table name(s) where samples will be written to.
      sequence_length: The sequence_length used to write to the given table.
      stride_length: The integer stride for the sliding window for overlapping
        sequences.  The default value of `1` creates an item for every window.
        Using `L = sequence_length` this means items are created for times `{0,
        1, .., L-1}, {1, 2, .., L}, ...`.  In contrast, `stride_length = L` will
        create an item only for disjoint windows `{0, 1, ..., L-1}, {L, ..., 2 *
        L - 1}, ...`.
      priority: Initial priority for new samples in the RB.
      pad_end_of_episodes: At the end of an episode, the cache is dropped by
        default. When `pad_end_of_episodes = True`, the cache gets padded with
        boundary steps (last->first) with `0` values everywhere and padded items
        of `sequence_length` are written to Reverb.
      tile_end_of_episodes: If `pad_end_of_episodes` is True then, the last
        padded item starts with a boundary step from the episode.

        When this option is True the following items will be generated:

        F, M, L, P
        M, L, P, P
        L, P, P, P

        If False, only a single one will be generated:

        F, M, L, P

        For training recurrent models on environments where required information
        is only available at the start of the episode it is useful to set
        `tile_end_of_episodes=False` and the `sequence_length` to be the length
        of the longest episode.
    Raises:
      ValueError: If `tile_end_of_episodes` is set without
        `pad_end_of_episodes`.
    """
    if isinstance(table_name, Text):
      self._table_names = [table_name]
    else:
      self._table_names = table_name
    self._sequence_length = sequence_length
    self._stride_length = stride_length
    self._priority = priority
    self._pad_end_of_episodes = pad_end_of_episodes
    self._tile_end_of_episodes = tile_end_of_episodes

    if tile_end_of_episodes and not pad_end_of_episodes:
      raise ValueError("Must set `pad_end_of_episodes=True` when using "
                       "`tile_end_of_episodes`")

    self._py_client = py_client
    # TODO(b/153700282): Use a single writer with max_sequence_length=max(...)
    # once Reverb Dataset with emit_timesteps=True returns properly shaped
    # sequences.
    self._writer = py_client.writer(max_sequence_length=sequence_length)
    self._cached_steps = 0
    self._last_trajectory = None

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
    """
    self._last_trajectory = trajectory
    self._writer.append(trajectory)
    self._cached_steps += 1

    # If the fixed sequence length is reached, write the sequence.
    self._write_cached_steps()

    # If it happens to be the end of the episode, clear the cache. Pad first and
    # write the items into Reverb if required.
    if trajectory.is_boundary():
      if self._pad_end_of_episodes:
        self.reset(write_cached_steps=True)
      else:
        self.reset(write_cached_steps=False)

  def _sequence_lengths_reached(self) -> bool:
    """Whether the cache has sufficient steps to write a new item into Reverb."""
    return (self._cached_steps >= self._sequence_length) and (
        self._cached_steps - self._sequence_length) % self._stride_length == 0

  def _write_cached_steps(self) -> None:
    """Writes the cached steps iff there is enough data in the cache.

    **Note**: The method does *not* clear the cache after writing.
    """

    if self._sequence_lengths_reached():
      for table_name in self._table_names:
        self._writer.create_item(
            table=table_name,
            num_timesteps=self._sequence_length,
            priority=self._priority)

    return None

  def flush(self):
    """Ensures that items are pushed to the service.

    Note: The items are not always immediately pushed. This method is often
    needed when `rate_limiter_timeout_ms` is set for the replay buffer.
    By calling this method before the `learner.run()`, we ensure that there is
    enough data to be consumed.
    """
    self._writer.flush()

  def _get_padding_step(
      self, example_trajectory: trajectory_lib.Trajectory
  ) -> trajectory_lib.Trajectory:
    """Get the padding step to append to the cache."""
    zero_step = trajectory_lib.boundary(
        tf.nest.map_structure(tf.zeros_like, example_trajectory.observation),
        tf.nest.map_structure(tf.zeros_like, example_trajectory.action),
        tf.nest.map_structure(tf.zeros_like, example_trajectory.policy_info),
        tf.nest.map_structure(tf.zeros_like, example_trajectory.reward),
        tf.nest.map_structure(tf.zeros_like, example_trajectory.discount),
    )
    return zero_step

  def reset(self, write_cached_steps: bool = True) -> None:
    """Resets the state of the observer.

    **Note**: Reset should be called only after all collection has finished
    in a standard workflow. No need to manually call reset between episodes.

    Args:
      write_cached_steps: boolean flag indicating whether we want to write the
        cached trajectory. When this argument is True, the function attempts to
        write the cached data before resetting (optionally with padding).
        Otherwise, the cached data gets dropped.
    """
    # Write the cached steps if requested and if the cache is not empty.
    if write_cached_steps and self._last_trajectory is not None:
      # Pad the cache and write all the cached steps into Reverb when padding is
      # enabled and `write_cached_steps` is set to `True`.
      if self._pad_end_of_episodes:
        zero_step = self._get_padding_step(self._last_trajectory)

        if self._tile_end_of_episodes:
          pad_range = range(self._sequence_length - 1)
        else:
          pad_range = range(self._sequence_length - self._cached_steps)

        for _ in pad_range:
          self._writer.append(zero_step)
          self._cached_steps += 1
          self._write_cached_steps()
      # Write the cached trajectories without padding, if the cache contains
      # enough steps to write a full item.
      elif self._sequence_lengths_reached():
        self._write_cached_steps()
      else:
        raise ValueError(
            "write_cached_steps is True, but not enough steps remain in the "
            "cache to write an item with sequence_length={}, consider enabling "
            "pad_end_of_episodes.".format(self._sequence_length))

    self._cached_steps = 0
    self._last_trajectory = None

    self.close()
    self.open()

  def open(self) -> None:
    """Open the writer of the observer."""
    if self._writer is None:
      self._writer = self._py_client.writer(
          max_sequence_length=self._sequence_length)
      self._cached_steps = 0

  def close(self) -> None:
    """Closes the writer of the observer.

    **Note**: Using the observer after it is closed (and not reopened) is not
      supported.
    """
    if self._writer is not None:
      # self._writer.end_episode()
      self._writer.flush()
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

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Writes the trajectory into the underlying replay buffer.

    Allows trajectory to be a flattened trajectory. No batch dimension allowed.

    Args:
      trajectory: The trajectory to be written which could be (possibly nested)
        trajectory object or a flattened version of a trajectory. It assumes
        there is *no* batch dimension.
    """
    # We record the last step called to generate the padding step when padding
    # is enabled.
    self._last_trajectory = trajectory
    self._writer.append(trajectory)
    self._cached_steps += 1

    self._write_cached_steps()
