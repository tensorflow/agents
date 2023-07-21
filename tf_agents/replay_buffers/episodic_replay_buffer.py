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

"""An episodic replay buffer of nests of Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.replay_buffers import episodic_table
from tf_agents.replay_buffers import replay_buffer as replay_buffer_base
from tf_agents.replay_buffers import table
from tf_agents.utils import common

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.data.util import nest as data_nest  # TF internal
from tensorflow.python.layers import utils  # TF internal
from tensorflow.python.ops import list_ops  # TF internal
from tensorflow.python.ops.distributions import util as distributions_util  # TF internal
# pylint:enable=g-direct-tensorflow-import


# The ID value for episode id holders that do not point to an actual episode.
_INVALID_EPISODE_ID = -1


Episodes = collections.namedtuple('Episodes',
                                  ['length', 'completed', 'tensor_lists'])

BufferInfo = collections.namedtuple('BufferInfo', ['ids'])


@gin.configurable
class EpisodicReplayBuffer(replay_buffer_base.ReplayBuffer):
  """An episodic ReplayBuffer with Uniform sampling."""

  # Internal details of the EpisodicReplayBuffer:
  #
  #  The EpisodicReplayBuffer is composed of the following objects:
  #     _data_table: EpisodicTable structure data_spec, capacity _capacity.
  #     _id_table: Table containing single int64 id, capacity _capacity.
  #     _episodes_loc_to_id_map: counter variable, capacity _capacity.
  #     _episode_lengths: counter variable, capacity _capacity.
  #     _episode_completed: uint8(bool) variable, capacity _capacity.
  #     _last_episode: scalar int variable.
  #     _add_episode_critical_section: CriticalSection for adding new episodes.
  #
  #  Users call create_episode_ids() to get a scalar or vector int64 Tensor
  #  of size num_episodes (or scalar, if num_episodes is None).
  #
  #  All operations map values from an input Tensor "episode_id[s]" to
  #  an episode_location.  The calculation is:
  #     episode_location = episode_id % capacity.
  #
  #  New episodes are emitted by incrementing _last_episode inside the critical
  #  section, and returning the new value.  The values of new episode ids
  #  are logical values: they always go up.
  #
  #  To get the current logical id for a given concrete location in the replay
  #  buffer, look it up via _episodes_loc_to_id_map[episode_id].
  #

  def __init__(self,
               data_spec,
               capacity=1000,
               completed_only=False,
               buffer_size=8,
               name_prefix='EpisodicReplayBuffer',
               device='cpu:*',
               seed=None,
               begin_episode_fn=None,
               end_episode_fn=None,
               dataset_drop_remainder=False,
               dataset_window_shift=None):
    """Creates an EpisodicReplayBuffer.

    This class receives a dataspec and capacity and creates a replay buffer
    supporting read/write operations, organized into episodes.
    This uses an underlying EpisodicTable with capacity equal to
    capacity.  Each row in the table can have an episode of unbounded
    length.

    Args:
      data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing
        a single item that can be stored in this buffer.
      capacity: An integer, the maximum number of episodes.
      completed_only: Scalar bool.  Whether to sample full episodes
        (in as_dataset if `num_steps = None`), and whether to sample subsets
        from full episodes (in as_dataset `num_steps != None`).
      buffer_size: How many episode IDs to precalculate in a buffer when using
        `as_dataset(..., single_deterministic_pass=False)`.

        This parameter controls how often episode IDs are sampled
        according to their lengths in the `tf.data.Dataset` returned by
        `as_dataset`.  Choosing a small number means episodes are sampled
        more frequently, which is expensive, but new data in the replay buffer
        is seen more quickly by the resulting dataset.  Choosing a larger number
        means the sampling is less frequent and less costly, but new episodes
        and updated episode lengths may not be respected for up to
        `buffer_size` requests to the `Dataset` iterator.

        For example, if `buffer_size > 1` then a new episode may be added to
        the replay buffer, but this episode won't be "seen" by the
        `Dataset` for up to `buffer_size - 1` more accesses.
      name_prefix: A prefix for variable and op names created by this class.
      device: A TensorFlow device to place the Variables and ops.
      seed: optional random seed for sampling operations.
      begin_episode_fn: A function that maps batched tensors respecting
        `data_spec` to a boolean scalar or vector Tensor, indicating whether the
        given entries are the start of a new episode.

        Used by `add_batch`, `add_sequence`, and `extend_episodes` to indicate
        whether the `episode_id` should be incremented.

        Default value:
        ```python
        begin_episode_fn = lambda traj: traj.is_first()
        ```
      end_episode_fn: A function that maps batched tensors respecting
        `data_spec` to a boolean scalar or vector Tensor, indicating whether the
        given entries are the end of an episode.

        **NOTE** The current default behavior is to mark an episode as
        completed once it receives its final reward.  However,
        additional frames may be added to the episode after this.
        Typically, exactly one boundary frame (LAST -> FIRST), is likely
        to be added to the episode after it is marked as completed).

        Used by `add_batch`, `add_sequence`, and `extend_episodes` to
        indicate whether to end the episode at `episode_id`.

        Default value:
        ```python
        begin_episode_fn = lambda traj: traj.is_last()
        ```
      dataset_drop_remainder: If `True`, then when calling `as_dataset` with
        arguments `sample_batch_size is not None`, the final batch will be
        dropped if it does not contain exactly `sample_batch_size` items.  This
        is helpful for static shape inference as the resulting tensors will
        always have leading dimension `sample_batch_size` instead of `None`.
      dataset_window_shift: Window shift used when calling
        `as_dataset` with arguments `single_deterministic_pass=True` and
        `num_steps is not None`.  This determines how the resulting frames are
        windowed.  If `None`, then there is no overlap created between frames
        and each frame is seen exactly once.  For example, if `max_length=5`,
        `num_steps=2`, `sample_batch_size=None`, and
        `dataset_window_shift=None`, then the datasets returned will have
        frames `{[0, 1], [2, 3], [4]}`.

        If `num_steps is not None`, then windows are created
        with a window overlap of `dataset_window_shift` and you will see each
        frame up to `num_steps` times.  For example, if `max_length=5`,
        `num_steps=2`, `sample_batch_size=None`, and `dataset_window_shift=1`,
        then the datasets returned will have windows of shifted repeated frames:
        `{[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]}`.

        For more details, see the documentation of `tf.data.Dataset.window`,
        specifically for the `shift` argument.

        The default behavior is to not overlap frames
        (`dataset_window_shift=None`) but users often want to see all
        combinations of frame sequences, in which case `dataset_window_shift=1`
        is the appropriate value.
    """
    super(EpisodicReplayBuffer, self).__init__(
        data_spec,
        capacity,
        stateful_dataset=True)
    # Create tables `table_fn(data_spec, capacity)` that
    # can read/write nested tensors.
    table_fn = table.Table

    # Create tables `episode_table_fn(data_spec, capacity, name_prefix)`
    # that can read/write nested variable-length episodes
    # (in practice this means using TensorLists).
    episode_table_fn = episodic_table.EpisodicTable

    if begin_episode_fn is None:
      def _begin_episode_fn(t):
        is_first = getattr(t, 'is_first', None)
        if not callable(is_first):
          raise TypeError(
              'Argument t is not a Trajectory; did you forget to pass the '
              'proper begin_episode_fn to EpisodicReplayBuffer?  Saw: \'{}\''
              .format(t))
        return is_first()
      begin_episode_fn = _begin_episode_fn
    if end_episode_fn is None:
      def _end_episode_fn(t):
        is_last = getattr(t, 'is_last', None)
        if not callable(is_last):
          raise TypeError(
              'Argument t is not a Trajectory; did you forget to pass the '
              'proper end_episode_fn to EpisodicReplayBuffer?  Saw: \'{}\''
              .format(t))
        return is_last()
      end_episode_fn = _end_episode_fn
    if not callable(begin_episode_fn):
      raise TypeError(
          'begin_episode_fn is not callable: {}'.format(begin_episode_fn))
    if not callable(end_episode_fn):
      raise TypeError(
          'end_episode_fn is not callable: {}'.format(end_episode_fn))
    self._begin_episode_fn = begin_episode_fn
    self._end_episode_fn = end_episode_fn
    self._id_spec = specs.TensorSpec([], dtype=tf.int64, name='id')
    self._name_prefix = name_prefix
    self._device = device
    self._seed = seed
    self._completed_only = completed_only
    self._buffer_size = buffer_size
    self._dataset_window_shift = dataset_window_shift
    self._dataset_drop_remainder = dataset_drop_remainder
    self._num_writes = common.create_variable('num_writes_counter')

    with tf.device(self._device):
      self._data_table = episode_table_fn(
          self._data_spec, self._capacity, self._name_prefix)
      # The episode ids
      self._id_table = table_fn(self._id_spec, self._capacity)
      self._episodes_loc_to_id_map = common.create_variable(
          'episodes_loc_to_id_map', shape=[self._capacity],
          initial_value=_INVALID_EPISODE_ID)
      self._episode_lengths = common.create_variable(
          'episode_lengths', shape=[self._capacity], initial_value=0)
      # Marks episodes as completed or not.
      # TODO(b/80430723) Add a way for users to mark episodes completed.
      # TODO(b/76154485) Change to tf.bool.
      self._episode_completed = common.create_variable(
          'episode_completed',
          shape=[self._capacity],
          dtype=tf.uint8,
          initial_value=0)
      # The last episode id so far in the table.
      self._last_episode = common.create_variable(
          'last_episode', initial_value=_INVALID_EPISODE_ID)
      self._add_episode_critical_section = tf.CriticalSection(
          name='add_episode')

  @property
  def num_writes(self):
    return self._num_writes

  @property
  def name_prefix(self):
    return self._name_prefix

  def _num_frames(self):
    """Returns the number of frames in the buffer."""
    return tf.math.reduce_sum(self._episode_lengths)

  def create_episode_ids(self, num_episodes=None):
    """Returns a new tensor containing initial invalid episode ID(s).

    This tensor is meant to be passed to methods like `add_batch` and
    `extend_episodes`; those methods will return an updated set of episode id
    values in their output.  To keep track of updated episode IDs across
    multiple TF1 session run calls, the `episode_ids` may be read out and passed
    back in by the user, or stored in a `tf.Variable`.  A helper class which
    does this for you is available in this module, it is called
    `StatefulEpisodicReplayBuffer`.

    A simple non-`Variable` way to do this (in TF1) is:

    ```python
    data = collect_data_tf_op()
    episode_ids = tf.placeholder_with_default(rb.create_episode_ids(3), [3])
    new_episode_ids = rb.add_batch(data, episode_ids)

    ids = session.run(episode_ids)
    while True:
      ...
      ids = session.run(new_episode_ids, feed_dict=dict(episode_ids=ids))
    ```

    The initial value of these ids is subject to change, but currently set
    to `-1`.  When methods like `add_batch` see entries like this, they
    reserve a new (valid) id for this entry in the buffer and return the
    associated id in this location.

    Args:
      num_episodes: (Optional) int32, number of episode IDs to create.
        This may be a tensor.  If `None`, a scalar ID tensor is returned.

    Returns:
      An int64 Tensor containing initial episode(s) ID(s).

    Raises:
      ValueError: If `num_episodes` is bigger than capacity, or non-scalar.
    """
    if tf.is_tensor(num_episodes):
      if num_episodes.shape.rank != 0:
        raise ValueError('num_episodes must be a scalar, but saw shape: {}'
                         .format(num_episodes.shape))
      return tf.fill(
          [num_episodes],
          tf.convert_to_tensor(_INVALID_EPISODE_ID, dtype=tf.int64),
          name='episode_id')

    shape = ()
    if num_episodes is not None and num_episodes > 0:
      if num_episodes > self._capacity:
        raise ValueError('Buffer cannot create episode_ids when '
                         'num_episodes {} > capacity {}.'.format(
                             num_episodes, self._capacity))
      shape = (num_episodes,)
    return tf.constant(
        _INVALID_EPISODE_ID, shape=shape, dtype=tf.int64, name='episode_id')

  def add_sequence(self, items, episode_id):
    """Adds a sequence of items to the replay buffer for the selected episode.

    Args:
      items: A sequence of items to be added to the buffer. Items will have the
        same structure as the data_spec of this class, but the tensors in items
        will have an outer sequence dimension in addition to the corresponding
        spec in data_spec.
      episode_id: A Tensor containing the current episode_id.

    Returns:
      An updated episode id Tensor.  Accessing this episode id value will,
      as a side effect, start or end the current episode in the buffer.
    """
    episode_id.shape.assert_has_rank(0)
    with tf.device(self._device):
      with tf.name_scope('add_steps'):
        # If users pass in, e.g., a python list [2, 3, 4] of type int32
        # but the data_spec requires an int64, then the user will get a very
        # confusing error much deeper in the TensorList code.  Doing the
        # conversion here either converts when necessary, or raises an error
        # on incompatible types earlier in the run.
        items = tf.nest.map_structure(
            lambda x, spec: tf.convert_to_tensor(value=x, dtype=spec.dtype),
            items, self._data_spec)
        item_0 = tf.nest.flatten(items)[0]
        num_steps = tf.cast(
            tf.compat.dimension_value(item_0.shape[0]) or
            tf.shape(input=item_0)[0], tf.int64)
        # If begin_episode is True, then the increment of the episode_id happens
        # before trying to add anything to the buffer, regardless of whether the
        # item will actually be added.
        begin_episode = self._begin_episode_fn(items)
        end_episode = self._end_episode_fn(items)
        new_episode_id = self._get_episode_id(
            episode_id, begin_episode, end_episode)
        episode_location = self._get_episode_id_location(new_episode_id)

        def _add_steps():
          """Add sequence of items to the buffer."""
          inc_episode_length = self._increment_episode_length_locked(
              episode_location, num_steps)
          write_data_op = self._data_table.append(episode_location, items)
          with tf.control_dependencies([inc_episode_length, write_data_op]):
            return tf.identity(new_episode_id)

        # Accessing episode_id may modify
        # self._episodes_loc_to_id_map, so ensure it is executed
        # before the tf.equal.
        with tf.control_dependencies([new_episode_id]):
          episode_valid = tf.equal(
              self._episodes_loc_to_id_map[episode_location], new_episode_id)
        def _maybe_add_steps():
          return self._add_episode_critical_section.execute(_add_steps)
        return utils.smart_cond(
            episode_valid,
            _maybe_add_steps,
            lambda: tf.identity(new_episode_id),
            name='conditioned_add_steps')

  def add_batch(self, items, episode_ids):
    """Adds a batch of single steps for the corresponding episodes IDs.

    Args:
      items: A batch of items to be added to the buffer. Items will have the
        same structure as the data_spec of this class, but the tensors in items
        will have an extra outer dimension `(num_episodes, ...)` in addition to
        the corresponding spec in data_spec.
      episode_ids: A int64 vector `Tensor` containing the ids of the
        episodes the items are being added to. Shaped `(num_episodes,)`.

    Returns:
      A `Tensor` containing the updated episode ids.  Accessing or executing
      this tensor also adds `items` to the replay buffer.
    """
    episode_ids.shape.assert_has_rank(1)
    with tf.device(self._device):
      with tf.name_scope('add_batch'):
        # If begin_episode is True, then the increment of the episode_id happens
        # before trying to add anything to the buffer, regardless of whether the
        # item will actually be added.

        begin_episode = self._begin_episode_fn(items)
        end_episode = self._end_episode_fn(items)
        batch_episode_ids = self._get_batch_episode_ids(episode_ids,
                                                        begin_episode,
                                                        end_episode)
        episodes_locations = tf.math.mod(batch_episode_ids, self._capacity)
        # Accessing episode_id may modify self._episodes_loc_to_id_map, so
        # ensure it is executed before
        with tf.control_dependencies([episodes_locations]):
          episode_valid = tf.equal(
              self._episodes_loc_to_id_map.sparse_read(episodes_locations),
              batch_episode_ids)

        def _add_batch():
          """Add elements to the appropiate episode_locations."""
          ids_to_update = tf.reshape(tf.compat.v1.where(episode_valid), [-1])
          episodes_locations_ = tf.gather(episodes_locations, ids_to_update)
          filter_items = lambda item: tf.gather(item, ids_to_update)
          items_ = tf.nest.map_structure(filter_items, items)
          write_data_op = self._data_table.add(episodes_locations_, items_)
          inc_episode_lengths = self._increment_episode_length_locked(
              episodes_locations_)
          inc_write_counter_op = self._num_writes.assign_add(1)
          with tf.control_dependencies([
              write_data_op, inc_episode_lengths, inc_write_counter_op]):
            return tf.identity(batch_episode_ids)

        num_adds = tf.reduce_sum(input_tensor=tf.cast(episode_valid, tf.int64))

        def _maybe_add_batch():
          return self._add_episode_critical_section.execute(_add_batch)

        return tf.cond(
            pred=num_adds > 0,
            true_fn=_maybe_add_batch,
            false_fn=lambda: episode_ids)

  def gather_all(self):
    """Returns all the items in buffer.

    Returns:
      Returns all the items currently in the buffer. Returns a tensor
      of shape [1, SUM(T_i), ...]. Since episodes can be of different lengths,
      all steps of all episodes are grouped into one batch. Thus the first
      dimension is batch size = 1, the second dimension is of size equal to
      the sum of all timesteps for all episodes (SUM(T_i) for i in episode_ids).
      The remaining dimensions are the shape of the spec of items in the buffer.
    """
    items, _ = self._gather_all()
    return items

  # Defining abstract methods from replay_buffers.ReplayBuffer

  def _add_batch(self, items):
    raise NotImplementedError("""add_batch(items) is not implemented in
      EpisodicReplayBuffer. Use add_batch(items, episode_ids) instead""")

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=None):
    """Returns an episode sampled uniformly from the buffer.

    Args:
      sample_batch_size: Not used
      num_steps: Not used
      time_stacked: Not used

    Returns:
      A 2-tuple containing:

        - An episode sampled uniformly from the buffer.
        - BufferInfo NamedTuple, containing the episode id.
    """
    with tf.device(self._device):
      with tf.name_scope('get_next'):
        episode_id = self._sample_episode_ids(shape=[], seed=self._seed)
        row = self._get_episode_id_location(episode_id)
        data = self._data_table.get_episode_values(row)
        id_ = self._id_table.read(row)
    return data, BufferInfo(ids=id_)

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  sequence_preprocess_fn=None,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """Creates a dataset that returns episodes entries from the buffer.

    The dataset behaves differently depending on if `num_steps` is provided or
    not.  If `num_steps = None`, then entire episodes are sampled uniformly at
    random from the buffer.  If `num_steps != None`, then we attempt to sample
    uniformly across frames of all the episodes, and return subsets of length
    `num_steps`.  The algorithm for this is roughly:

    1. Sample an episode with a probability proportional to its length.
    2. If the length of the episode is less than `num_steps`, drop it.
    3. Sample a starting location `start` in `[0, len(episode) - num_steps]`
    4. Take a slice `[start, start + num_steps]`.

    The larger `num_steps` is, the higher the likelihood of edge effects (e.g.,
    certain frames not being visited often because they are near the start
    or end of an episode).  In the worst case, if `num_steps` is greater than
    most episode lengths, those episodes will never be visited.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.) Scalar int.  How many contiguous frames to get
        per entry. Default is `None`: return full-length episodes.
      sequence_preprocess_fn: (Optional.) Preprocessing function for sequences
        before they are sharded into subsequences of length `num_steps` and
        batched.
      num_parallel_calls: Number of parallel calls to use in the
        dataset pipeline when extracting episodes.  Default is to have
        tensorflow determine the optimal number of calls.

    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:

        - An item or sequence of items sampled uniformly from the buffer.
        - BufferInfo NamedTuple, containing the episode id.

    Raises:
      ValueError: If the data spec contains lists that must be converted to
        tuples.
      NotImplementedError: If `sequence_preprocess_fn != None` is passed in.
    """
    if sequence_preprocess_fn is not None:
      raise NotImplementedError('sequence_preprocess_fn is not supported.')

    # data_tf.nest.flatten does not flatten python lists, tf.nest.flatten does.
    if tf.nest.flatten(self._data_spec) != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  {}'.format(
              tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

    seed_per_episode = distributions_util.gen_new_seed(
        self._seed,
        salt='per_episode')

    episode_id_buffer_size = self._buffer_size * (sample_batch_size or 1)

    def _get_episode_locations(_):
      """Sample episode ids according to value of num_steps."""
      if num_steps is None:
        # Just want to get a uniform sampling of episodes.
        episode_ids = self._sample_episode_ids(
            shape=[episode_id_buffer_size], seed=self._seed)
      else:
        # Want to try to sample uniformly from frames, which means
        # sampling episodes by length.
        episode_ids = self._sample_episode_ids(
            shape=[episode_id_buffer_size],
            weigh_by_episode_length=True,
            seed=self._seed)
      episode_locations = self._get_episode_id_location(episode_ids)

      if self._completed_only:
        return tf.boolean_mask(
            tensor=episode_locations,
            mask=self._episode_completed.sparse_read(episode_locations))
      else:
        return episode_locations

    ds = tf.data.experimental.Counter().map(_get_episode_locations).unbatch()

    if num_steps is None:
      @tf.autograph.experimental.do_not_convert
      def _read_data_and_id(row):
        return (
            self._data_table.get_episode_values(row),
            self._id_table.read(row))
      ds = ds.map(_read_data_and_id, num_parallel_calls=num_parallel_calls)
    else:
      @tf.autograph.experimental.do_not_convert
      def _read_tensor_list_and_id(row):
        """Read the TensorLists out of the table row, get id and num_frames."""
        # Return a flattened tensor list
        flat_tensor_lists = tuple(
            tf.nest.flatten(self._data_table.get_episode_lists(row)))
        # Due to race conditions, not all entries may have been written for the
        # given episode.  Use the minimum list length to identify the full valid
        # available length.
        num_frames = tf.reduce_min(
            [list_ops.tensor_list_length(l) for l in flat_tensor_lists])
        return flat_tensor_lists, self._id_table.read(row), num_frames

      ds = ds.map(
          _read_tensor_list_and_id, num_parallel_calls=num_parallel_calls)

      def _filter_by_length(unused_1, unused_2, num_frames):
        # Remove episodes that are too short.
        return num_frames >= num_steps

      ds = ds.filter(_filter_by_length)

      @tf.autograph.experimental.do_not_convert
      def _random_slice(flat_tensor_lists, id_, num_frames):
        """Take a random slice from the episode, of length num_steps."""
        # Sample uniformly between [0, num_frames - num_steps]
        start_slice = tf.random.uniform((),
                                        minval=0,
                                        maxval=num_frames - num_steps + 1,
                                        dtype=tf.int32,
                                        seed=seed_per_episode)
        end_slice = start_slice + num_steps

        flat_spec = tf.nest.flatten(self._data_spec)

        # Pull out frames in [start_slice, start_slice + num_steps]
        flat = tuple(
            list_ops.tensor_list_gather(  # pylint: disable=g-complex-comprehension
                t, indices=tf.range(start_slice, end_slice),
                element_dtype=spec.dtype, element_shape=spec.shape)
            for t, spec in zip(flat_tensor_lists, flat_spec))
        return flat, id_

      ds = ds.map(_random_slice, num_parallel_calls=num_parallel_calls)

      def set_shape_and_restore_structure(flat_data, id_):
        def restore_shape(t_sliced):
          if t_sliced.shape.rank is not None:
            t_sliced.set_shape([num_steps] + [None] * (t_sliced.shape.rank - 1))
            return t_sliced
        shaped_flat = [restore_shape(x) for x in flat_data]
        return tf.nest.pack_sequence_as(self._data_spec, shaped_flat), id_

      ds = ds.map(set_shape_and_restore_structure)

    if sample_batch_size:
      if num_steps is None:
        raise ValueError("""`num_steps` must be set if `sample_batch_size` is
                         set in EpisodicReplayBuffer as_dataset.""")
      # We set drop_remainder on this batch since the dataset never ends,
      # therefore setting this will not cause any lost data and allows the
      # output tensors to have a definite leading dimension of
      # `sample_batch_size`.
      ds = ds.batch(sample_batch_size, drop_remainder=True)

    return ds

  def _single_deterministic_pass_dataset(
      self,
      sample_batch_size=None,
      num_steps=None,
      sequence_preprocess_fn=None,
      num_parallel_calls=tf.data.experimental.AUTOTUNE):
    """Creates a dataset that returns entries from the buffer in fixed order.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
        **NOTE** This argument may only be provided when
        `num_steps is not None`.  Otherwise the episodes may be different
        lengths and cannot be batched.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.  Required if
        `sample_batch_size` is provided.
      sequence_preprocess_fn: (Optional.) Preprocessing function for sequences
        before they are sharded into subsequences of length `num_steps` and
        batched.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.  Note, that the parallelism here is
        not "sloppy", in that setting this value does not affect the order
        in which frames are returned.

    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:

        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).

    Raises:
      ValueError: If `sample_batch_size is not None` but `num_steps is None`.
        When `num_steps is None`, the episodes returned may have different
        lengths, and there is no unique way to batch them.
      NotImplementedError: If `sequence_preprocess_fn != None` is passed in.
    """
    if sequence_preprocess_fn is not None:
      raise NotImplementedError('sequence_preprocess_fn is not supported.')
    if sample_batch_size is not None and num_steps is None:
      raise ValueError(
          'When requesting a batched dataset from EpisodicReplayBuffer, '
          'num_steps must be provided (but saw num_steps=None).')

    drop_remainder = self._dataset_drop_remainder
    window_shift = self._dataset_window_shift

    def get_episode_ids(_):
      min_frame_offset, max_frame_offset = _valid_range_ids(
          self._get_last_episode_id(), self._capacity)
      return tf.data.Dataset.range(min_frame_offset, max_frame_offset)

    # Instead of calling get_episode_ids and creating a dataset from this,
    # we instead build a dataset whose iterator recalculates the available
    # episode_ids in the dataset whenever it is reinitialized.  We
    # want to do this because the RB valid episodes can change over time;
    # specifically the RB may be empty when this dataset is first created.
    episode_ids_ds = tf.data.Dataset.range(1).flat_map(get_episode_ids)

    def read_episode(episode_id):
      row = self._get_episode_id_location(episode_id)
      return self._data_table.get_episode_values(row)

    ds = (episode_ids_ds
          .map(read_episode, num_parallel_calls=num_parallel_calls))
    if sample_batch_size is None:
      if num_steps is not None:
        # Disable autograph to make debugging errors easier.
        @tf.autograph.experimental.do_not_convert
        def group_windows(windowed):
          return tf.data.Dataset.zip(
              tf.nest.map_structure(
                  lambda d: d.batch(num_steps, drop_remainder=drop_remainder),
                  windowed))
        ds = (ds.unbatch()
              .window(num_steps, shift=window_shift)
              .flat_map(group_windows))
    else:
      # sample_batch_size is not None, which also implies num_steps is not None
      # per the check at the top of this function.
      assert num_steps is not None

      # Split up the replay buffer into sample_batch_size parallel datasets.
      ds_shards = (tf.data.Dataset.range(sample_batch_size)
                   .map(lambda i: ds.shard(sample_batch_size, i)))
      # In each dataset, convert different-length episodes to blocks of size
      # num_steps.  The very final blocks may be dropped if their size is not a
      # multiple of num_steps.
      # Disable autograph to make debugging errors easier.
      @tf.autograph.experimental.do_not_convert
      def rebatch(ds_):
        def batch_nest(window):
          return tf.data.Dataset.zip(
              tf.nest.map_structure(
                  lambda d: d.batch(num_steps, drop_remainder=True),
                  window))
        return (ds_
                .unbatch()
                .window(num_steps, shift=window_shift)
                .flat_map(batch_nest))
      ds_shards = ds_shards.map(rebatch)
      ds = ds_shards.interleave(lambda ds_: ds_)
      # Batch by sample_batch_size from the interleaved stream.
      ds = ds.batch(sample_batch_size, drop_remainder=drop_remainder)

    return ds

  def _gather_all(self):
    """Returns all the items currently in the buffer.

    Returns:
      A tuple containing two entries:
        - All the items currently in the buffer (nested).
        - The items ids.

    Raises:
      ValueError: If the data spec contains lists that must be converted to
        tuples.
    """
    if tf.nest.flatten(self._data_spec) != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  %s' %
          tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec))

    min_val, max_val = _valid_range_ids(self._get_last_episode_id(),
                                        self._capacity)

    def get_episode_and_id(id_):
      row = self._get_episode_id_location(id_)
      data = self._data_table.get_episode_values(row)
      n = tf.shape(tf.nest.flatten(data)[0])[0]
      id_repeated = tf.fill([n], id_)
      return (tuple(tf.nest.flatten(data)), id_repeated)

    episode_lengths = self._episode_lengths.read_value()
    if self._completed_only:
      episode_lengths *= tf.cast(self._episode_completed, dtype=tf.int64)
    total_length = tf.reduce_sum(input_tensor=episode_lengths)

    def via_iterator():
      """If total_length > 0, create a dataset iterator to concat episodes."""
      valid_episodes = tf.range(min_val, max_val)
      ds = tf.data.Dataset.from_tensor_slices(valid_episodes)

      if self._completed_only:
        # Filter out incomplete episodes.
        def check_completed(id_):
          return tf.cast(self._episode_completed.sparse_read(id_), tf.bool)
        ds = ds.filter(check_completed)

      def _unflatten(flat_data, id_):
        return tf.nest.pack_sequence_as(self._data_spec, flat_data), id_

      ds = (
          ds.map(get_episode_and_id).unbatch()
          # Batch all the frames in the buffer.  Request a larger amount in
          # case the buffer grows between the construction of total_length and
          # the call to .map().
          .batch(10 + 2 * total_length).map(_unflatten)).batch(1)

      # Use ds.take(1) in case we haven't requested a large enough batch (the
      # replay buffer has grown too quickly), since get_single_element requires
      # that the dataset only has a single entry; and we don't consider this
      # case to be an error.
      return tf.data.experimental.get_single_element(ds.take(1))

    def empty():

      def _empty_from_spec(spec):
        return tf.zeros([0] + spec.shape.as_list(), spec.dtype, name='empty')

      empty_data = tf.nest.map_structure(_empty_from_spec, self._data_spec)
      empty_id = tf.zeros([], dtype=tf.int64)
      return empty_data, empty_id

    return tf.cond(pred=total_length > 0, true_fn=via_iterator, false_fn=empty)

  def _clear(self, clear_all_variables=False):
    """Clears the replay buffer.

    Args:
      clear_all_variables: Boolean to indicate whether to clear all variables or
      just the data table and episode lengths (i.e. keep the current episode ids
      that are in flight in the buffer).
    Returns:
      An op to clear the buffer.
    """
    assignments = [
        self._episode_lengths.assign(tf.zeros_like(self._episode_lengths))]
    assignments += [self._num_writes.assign(tf.zeros_like(self._num_writes))]

    if clear_all_variables:
      zero_vars = self._id_table.variables() + [self._episode_completed]
      assignments += [var.assign(tf.zeros_like(var)) for var in zero_vars]
      neg_one_vars = [self._episodes_loc_to_id_map, self._last_episode]
      assignments += [var.assign(_INVALID_EPISODE_ID * tf.ones_like(var))
                      for var in neg_one_vars]

    return tf.group(self._data_table.clear(), assignments, name='clear')

  # Other private methods.

  def _completed_episodes(self):
    """Get a list of completed episode ids in the replay buffer.

    Returns:
      An int64 vector of length at most `capacity`.
    """
    def _completed_episodes():
      completed_mask = tf.equal(self._episode_completed, 1)
      return tf.boolean_mask(
          tensor=self._episodes_loc_to_id_map, mask=completed_mask)
    return self._add_episode_critical_section.execute(_completed_episodes)

  def _get_episode(self, episode_id):
    """Gets the current steps of the episode_id.

    Args:
      episode_id: A Tensor with the episode_id.

    Returns:
      A nested tuple/list of Tensors with all the items/steps of the episode.
      Each Tensor has shape `(episode_length,) + TensorSpec.shape`.

    Raises:
      InvalidArgumentException: (at runtime) if episode_id is not valid.
    """
    with tf.device(self._device), tf.name_scope('get_episode'):
      episode_id = tf.convert_to_tensor(
          episode_id, dtype=tf.int64, name='episode_id')
      episode_location = self._get_episode_id_location(episode_id)
      # Accessing episode_id may modify _episodes_loc_to_id_map upstream, so
      # ensure that we've performed that modification *first*.
      with tf.control_dependencies([episode_id]):
        episode_at_location = self._episodes_loc_to_id_map[episode_location]
      episode_valid = tf.equal(episode_at_location, episode_id)
      assert_valid = tf.Assert(episode_valid, [
          'Episode id', episode_id, 'is not valid.  It points to location',
          episode_location, 'but the episode at that location is currently id',
          episode_at_location
      ])
      with tf.control_dependencies([assert_valid, episode_location]):
        return self._data_table.get_episode_values(episode_location)

  def _maybe_end_episode(self, episode_id, end_episode=False):
    """Mark episode ID as complete when end_episode is True.

    Args:
      episode_id: A Tensor containing the current episode_id.
      end_episode: A Boolean Tensor whether should end the episode_id.

    Returns:
      A Boolean Tensor whether the episode was marked as complete.
    """
    episode_location = self._get_episode_id_location(episode_id)

    def _maybe_end_episode_id():
      """Maybe end episode ID."""
      def _end_episode_id():
        return tf.group(
            tf.compat.v1.scatter_update(self._episode_completed,
                                        [episode_location], 1))

      episode_valid = tf.equal(
          self._episodes_loc_to_id_map.sparse_read(episode_location),
          episode_id)

      pred_value = end_episode & (episode_id >= 0) & episode_valid
      if pred_value.shape.rank != 0:
        raise ValueError('Invalid condition shape: {} (should be scalar).'
                         .format(pred_value.shape))
      maybe_end = tf.cond(
          pred=pred_value,
          true_fn=_end_episode_id,
          false_fn=tf.no_op,
          name='maybe_end_episode_id')
      with tf.control_dependencies([maybe_end]):
        return self._episode_completed.sparse_read(episode_location) > 0

    return self._add_episode_critical_section.execute(_maybe_end_episode_id)

  def _maybe_end_batch_episodes(self, batch_episode_ids, end_episode=False):
    """Mark episode ID as complete when end_episode is True.

    Args:
      batch_episode_ids: A Tensor int64 with a batch of episode_ids
        with shape `(batch_size,)`.
      end_episode: A Boolean Tensor whether should end all batch_episode_ids,
        or Tensor with shape `(batch_size,)` to mark which ones to end.
    Returns:
      A `bool` Tensor `(batch_size,)` with the episode_ids marked as complete.
    """
    episodes_location = self._get_episode_id_location(batch_episode_ids)

    def _execute():
      """Maybe end episode ID."""
      valid_episodes = tf.equal(
          batch_episode_ids,
          self._episodes_loc_to_id_map.sparse_read(episodes_location))
      maybe_end_mask = end_episode & (batch_episode_ids >= 0) & valid_episodes
      episodes_location_ = tf.boolean_mask(
          tensor=episodes_location, mask=maybe_end_mask)
      update_completed = tf.compat.v1.scatter_update(self._episode_completed,
                                                     episodes_location_, 1)
      with tf.control_dependencies([update_completed]):
        return self._episode_completed.sparse_read(episodes_location) > 0

    return self._add_episode_critical_section.execute(_execute)

  def _get_episode_id(self,
                      episode_id,
                      begin_episode=False,
                      end_episode=False):
    """Increments the episode_id when begin_episode is True.

    Args:
      episode_id: A Tensor containing the current episode_id.
      begin_episode: A Boolean Tensor whether should increment the episode_id.
      end_episode: A Boolean Tensor whether should end the episode_id.

    Returns:
      An updated episode id value.  Accessing this episode id value will,
      as a side effect, start or end the current episode in the var.
    """
    def _assign_new_episode_id():
      """Increment the episode_id inside a critical section."""
      new_episode_id = self._last_episode.assign_add(1)
      episode_location = self._get_episode_id_location(new_episode_id)
      update_mapping = tf.compat.v1.scatter_update(
          self._episodes_loc_to_id_map, episode_location, new_episode_id)
      update_completed = tf.compat.v1.scatter_update(self._episode_completed,
                                                     episode_location, 0)
      reset_data = self._data_table.clear_rows(
          tf.expand_dims(episode_location, 0))
      reset_length = tf.compat.v1.scatter_update(self._episode_lengths,
                                                 episode_location, 0)
      with tf.control_dependencies([
          update_mapping, update_completed, reset_data, reset_length]):
        return tf.identity(new_episode_id)

    def _get_new_episode_id():
      return self._add_episode_critical_section.execute(_assign_new_episode_id)

    begin_episode = tf.convert_to_tensor(
        value=begin_episode, name='begin_episode')
    end_episode = tf.convert_to_tensor(value=end_episode, name='end_episode')
    # If episode_id value is still -1 we need to assign a proper value.
    pred_value = begin_episode | (episode_id < 0)
    if pred_value.shape.rank != 0:
      raise ValueError('Invalid condition predicate shape: {} '
                       '(should be scalar).'
                       .format(pred_value.shape))
    updated_episode_id = tf.cond(
        pred=pred_value,
        true_fn=_get_new_episode_id,
        false_fn=lambda: tf.identity(episode_id),
        name='get_episode_id')

    # _maybe_end_episode acquires the critical section.
    mark_completed = self._maybe_end_episode(updated_episode_id, end_episode)
    with tf.control_dependencies([mark_completed]):
      updated_episode_id = tf.identity(updated_episode_id)
    return updated_episode_id

  def _get_batch_episode_ids(self,
                             batch_episode_ids,
                             begin_episode=False,
                             end_episode=False,
                             mask=None):
    """Increments the episode_id of the elements that have begin_episode True.

    Mark as completed those that have end_episode True.

    Args:
      batch_episode_ids: A tf.int64 tensor with shape `(num_episodes,)`
        containing a one or more episodes IDs.
      begin_episode: A Boolean Tensor whether should increment each episode ID.
         It can be a scalar or a vector with the same dimensions of
         batch_episode_ids.
      end_episode: A Boolean Tensor whether should end each episode ID.
         It can be a scalar or a vector with the same dimensions of
         batch_episode_ids.
      mask: An optional Boolean Tensor to select which IDs are updated.
    Returns:
      An Tensor shaped `(num_episodes,)` with the updated episode IDs.
    Raises:
      ValueError: If the shape of `begin_episode` is not compatible with
        `batch_episode_ids`.

    """
    if batch_episode_ids.shape.rank != 1:
      raise ValueError(
          'batch_episode_ids must be a vector with 1 dimension')
    # If batch_episode_ids value is still -1 we need to assign a
    # proper value.  Find which IDs need to be updated.
    begin_episode = tf.convert_to_tensor(value=begin_episode)
    end_episode = tf.convert_to_tensor(value=end_episode)
    ids_to_update_mask = ((batch_episode_ids < 0) | begin_episode)
    if mask is not None:
      ids_to_update_mask &= mask

    def _update_batch_episode_ids():
      """Increment the episode_id inside a critical section."""
      num_ids = tf.reduce_sum(
          input_tensor=tf.cast(ids_to_update_mask, tf.int64))
      end_id = self._last_episode.assign_add(num_ids).value() + 1
      start_id = end_id - num_ids
      new_batch_episode_ids = tf.range(start_id, end_id)
      # Update when b/74385543 is fixed.
      ids_to_update = tf.compat.v1.where(ids_to_update_mask)
      scattered_updated_episode_ids = tf.scatter_nd(
          ids_to_update, new_batch_episode_ids,
          shape=tf.shape(batch_episode_ids, out_type=tf.int64))
      updated_batch_episode_ids = tf.compat.v1.where(
          ids_to_update_mask,
          scattered_updated_episode_ids,
          batch_episode_ids)
      episode_locations = tf.math.mod(new_batch_episode_ids, self._capacity)
      update_mapping = tf.compat.v1.scatter_update(self._episodes_loc_to_id_map,
                                                   [episode_locations],
                                                   [new_batch_episode_ids])
      reset_completed = tf.compat.v1.scatter_update(self._episode_completed,
                                                    [episode_locations], 0)
      reset_data = self._data_table.clear_rows(episode_locations)
      reset_length = tf.compat.v1.scatter_update(self._episode_lengths,
                                                 episode_locations, 0)
      with tf.control_dependencies([
          update_mapping, reset_completed, reset_data, reset_length]):
        return tf.identity(updated_batch_episode_ids)

    episode_ids = self._add_episode_critical_section.execute(
        _update_batch_episode_ids)

    # _maybe_end_batch_episodes acquires the critical section.
    mark_completed = self._maybe_end_batch_episodes(episode_ids, end_episode)

    with tf.control_dependencies([mark_completed]):
      episode_ids = tf.identity(episode_ids)
    return episode_ids

  def _get_episode_id_location(self, episode_id):
    return tf.math.mod(episode_id, self._capacity)

  def _sample_episode_ids(self, shape, weigh_by_episode_length=False,
                          seed=None):
    """Samples episode ids from the replay buffer."""
    last_id = self._get_last_episode_id()
    assert_nonempty = tf.compat.v1.assert_non_negative(
        last_id,
        message='EpisodicReplayBuffer is empty. Make sure to add items '
        'before sampling the buffer.')
    if weigh_by_episode_length:
      # Sample episodes proportional to length.
      with tf.control_dependencies([assert_nonempty]):
        num_episodes = tf.minimum(self._last_episode + 1, self._capacity)
        episode_lengths = self._episode_lengths[:num_episodes]
        logits = tf.math.log(tf.cast(episode_lengths, tf.float32))
        return tf.reshape(
            tf.random.categorical(
                [logits],  # shape is: [1, num_episodes]
                num_samples=tf.reduce_prod(shape),
                seed=seed,
                dtype=tf.int64),
            shape)
    else:
      min_val, max_val = _valid_range_ids(self._get_last_episode_id(),
                                          self._capacity)
      with tf.control_dependencies([assert_nonempty]):
        return tf.random.uniform(
            shape, minval=min_val, maxval=max_val, dtype=tf.int64, seed=seed)

  def _get_last_episode_id(self):
    def last_episode():
      return self._last_episode.value()

    return self._add_episode_critical_section.execute(last_episode)

  def _increment_episode_length_locked(self, episode_id, increment=1):
    """Increments the length of episode_id in a thread safe manner.

    NOTE: This method should only be called inside a critical section.

    Args:
      episode_id: int64 scalar of vector. ID(s) of the episode(s) for which we
        will increase the length.
      increment: Amount to increment episode_length by.
    Returns:
      An op that increments the last_id.
    Raises:
      ValueError: If `len(episode_id.shape) > 1`.
    """
    episode_location = self._get_episode_id_location(episode_id)

    def _assign_add():
      new_length = self._episode_lengths[episode_location] + increment
      update_length = tf.compat.v1.scatter_update(
          self._episode_lengths, [episode_location], new_length)
      with tf.control_dependencies([update_length]):
        return tf.identity(new_length)

    def _assign_add_multiple():
      new_length = tf.gather(self._episode_lengths, episode_location)
      new_length += increment
      update_length = tf.compat.v1.scatter_update(self._episode_lengths,
                                                  episode_location, new_length)
      with tf.control_dependencies([update_length]):
        return tf.identity(new_length)

    if episode_location.shape.rank == 0:
      return _assign_add()
    elif episode_location.shape.rank == 1:
      return _assign_add_multiple()
    else:
      raise ValueError('episode_id must have rank <= 1')

  def _get_valid_ids_mask_locked(self, episode_ids):
    """Returns a mask of whether the given IDs are valid. Caller must lock."""
    episode_locations = self._get_episode_id_location(episode_ids)
    location_matches_id = tf.equal(
        episode_ids,
        self._episodes_loc_to_id_map.sparse_read(episode_locations))
    # Note that the above map is initialized with -1s.
    return (episode_ids >= 0) & location_matches_id

  def get_valid_ids_mask(self, episode_ids):
    """Returns a mask of whether the given IDs are valid."""
    with tf.device(self._device):
      with tf.name_scope('get_valid_ids_mask'):
        return self._add_episode_critical_section.execute(
            lambda: self._get_valid_ids_mask_locked(episode_ids))

  def extract(self, locations, clear_data=False):
    """Extracts Episodes with the given IDs.

    Args:
      locations: A `1-D` Tensor of locations to extract from (note, this is
        NOT the same as an episode ids variable).  It's up to the user to
        ensure that only valid episode locations are requested (i.e.,
        values should be between `0` and `self.capacity`).
        Passing locations that are out of bounds will lead to runtime errors.
      clear_data: If `True`, clears the extracted data from this buffer.

    Returns:
      episodes: An Episodes object with an outer dimension of the same size as
        locations.
    """
    locations = tf.cast(locations, dtype=tf.int64, name='locations')
    locations.shape.assert_has_rank(1)

    def _extract_locked():
      """Does the above within the buffer's critical section."""
      episodes = Episodes(
          length=self._episode_lengths.sparse_read(locations),
          completed=self._episode_completed.sparse_read(locations),
          tensor_lists=self._data_table.get_episode_lists(locations))
      if clear_data:
        with tf.control_dependencies(tf.nest.flatten(episodes)):
          clear_rows = self._data_table.clear_rows(locations)
          clear_lengths = tf.compat.v1.scatter_update(self._episode_lengths,
                                                      locations, 0)
          clear_completed = tf.compat.v1.scatter_update(self._episode_completed,
                                                        locations, 0)
        with tf.control_dependencies(
            [clear_rows, clear_lengths, clear_completed]):
          episodes = tf.nest.map_structure(tf.identity, episodes)
      return episodes

    with tf.device(self._device):
      with tf.name_scope('extract'):
        return self._add_episode_critical_section.execute(_extract_locked)

  def extend_episodes(self,
                      episode_ids,
                      episode_ids_indices,
                      episodes):
    """Extends a batch of episodes in this buffer.

    Args:
      episode_ids: A int64 vector containing the ids of the
        episodes the items are being added to.  Shaped `(max_num_episodes,)`.
      episode_ids_indices: An int64 vector containing the locations in
        `episode_ids` that are being extended.  Shaped `(num_episodes,)`,
        where `num_episodes <= max_num_episodes`.  Rows in `episodes`
        correspond to locations in `episode_ids_indices`.
      episodes: An `Episodes` tuple containing the extension data. Tensors must
        have outer dimension of size `num_episodes`.

    Returns:
      A `Tensor` containing the updated episode ids.  Accessing or executing
      this tensor also extends episodes in the replay buffer.
    """
    episode_ids.shape.assert_has_rank(1)
    episode_ids_indices = tf.convert_to_tensor(
        value=episode_ids_indices, name='episode_ids_indices')
    episode_ids_indices.shape.assert_has_rank(1)

    def _extend_locked(episode_ids, expanded_episode_ids):
      """Does the above within the buffer's critical section."""
      episode_locations = self._get_episode_id_location(episode_ids)
      episode_valid = tf.equal(
          self._episodes_loc_to_id_map.sparse_read(episode_locations),
          episode_ids)
      episode_valid_idx = tf.reshape(tf.compat.v1.where(episode_valid), [-1])
      episode_locations = tf.gather(episode_locations, episode_valid_idx)
      increment_lengths = self._increment_episode_length_locked(
          episode_locations,
          tf.gather(episodes.length, episode_valid_idx))
      set_completed = tf.compat.v1.scatter_update(
          self._episode_completed, episode_locations,
          tf.gather(episodes.completed, episode_valid_idx))
      extend = self._data_table.extend(
          episode_locations,
          tf.nest.map_structure(lambda tl: tf.gather(tl, episode_valid_idx),
                                episodes.tensor_lists))
      with tf.control_dependencies([increment_lengths, set_completed, extend]):
        return tf.identity(expanded_episode_ids)

    with tf.device(self._device):
      with tf.name_scope('extend'):
        episode_ids_indices_shape = tf.shape(episode_ids_indices)
        begin_episode = self._begin_episode_fn(episodes)
        begin_episode = tf.broadcast_to(
            begin_episode, episode_ids_indices_shape, name='begin_episode')
        column_indices = tf.reshape(episode_ids_indices, [-1, 1])
        episode_ids_shape = tf.shape(input=episode_ids)
        # We expand the tensors below from size `num_episodes` (the size of
        # episode_ids_indices) to tensors of  size `max_num_episodes` (the size
        # of episode_ids).
        expanded_begin_episode = tf.scatter_nd(column_indices, begin_episode,
                                               episode_ids_shape)
        expanded_mask = tf.scatter_nd(column_indices,
                                      tf.fill(episode_ids_indices_shape, True),
                                      episode_ids_shape)
        expanded_episode_ids = self._get_batch_episode_ids(
            episode_ids,
            begin_episode=expanded_begin_episode,
            mask=expanded_mask)
        episode_ids = tf.gather(expanded_episode_ids, episode_ids_indices)
        return self._add_episode_critical_section.execute(
            lambda: _extend_locked(episode_ids, expanded_episode_ids))


class StatefulEpisodicReplayBuffer(replay_buffer_base.ReplayBuffer):
  """Wrapper enabling use of `EpisodicReplayBuffer` with a `Driver`.

  This wrapper keeps track of episode ids in a `tf.Variable`.

  Use:
  ```python
  tf_env = ...
  rb = EpisodicReplayBuffer(...)
  stateful_rb = StatefulEpisodicReplayBuffer(rb, num_episodes=tf_env.batch_size)
  driver = DynamicEpisodeDriver(
      tf_env, policy, ...,
      observers=[
        lambda traj: stateful_rb.add_batch(traj),
      ])
  driver.run()
  ```
  """

  def __init__(self, replay_buffer, num_episodes=None):
    """Create a `StatefulEpisodicReplayBuffer` for `num_episodes` batches.

    Args:
      replay_buffer: An instance of `EpisodicReplayBuffer`.
      num_episodes: (Optional) integer, number of episode IDs to create.
        If `None`, a scalar ID variable is returned.

    Raises:
      TypeError: If `replay_buffer` is not an `EpisodicReplayBuffer`.
    """
    super(StatefulEpisodicReplayBuffer, self).__init__(
        replay_buffer.data_spec, replay_buffer.capacity)

    if not isinstance(replay_buffer, EpisodicReplayBuffer):
      raise TypeError(
          'Expected an EpisodicReplayBuffer, saw {}'.format(replay_buffer))
    shape = ()
    if num_episodes and num_episodes > 0:
      if num_episodes > replay_buffer.capacity:
        raise ValueError('Buffer cannot create episode_ids when '
                         'num_episodes {} > capacity {}.'.format(
                             num_episodes, replay_buffer.capacity))
      shape = (num_episodes,)
    self._replay_buffer = replay_buffer
    self._episode_ids_var = common.create_variable(
        'episode_id', initial_value=_INVALID_EPISODE_ID,
        shape=shape, use_local_variable=True)

  @property
  def episode_ids(self):
    """Returns the `tf.Variable` tracking the episode ids."""
    return self._episode_ids_var

  @common.function_in_tf1()
  def add_batch(self, items):
    """Adds a batch of single steps for the corresponding episodes IDs.

    Args:
      items: A batch of items to be added to the buffer. Items will have the
        same structure as the data_spec of this class, but the tensors in items
        will have an extra outer dimension `(num_episodes, ...)` in addition to
        the corresponding spec in data_spec.

    Returns:
      A `Tensor` containing the updated episode ids.  Accessing or executing
      this varaible also adds `items` to the replay buffer.
    """
    new_episode_ids = self._replay_buffer.add_batch(
        items=items, episode_ids=self._episode_ids_var)
    self._episode_ids_var.assign(new_episode_ids)
    return new_episode_ids

  @common.function_in_tf1()
  def add_sequence(self, items):
    """Adds a sequence of items to the replay buffer for the selected episode.

    Args:
      items: A sequence of items to be added to the buffer. Items will have the
        same structure as the data_spec of this class, but the tensors in items
        will have an outer sequence dimension in addition to the corresponding
        spec in data_spec.

    Returns:
      An updated episode id value.  Accessing this episode id value will,
      as a side effect, start or end the current episode.
    """
    new_episode_id = self._replay_buffer.add_sequence(
        items=items, episode_id=self._episode_ids_var)
    self._episode_ids_var.assign(new_episode_id)
    return new_episode_id

  @common.function_in_tf1()
  def extend_episodes(self,
                      episode_ids_indices,
                      episodes):
    """Extends a batch of episodes in this buffer.

    Args:
      episode_ids_indices: An int64 vector containing the locations in
        `self.episode_ids` that are being extended.  Shaped `(num_episodes,)`,
        where `num_episodes <= max_num_episodes`.  Rows in `episodes` and
        `begin_episode` correspond to locations in `episode_ids_indices`.
      episodes: An `Episodes` tuple containing the extension data. Tensors must
        have outer dimension of size `num_episodes`.

    Returns:
      A `Tensor` containing the updated episode ids.  Accessing or executing
      this tensor also extends the replay buffer.
    """
    new_episode_ids = self._replay_buffer.extend_episodes(
        episode_ids=self._episode_ids_var,
        episode_ids_indices=episode_ids_indices,
        episodes=episodes)
    self._episode_ids_var.assign(new_episode_ids)
    return new_episode_ids

  def _get_next(
      self, sample_batch_size=None, num_steps=None, time_stacked=None):
    return self._replay_buffer.get_next(
        sample_batch_size, num_steps, time_stacked)

  def _as_dataset(
      self, sample_batch_size=None, num_steps=None,
      sequence_preprocess_fn=None, num_parallel_calls=None):
    return self._replay_buffer.as_dataset(
        sample_batch_size,
        num_steps,
        sequence_preprocess_fn=sequence_preprocess_fn,
        num_parallel_calls=num_parallel_calls)


def _valid_range_ids(last_id, capacity):
  """Returns the [min_val, max_val) range of ids.

  Args:
    last_id: A tensor that indicates the last id stored in the replay buffer.
    capacity: The maximum number of elements that the replay buffer can hold.

  Returns:
    A tuple (min_id, max_id) for the range [min_id, max_id) of valid ids.
  """
  min_id_non_full = tf.constant(0, dtype=tf.int64)
  max_id_non_full = tf.maximum(last_id + 1, 0)

  min_id_full = tf.cast(last_id + 1 - capacity, dtype=tf.int64)
  max_id_full = tf.cast(last_id + 1, dtype=tf.int64)

  return (
      tf.where(last_id < capacity, min_id_non_full, min_id_full),
      tf.where(last_id < capacity, max_id_non_full, max_id_full))
