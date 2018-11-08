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

"""Uniform replay buffer in python.

The base class provides all the functionalities of a uniform replay buffer:
  - add samples in a First In First Out way.
  - read samples uniformly.

PyTrajectoryHashedUniformReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import threading

import numpy as np
import tensorflow as tf

from tf_agents.environments import trajectory
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils

nest = tf.contrib.framework.nest


class NumpyStorage(tf.contrib.checkpoint.Checkpointable):
  """A class to store nested objects in a collection of numpy arrays."""

  def __init__(self, data_spec, capacity):
    """Creates a NumpyStorage object.

    Args:
      data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
        single item that can be stored in this table.
      capacity: The maximum number of items that can be stored in the buffer.

    Raises:
      ValueError: If data_spec is not an instance or nest of ArraySpecs.
    """
    self._capacity = capacity
    if not all([isinstance(spec, array_spec.ArraySpec)
                for spec in nest.flatten(data_spec)]):
      raise ValueError('The data_spec parameter must be an instance or nest of '
                       'array_spec.ArraySpec. Got: {}'.format(data_spec))
    self._data_spec = data_spec
    self._flat_specs = nest.flatten(data_spec)
    self._np_state = tf.contrib.checkpoint.NumpyState()

    self._buf_names = tf.contrib.checkpoint.NoDependency([])
    for idx in range(len(self._flat_specs)):
      self._buf_names.append('buffer{}'.format(idx))
      # Set each buffer to a sentinel value (real buffers will never be
      # scalars) rather than a real value so that if they are restored from
      # checkpoint, we don't end up double-initializing. We don't leave them
      # as unset because setting them to a numpy value tells the checkpointer
      # this will be a checkpointed attribute and it creates TF ops for it.
      setattr(self._np_state, self._buf_names[idx], np.int64(0))

  def _array(self, index):
    """Creates or retrieves one of the numpy arrays backing the storage."""
    array = getattr(self._np_state, self._buf_names[index])
    if np.isscalar(array) or array.ndim == 0:
      spec = self._flat_specs[index]
      shape = (self._capacity,) + spec.shape
      array = np.zeros(shape=shape, dtype=spec.dtype)
      setattr(self._np_state, self._buf_names[index], array)
    return array

  def get(self, idx):
    """Get value stored at idx."""
    encoded_item = []
    for buf_idx in range(len(self._flat_specs)):
      encoded_item.append(self._array(buf_idx)[idx])
    return nest.pack_sequence_as(self._data_spec, encoded_item)

  def set(self, table_idx, value):
    """Set table_idx to value."""
    for nest_idx, element in enumerate(nest.flatten(value)):
      self._array(nest_idx)[table_idx] = element


class PyUniformReplayBuffer(tf.contrib.checkpoint.Checkpointable):
  """A uniform replay buffer.

  Writing and reading to this replay buffer is thread safe.
  """

  def __init__(self, data_spec, capacity):
    """Creates a PyUniformReplayBuffer.

    Args:
      data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
        single item that can be stored in this buffer.
      capacity: The maximum number of items that can be stored in the buffer.
    """
    self._capacity = capacity
    if not hasattr(self, '_data_spec'):
      # If subclass has already set data_spec, ignore this one (there may be
      # spec differences between what the subclass encodes and what the
      # subclass decodes).
      self._data_spec = data_spec
    self._storage = NumpyStorage(data_spec, capacity)
    self._lock = threading.Lock()
    self._np_state = tf.contrib.checkpoint.NumpyState()

    # Adding elements to the replay buffer is done in a circular way.
    # Keeps track of the actual size of the replay buffer and the location
    # where to add new elements.
    self._np_state.size = np.int64(0)
    self._np_state.cur_id = np.int64(0)

    # Total number of items that went through the replay buffer.
    self._np_state.item_count = np.int64(0)

  def _encode(self, item):
    """Encodes an item (before adding it to the buffer)."""
    return item

  def _decode(self, encoded_item):
    """Decodes an item."""
    return encoded_item

  def _on_delete(self, encoded_item):
    """Do any necessary cleanup."""
    pass

  @property
  def size(self):
    return self._np_state.size

  def add(self, item):
    """Adds an item to the replay buffer.

    When the replay buffer is full, the item replaces the oldest item.

    Args:
      item: An array or list/tuple/nest of arrays representing a single item to
        be added to the replay buffer. `item` must match the data_spec of this
        class.
    """
    encoded_item = self._encode(item)
    with self._lock:
      if self._np_state.size == self._capacity:
        # If we are at capacity, we are deleting element cur_id.
        self._on_delete(self._storage.get(self._np_state.cur_id))
      self._storage.set(self._np_state.cur_id, encoded_item)
      self._np_state.size = np.minimum(self._np_state.size + 1, self._capacity)
      self._np_state.cur_id = (self._np_state.cur_id + 1) % self._capacity
      self._np_state.item_count += 1

  def get_next(self, num_steps=1, time_stacked=True):
    with self._lock:
      if self._np_state.size <= 0:
        raise ValueError('Read error: empty replay buffer')

      idx = np.random.randint(self._np_state.size - num_steps + 1)
      if self._np_state.size == self._capacity:
        # If the buffer is full, add cur_id (head of circular buffer) so that
        # we sample from the range [cur_id, cur_id + size - num_steps]. We will
        # modulo the size below.
        idx += self._np_state.cur_id

      if num_steps > 1:
        # TODO(sfishman): Try getting data from numpy in one shot rather than
        # num_steps.
        encoded_item = [self._storage.get((idx + n) % self._capacity)
                        for n in range(num_steps)]
      else:
        encoded_item = self._storage.get(idx % self._capacity)

    if num_steps > 1:
      item = [self._decode(item) for item in encoded_item]
      if time_stacked:
        item = nest_utils.stack_nested_arrays(item)
    else:
      item = self._decode(encoded_item)
    return item

  def as_dataset(self, batch_size=None, num_steps=1):
    """Returns a dataset that samples a random trajectory or sequence.

    Args:
      batch_size: If None, return a single item from the replay buffer.
        Otherwise, batch `batch_size` items along the first dimension.
      num_steps: The number of steps to fetch per sample. Steps will be
        stacked along the second dimension.

    Returns:
      If num_steps == 1: a Dataset returning single items from the replay
        buffer, with a batch dimension added in front if batch_size is not
        None.
      If num_steps > 1: a Dataset returning items from
        the replay buffer, where each element in the item nest is of shape
        [B x T x F...], where B is the batch dimension, T is the time
        dimension, and F is the feature shape. If batch_size is None, there is
        no batch dimension.
    """
    data_spec = self._data_spec
    if batch_size is not None:
      data_spec = array_spec.add_outer_dims_nest(data_spec, (batch_size,))
    if num_steps > 1:
      data_spec = (data_spec,) * num_steps
    shapes = tuple(s.shape for s in nest.flatten(data_spec))
    dtypes = tuple(s.dtype for s in nest.flatten(data_spec))

    def generator_fn():
      while True:
        if batch_size is not None:
          batch = [self.get_next(num_steps=num_steps, time_stacked=False)
                   for _ in range(batch_size)]
          item = nest_utils.stack_nested_arrays(batch)
        else:
          item = self.get_next(num_steps=num_steps, time_stacked=False)
        yield tuple(nest.flatten(item))

    def time_stack(*structures):
      time_axis = 0 if batch_size is None else 1
      return nest.map_structure(
          lambda *elements: tf.stack(elements, axis=time_axis), *structures)

    ds = tf.data.Dataset.from_generator(generator_fn, dtypes, shapes).map(
        lambda *items: nest.pack_sequence_as(data_spec, items))
    if num_steps > 1:
      return ds.map(time_stack)
    else:
      return ds


class FrameBuffer(tf.contrib.checkpoint.PythonStateWrapper):
  """Saves some frames in a memory efficient way.

  Thread safety: cannot add multiple frames in parallel.
  """

  def __init__(self):
    self._frames = {}

  def add_frame(self, frame):
    """Add a frame to the buffer.

    Args:
      frame: Numpy array.

    Returns:
      A deduplicated frame.
    """
    h = hash(frame.tostring())
    if h in self._frames:
      _, refcount = self._frames[h]
      self._frames[h] = (frame, refcount + 1)
      return h
    self._frames[h] = (frame, 1)
    return h

  def __len__(self):
    return len(self._frames)

  def _serialize(self):
    """Callback for `PythonStateWrapper` to serialize the dictionary."""
    return pickle.dumps(self._frames)

  def _deserialize(self, string_value):
    """Callback for `PythonStateWrapper` to deserialize the array."""
    self._frames = pickle.loads(string_value)

  def compress(self, observation, split_axis=-1):
    # e.g. When split_axis is -1, turns an array of size 84x84x4
    # into a list of arrays of size 84x84x1.
    frame_list = np.split(observation, observation.shape[split_axis],
                          split_axis)
    return np.array([self.add_frame(f) for f in frame_list])

  def decompress(self, observation, split_axis=-1):
    frames = [self._frames[h][0] for h in observation]
    return np.concatenate(frames, axis=split_axis)

  def on_delete(self, observation, split_axis=-1):
    for h in observation:
      frame, refcount = self._frames[h]
      if refcount > 1:
        self._frames[h] = (frame, refcount - 1)
      else:
        del self._frames[h]


class PyTrajectoryHashedUniformReplayBuffer(PyUniformReplayBuffer):
  """Uniform replay buffer of trajectories with optimized underlying storage.
  """

  def __init__(self, data_spec, capacity, log_interval=None):
    if not isinstance(data_spec, trajectory.Trajectory):
      raise ValueError(
          'data_spec must be the spec of a trajectory: {}'.format(data_spec))
    self._data_spec = data_spec
    super(PyTrajectoryHashedUniformReplayBuffer, self).__init__(
        self._compressed_data_spec(), capacity)
    self._frame_buffer = FrameBuffer()
    self._lock_frame_buffer = threading.Lock()
    self._log_interval = log_interval

  def _compressed_data_spec(self):
    observation = self._data_spec.observation
    observation = array_spec.ArraySpec(
        shape=(observation.shape[-1],), dtype=np.int64)
    return self._data_spec._replace(observation=observation)

  def _encode(self, traj):
    """Encodes a trajectory for efficient storage.

    The observations in this trajectory are replaced by a compressed
    version of the observations: each frame is only stored exactly once.

    Args:
      traj: The original trajectory.

    Returns:
      The same trajectory where frames in the observation have been
      de-duplicated.
    """
    with self._lock_frame_buffer:
      observation = self._frame_buffer.compress(traj.observation)

    if (self._log_interval and
        self._np_state.item_count % self._log_interval == 0):
      tf.logging.info('Effective Replay buffer frame count: {}'.format(
          len(self._frame_buffer)))

    return traj._replace(observation=observation)

  def _decode(self, encoded_trajectory):
    """Decodes a trajectory.

    The observation in the trajectory has been compressed so that no frame
    is present more than once in the replay buffer. Uncompress the observations
    in this trajectory.

    Args:
      encoded_trajectory: The compressed version of the trajectory.

    Returns:
      The original trajectory (uncompressed).
    """
    observation = self._frame_buffer.decompress(encoded_trajectory.observation)
    return encoded_trajectory._replace(observation=observation)

  def _on_delete(self, encoded_trajectory):
    with self._lock_frame_buffer:
      self._frame_buffer.on_delete(encoded_trajectory.observation)

