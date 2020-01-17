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

"""NumpyStorage stores nested objects across multiple numpy arrays."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import array_spec

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.training.tracking import base  # TF internal
from tensorflow.python.training.tracking import data_structures  # TF internal
# pylint:enable=g-direct-tensorflow-import


# TODO(b/126551076) Migrate to public APIs
class NumpyState(base.Trackable):
  """A checkpointable object whose NumPy array attributes are saved/restored.

  Example usage:

  ```python
  arrays = numpy_storage.NumpyState()
  checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
  arrays.x = np.ones([3, 4])
  directory = self.get_temp_dir()
  prefix = os.path.join(directory, 'ckpt')
  save_path = checkpoint.save(prefix)
  arrays.x[:] = 0.
  assert (arrays.x == np.zeros([3, 4])).all()
  checkpoint.restore(save_path)
  assert (arrays.x == np.ones([3, 4])).all()

  second_checkpoint = tf.train.Checkpoint(
      numpy_arrays=numpy_storage.NumpyState())
  # Attributes of NumpyState objects are created automatically by restore()
  second_checkpoint.restore(save_path)
  assert (second_checkpoint.numpy_arrays.x == np.ones([3, 4])).all()
  ```

  Note that `NumpyState` objects re-create the attributes of the previously
  saved object on `restore()`. This is in contrast to TensorFlow variables, for
  which a `Variable` object must be created and assigned to an attribute.

  This snippet works both when graph building and when executing eagerly. On
  save, the NumPy array(s) are fed as strings to be saved in the checkpoint (via
  a placeholder when graph building, or as a string constant when executing
  eagerly). When restoring they skip the TensorFlow graph entirely, and so no
  restore ops need be run. This means that restoration always happens eagerly,
  rather than waiting for `checkpoint.restore(...).run_restore_ops()` like
  TensorFlow variables when graph building.
  """

  def _lookup_dependency(self, name):
    """Create placeholder NumPy arrays for to-be-restored attributes.

    Typically `_lookup_dependency` is used to check by name whether a dependency
    exists. We cheat slightly by creating a checkpointable object for `name` if
    we don't already have one, giving us attribute re-creation behavior when
    loading a checkpoint.

    Args:
      name: The name of the dependency being checked.

    Returns:
      An existing dependency if one exists, or a new `_NumpyWrapper` placeholder
      dependency (which will generally be restored immediately).
    """
    value = super(NumpyState, self)._lookup_dependency(name)
    if value is None:
      value = _NumpyWrapper(np.array([]))
      new_reference = base.TrackableReference(name=name, ref=value)
      self._unconditional_checkpoint_dependencies.append(new_reference)
      self._unconditional_dependency_names[name] = value
      super(NumpyState, self).__setattr__(name, value)
    return value

  def __getattribute__(self, name):
    """Un-wrap `_NumpyWrapper` objects when accessing attributes."""
    value = super(NumpyState, self).__getattribute__(name)
    if isinstance(value, _NumpyWrapper):
      return value.array
    return value

  def __setattr__(self, name, value):
    """Automatically wrap NumPy arrays assigned to attributes."""
    # TODO(b/126429928): Consider supporting lists/tuples.
    if isinstance(value, (np.ndarray, np.generic)):
      try:
        existing = super(NumpyState, self).__getattribute__(name)
        existing.array = value
        return
      except AttributeError:
        value = _NumpyWrapper(value)
        self._track_trackable(value, name=name, overwrite=True)
    elif (name not in ('_self_setattr_tracking', '_self_update_uid',
                       # TODO(b/130295584): Remove these non-_self aliases when
                       # sync issues are resolved.
                       '_setattr_tracking', '_update_uid')
          and getattr(self, '_setattr_tracking', True)):
      # Mixing restore()-created attributes with user-added checkpointable
      # objects is tricky, since we can't use the `_lookup_dependency` trick to
      # re-create attributes (we might accidentally steal the restoration for
      # another checkpointable object). For now `NumpyState` objects must be
      # leaf nodes. Theoretically we could add some extra arguments to
      # `_lookup_dependency` to figure out whether we should create a NumPy
      # array for the attribute or not.
      raise NotImplementedError(
          ('Assigned %s to the %s property of %s, which is not a NumPy array. '
           'Currently mixing NumPy arrays and other checkpointable objects is '
           'not supported. File a feature request if this limitation bothers '
           'you.') % (value, name, self))
    super(NumpyState, self).__setattr__(name, value)


class _NumpyWrapper(tf.train.experimental.PythonState):
  """Wraps a NumPy array for storage in an object-based checkpoint."""

  def __init__(self, array):
    """Specify a NumPy array to wrap.

    Args:
      array: The NumPy array to save and restore (may be overwritten).
    """
    self.array = array

  def serialize(self):
    """Callback to serialize the array."""
    string_file = io.BytesIO()
    try:
      np.save(string_file, self.array, allow_pickle=False)
      serialized = string_file.getvalue()
    finally:
      string_file.close()
    return serialized

  def deserialize(self, string_value):
    """Callback to deserialize the array."""
    string_file = io.BytesIO(string_value)
    try:
      self.array = np.load(string_file, allow_pickle=False)
    finally:
      string_file.close()


class NumpyStorage(tf.Module):
  """A class to store nested objects in a collection of numpy arrays.

  If a data_spec of `{'foo': ArraySpec(shape=(4,), dtype=np.uint8), 'bar':
  ArraySpec(shape=(3, 7), dtype=np.float32)}` were used, then this would create
  two arrays, one for the 'foo' key and one for the 'bar' key. The .get and
  .set methods would return/take Python dictionaries, but break down the
  component arrays before storing them.
  """

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
    if not all([
        isinstance(spec, array_spec.ArraySpec)
        for spec in tf.nest.flatten(data_spec)
    ]):
      raise ValueError('The data_spec parameter must be an instance or nest of '
                       'array_spec.ArraySpec. Got: {}'.format(data_spec))
    self._data_spec = data_spec
    self._flat_specs = tf.nest.flatten(data_spec)
    self._np_state = NumpyState()

    self._buf_names = data_structures.NoDependency([])
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
    return tf.nest.pack_sequence_as(self._data_spec, encoded_item)

  def set(self, table_idx, value):
    """Set table_idx to value."""
    for nest_idx, element in enumerate(tf.nest.flatten(value)):
      self._array(nest_idx)[table_idx] = element
