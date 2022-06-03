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

"""Keras layer to reshape inner dimensions (keeping outer dimensions the same).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tf_agents.typing import types


def InnerReshape(current_shape: types.Shape,  # pylint: disable=invalid-name
                 new_shape: types.Shape,
                 **kwargs) -> tf.keras.layers.Layer:
  """Returns a Keras layer that reshapes the inner dimensions of tensors.

  Each tensor passed to an instance of `InnerReshape`, will be reshaped to:

  ```python
  shape(tensor)[:-len(current_shape)] + new_shape
  ```

  (after its inner shape is validated against `current_shape`).  Note:
  The `current_shape` may contain `None` (unknown) dimension values.

  This can be helpful when switching between `Dense`, `ConvXd`, and `RNN` layers
  in TF-Agents networks, in ways that are agnostic to whether the input has
  either `[batch_size]` or `[batch_size, time]` outer dimensions.

  For example, to switch between `Dense`, `Conv2D`, and `GRU` layers:

  ```python
  net = tf_agents.networks.Sequential([
    tf.keras.layers.Dense(32),
    # Convert inner dim from [32] to [4, 4, 2] for Conv2D.
    tf_agents.keras_layers.InnerReshape([None], new_shape=[4, 4, 2]),
    tf.keras.layers.Conv2D(2, 3),
    # Convert inner HWC dims [?, ?, 2] to [8] for Dense/RNN.
    tf_agents.keras_layers.InnerReshape([None, None, 2], new_shape=[-1]),
    tf.keras.layers.GRU(2, return_state=True, return_sequences=True)
  ])
  ```

  Args:
    current_shape: The current (partial) shape for the inner dims.
      This should be a `list`, `tuple`, or `tf.TensorShape` with known rank.
      The given current_shape must be compatible with the inner shape of the
      input.  Examples - `[]`, `[None]`, `[None] * 3`, `[3, 3, 4]`,
      `[3, None, 4]`.
    new_shape: The new shape for the inner dims.  The length of
      `new_shape` need not match the length of `current_shape`, but if both
      shapes are fully defined then the total number of elements must match.
      It may have up to one flexible (`-1`) dimension.  Examples -
      `[3]`, `[]`, `[-1]`, `[-1, 3]`.
    **kwargs: Additionnal args to the Keras core layer constructor, e.g. `name`.

  Returns:
    A new Keras `Layer` that performs the requested reshape on incoming tensors.

  Raises:
    ValueError: If `current_shape` has unknown rank.
    ValueError: If both shapes are fully defined and the number of elements
      doesn't match.
  """
  current_shape = tf.TensorShape(tf.get_static_value(current_shape))
  if current_shape.rank is None:
    raise ValueError('current_shape must have known rank')
  new_shape = tf.TensorShape([
      None if d == -1 else d for d in tf.get_static_value(new_shape)])
  if (current_shape.is_fully_defined()
      and new_shape.is_fully_defined()
      and (current_shape.num_elements()
           != new_shape.num_elements())):
    raise ValueError(
        'Mismatched number of elements in current and new inner shapes: '
        '{} vs. {}'.format(current_shape, new_shape))

  def reshape(t):
    return _reshape_inner_dims(t, current_shape, new_shape)
  return tf.keras.layers.Lambda(
      lambda inputs: tf.nest.map_structure(reshape, inputs), **kwargs)


def _reshape_inner_dims(
    tensor: tf.Tensor,
    shape: tf.TensorShape,
    new_shape: tf.TensorShape) -> tf.Tensor:
  """Reshapes tensor to: shape(tensor)[:-len(shape)] + new_shape."""
  tensor_shape = tf.shape(tensor)
  ndims = shape.rank
  tensor.shape[-ndims:].assert_is_compatible_with(shape)
  new_shape_inner_tensor = tf.cast(
      [-1 if d is None else d for d in new_shape.as_list()], tf.int32)
  new_shape_outer_tensor = tf.cast(
      tensor_shape[:-ndims], tf.int32)
  full_new_shape = tf.concat(
      (new_shape_outer_tensor, new_shape_inner_tensor), axis=0)
  new_tensor = tf.reshape(tensor, full_new_shape)
  new_tensor.set_shape(tensor.shape[:-ndims] + new_shape)
  return new_tensor
