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

"""Utilities for dealing with CompositeTensors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def shape(tensor):
  if isinstance(tensor, tf.SparseTensor):
    return tensor.dense_shape
  else:
    return tf.shape(input=tensor, out_type=tf.int64)


def reshape(t, shape):  # pylint: disable=redefined-outer-name
  """Reshape composite tensor `t` to `shape`.

  Args:
    t: A `Tensor` or `SparseTensor`.
    shape: `1D` tensor, array, or list.  The new shape.

  Returns:
    The reshaped tensor.
  """
  return (tf.sparse.reshape(t, shape) if isinstance(t, tf.SparseTensor)
          else tf.reshape(t, shape))


def squeeze(t, axis):
  """Squeeze composite tensor along axis `axis`.

  Args:
    t: A `Tensor` or `SparseTensor`.
    axis: A python integer.

  Returns:
    The tensor with dimension `axis` removed.

  Raises:
    InvalidArgumentError: If `t` is a `SparseTensor` and has more than one index
    stored along `axis`.
  """
  if isinstance(t, tf.SparseTensor):
    # Fill in a dummy value if there are no elements in the tensor.
    indices_axis = t.indices[:, axis]
    all_zero = tf.reduce_all(tf.equal(indices_axis, 0))
    with tf.control_dependencies([
        tf.Assert(
            all_zero,
            ['Unable to squeeze SparseTensor {} axis {} '
             'because indices are not all equal to 0:', indices_axis])]):
      return tf.SparseTensor(
          indices=tf.concat(
              (t.indices[:, :axis], t.indices[:, axis + 1:]),
              axis=1),
          values=t.values,
          dense_shape=tf.concat(
              (t.dense_shape[:axis], t.dense_shape[axis + 1:]),
              axis=0))
  else:
    return tf.squeeze(t, [axis])


def expand_dims(t, axis):
  """Add a new dimension to tensor `t` along `axis`.

  Args:
    t: A `tf.Tensor` or `tf.SparseTensor`.
    axis: A `0D` integer scalar.

  Returns:
    An expanded tensor.

  Raises:
    NotImplementedError: If `t` is a `SparseTensor` and `axis != 0`.
  """
  if isinstance(t, tf.SparseTensor):
    if tf.is_tensor(axis) or axis != 0:
      raise NotImplementedError(
          'Can only expand_dims on SparseTensor {} on static axis 0, '
          'but received axis {}'.format(t, axis))
    n_elem = (
        t.indices.shape[0] or tf.get_static_shape(t.dense_shape)[0]
        or tf.shape(t.indices)[0])
    shape_ = tf.cast(t.shape, tf.int64)
    return tf.SparseTensor(
        indices=tf.concat((tf.zeros([n_elem, 1], dtype=tf.int64),
                           t.indices),
                          axis=1),
        values=t.values,
        dense_shape=tf.concat(([1], shape_), axis=0))
  else:
    return tf.expand_dims(t, axis)


def slice_from(tensor, axis, start):
  """Slice a composite tensor along `axis` from `start`.

  Examples:

  ```python
  slice_from(tensor, 2, 1) === tensor[:, :, 1:]
  sparse_to_dense(slice_from(sparse_tensor, 2, 1))
    === sparse_to_dense(sparse_tensor)[:, :, 1:]
  ```

  Args:
    tensor: A `Tensor` or `SparseTensor`.
    axis: A python integer.
    start: A `0D` scalar.

  Returns:
    The sliced composite tensor.
  """
  if isinstance(tensor, tf.SparseTensor):
    if not tf.is_tensor(start) and start < 0:
      start = tensor.dense_shape[axis] + start
    all_but_first = tf.reshape(
        tf.where(tensor.indices[:, axis] >= start),
        [-1])
    indices = tf.gather(tensor.indices, all_but_first)
    indices = tf.unstack(indices, axis=1)
    indices = tf.stack(indices[:axis]
                       + [indices[axis] - start]
                       + indices[axis + 1:],
                       axis=1)
    new_shape = tf.unstack(tensor.dense_shape)
    new_shape[axis] = new_shape[axis] - start
    return tf.SparseTensor(
        indices=indices,
        values=tf.gather(tensor.values, all_but_first),
        dense_shape=tf.stack(new_shape))
  else:
    ndims = len(tensor.shape)
    if ndims is None:
      raise ValueError(
          'Unable to slice a tensor with unknown rank: {}'.format(tensor))
    slices = tuple([slice(None)] * axis
                   + [slice(start, None)]
                   + [slice(None)] * (ndims - axis - 1))
    return tensor[slices]


def slice_to(tensor, axis, end):
  """Slice a composite tensor along `axis` from 0 to `end`.

  Examples:

  ```python
  slice_to(tensor, 2, -1) === tensor[:, :, :-1]
  sparse_to_dense(slice_to(sparse_tensor, 2, -1))
    === sparse_to_dense(sparse_tensor)[:, :, :-1]
  ```

  Args:
    tensor: A `Tensor` or `SparseTensor`.
    axis: A python integer.
    end: A `0D` scalar.

  Returns:
    The sliced composite tensor.
  """
  if isinstance(tensor, tf.SparseTensor):
    if not tf.is_tensor(end) and end < 0:
      end = tensor.dense_shape[axis] + end
    all_but_first = tf.reshape(
        tf.where(tensor.indices[:, axis] < end),
        [-1])
    new_shape = tf.unstack(tensor.dense_shape)
    new_shape[axis] = end
    return tf.SparseTensor(
        indices=tf.gather(tensor.indices, all_but_first),
        values=tf.gather(tensor.values, all_but_first),
        dense_shape=tf.stack(new_shape))
  else:
    ndims = len(tensor.shape)
    if ndims is None:
      raise ValueError(
          'Unable to slice a tensor with unknown rank: {}'.format(tensor))
    slices = tuple([slice(None)] * axis
                   + [slice(None, end)]
                   + [slice(None)] * (ndims - axis - 1))
    return tensor[slices]
