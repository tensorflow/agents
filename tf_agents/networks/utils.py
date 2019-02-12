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

"""Network utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BatchSquash(object):
  """Facilitates flattening and unflattening batch dims of a tensor.

  Exposes a pair of matched faltten and unflatten methods. After flattening only
  1 batch dimension will be left. This facilitates evaluating networks that
  expect inputs to have only 1 batch dimension.
  """

  def __init__(self, batch_dims):
    """Create two tied ops to flatten and unflatten the front dimensions.

    Args:
      batch_dims: Number of batch dimensions the flatten/unflatten ops should
        handle.

    Raises:
      ValueError: if batch dims is negative.
    """
    if batch_dims < 0:
      raise ValueError('Batch dims must be non-negative.')
    self._batch_dims = batch_dims
    self._original_tensor_shape = None

  def flatten(self, tensor):
    """Flattens and caches the tensor's batch_dims."""
    with tf.name_scope('batch_flatten'):
      if self._batch_dims == 1:
        return tensor

      self._original_tensor_shape = tf.shape(input=tensor)

      if tensor.shape[self._batch_dims:].is_fully_defined():
        return tf.reshape(tensor,
                          [-1] + tensor.shape[self._batch_dims:].as_list())

      return tf.reshape(
          tensor,
          tf.concat([[-1], tf.shape(input=tensor)[self._batch_dims:]], axis=0),
      )

  def unflatten(self, tensor):
    """Unflattens the tensor's batch_dims using the cached shape."""
    with tf.name_scope('batch_unflatten'):
      if self._batch_dims == 1:
        return tensor

      if self._original_tensor_shape is None:
        raise ValueError('Please call flatten before unflatten.')

      # pyformat: disable
      return tf.reshape(
          tensor,
          tf.concat([
              self._original_tensor_shape[:self._batch_dims],
              tf.shape(input=tensor)[1:]], axis=0)
      )
      # pyformat: enable


def mlp_layers(conv_layer_params=None,
               fc_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               name=None):
  """Generates conv and fc layers to encode into a hidden state.

  Args:
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
        is used.
      name: Name for the mlp layers.

  Returns:
     List of mlp layers.
  """
  if not kernel_initializer:
    kernel_initializer = tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

  layers = []

  if conv_layer_params:
    layers.extend([
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation_fn,
            kernel_initializer=kernel_initializer,
            name='/'.join([name, 'conv2d']) if name else None)
        for (filters, kernel_size, strides) in conv_layer_params
    ])
  layers.append(tf.keras.layers.Flatten())

  if fc_layer_params:
    layers.extend([
        tf.keras.layers.Dense(
            num_units,
            activation=activation_fn,
            kernel_initializer=kernel_initializer,
            name='/'.join([name, 'dense']) if name else None)
        for num_units in fc_layer_params
    ])
  return layers
