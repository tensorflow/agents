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
# Using Type Annotations.
from __future__ import print_function

import typing

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.typing import types
from tf_agents.utils import composite


def check_single_floating_network_output(
    output_spec: types.NestedSpec,
    expected_output_shape: typing.Tuple[int, ...],
    label: typing.Text):
  expected_output_shape = tuple(int(x) for x in expected_output_shape)
  if not (isinstance(output_spec, tf.TensorSpec)
          and output_spec.shape == expected_output_shape
          and output_spec.dtype.is_floating):
    raise ValueError(
        'Expected {} to emit a floating point tensor with inner dims '
        '{}; but saw network output spec: {}'
        .format(label, expected_output_shape, output_spec))


def maybe_permanent_dropout(rate, noise_shape=None, seed=None, permanent=False):
  """Adds a Keras dropout layer with the option of applying it at inference.

  Args:
    rate: the probability of dropping an input.
    noise_shape: 1D integer tensor representing the dropout mask multiplied to
      the input.
    seed: A Python integer to use as random seed.
    permanent: If set, applies dropout during inference and not only during
      training. This flag is used for approximated Bayesian inference.
  Returns:
    A function adding a dropout layer according to the parameters for the given
      input.
  """
  if permanent:
    def _keras_dropout(x):
      return tf.nn.dropout(
          x, rate=rate, noise_shape=noise_shape, seed=seed)
    return tf.keras.layers.Lambda(_keras_dropout)
  return tf.keras.layers.Dropout(rate, noise_shape, seed)


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

      self._original_tensor_shape = composite.shape(tensor)

      if tensor.shape[self._batch_dims:].is_fully_defined():
        return composite.reshape(
            tensor, [-1] + tensor.shape[self._batch_dims:].as_list())

      reshaped = composite.reshape(
          tensor,
          tf.concat([[-1], composite.shape(tensor)[self._batch_dims:]], axis=0),
      )
      # If the batch dimensions are all defined but the rest are undefined,
      # `reshaped` will have None as the first squashed dim since we are calling
      # tf.shape above. Since we know how many batch_dims we have, we can check
      # if all the elements we want to squash are defined, allowing us to
      # call ensure_shape to set the shape of the squashed dim. Note that this
      # is only implemented for tf.Tensor and not SparseTensors.
      if (isinstance(tensor, tf.Tensor) and
          tensor.shape[:self._batch_dims].is_fully_defined()):
        return tf.ensure_shape(
            reshaped,
            [np.prod(tensor.shape[:self._batch_dims], dtype=np.int64)] +
            tensor.shape[self._batch_dims:])
      return reshaped

  def unflatten(self, tensor):
    """Unflattens the tensor's batch_dims using the cached shape."""
    with tf.name_scope('batch_unflatten'):
      if self._batch_dims == 1:
        return tensor

      if self._original_tensor_shape is None:
        raise ValueError('Please call flatten before unflatten.')

      # pyformat: disable
      return composite.reshape(
          tensor,
          tf.concat([
              self._original_tensor_shape[:self._batch_dims],
              composite.shape(tensor)[1:]], axis=0)
      )
      # pyformat: enable


def mlp_layers(conv_layer_params=None,
               fc_layer_params=None,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               weight_decay_params=None,
               name=None):
  """Generates conv and fc layers to encode into a hidden state.

  Args:
    conv_layer_params: Optional list of convolution layers parameters, where
      each item is a length-three tuple indicating (filters, kernel_size,
      stride).
    fc_layer_params: Optional list of fully_connected parameters, where each
      item is the number of units in the layer.
    dropout_layer_params: Optional list of dropout layer parameters, each item
      is the fraction of input units to drop or a dictionary of parameters
      according to the keras.Dropout documentation. The additional parameter
      `permanent`, if set to True, allows to apply dropout at inference for
      approximated Bayesian inference. The dropout layers are interleaved with
      the fully connected layers; there is a dropout layer after each fully
      connected layer, except if the entry in the list is None. This list must
      have the same length of fc_layer_params, or be None.
    activation_fn: Activation function, e.g. tf.keras.activations.relu,.
    kernel_initializer: Initializer to use for the kernels of the conv and
      dense layers. If none is provided a default variance_scaling_initializer
      is used.
    weight_decay_params: Optional list of weight decay params for the fully
      connected layer.
    name: Name for the mlp layers.

  Returns:
    List of mlp layers.

  Raises:
    ValueError: If the number of dropout layer parameters does not match the
      number of fully connected layer parameters.
  """
  if kernel_initializer is None:
    kernel_initializer = tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal')

  layers = []

  if conv_layer_params is not None:
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

  if fc_layer_params is not None:
    if dropout_layer_params is None:
      dropout_layer_params = [None] * len(fc_layer_params)
    else:
      if len(dropout_layer_params) != len(fc_layer_params):
        raise ValueError('Dropout and full connected layer parameter lists have'
                         ' different lengths (%d vs. %d.)' %
                         (len(dropout_layer_params), len(fc_layer_params)))

    if weight_decay_params is None:
      weight_decay_params = [None] * len(fc_layer_params)
    else:
      if len(weight_decay_params) != len(fc_layer_params):
        raise ValueError('Weight decay and fully connected layer parameter '
                         'lists have different lengths (%d vs. %d.)' %
                         (len(weight_decay_params), len(fc_layer_params)))

    for num_units, dropout_params, weight_decay in zip(
        fc_layer_params, dropout_layer_params, weight_decay_params):
      kernel_regularizer = None
      if weight_decay is not None:
        kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
      layers.append(tf.keras.layers.Dense(
          num_units,
          activation=activation_fn,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          name='/'.join([name, 'dense']) if name else None))
      if not isinstance(dropout_params, dict):
        dropout_params = {'rate': dropout_params} if dropout_params else None

      if dropout_params is not None:
        layers.append(maybe_permanent_dropout(**dropout_params))

  return layers
