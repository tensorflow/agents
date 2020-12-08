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

"""SquashedOuterWrapper Keras Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Mapping, Text

import numpy as np
import tensorflow as tf

from tf_agents.networks import utils

__all__ = ['SquashedOuterWrapper']


class SquashedOuterWrapper(tf.keras.layers.Layer):
  """Squash the outer dimensions of input tensors; unsquash outputs.

  This layer wraps a Keras layer `wrapped` that cannot handle more than one
  batch dimension.  It squashes inputs' outer dimensions to a single larger
  batch then unsquashes the outputs of `wrapped`.

  The outer dimensions are the leftmost `rank(inputs) - inner_rank` dimensions.

  Examples:

  ```python
  batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
  layer = SquashedOuterWrapper(wrapped=batch_norm, inner_rank=3)

  inputs_0 = tf.random.normal((B, H, W, C))
  # batch_norm sees tensor of shape [B, H, W, C]
  # outputs_1 shape is [B, H, W, C]
  outputs_0 = layer(inputs_0)

  inputs_1 = tf.random.normal((B, T, H, W, C))
  # batch_norm sees a tensor of shape [B * T, H, W, C]
  # outputs_1 shape is [B, T, H, W, C]
  outputs_1 = layer(inputs_1)

  inputs_2 = tf.random.normal((B1, B2, T, H, W, C))
  # batch_norm sees a tensor of shape [B1 * B2 * T, H, W, C]
  # outputs_2 shape is [B1, B2, T, H, W, C]
  outputs_2 = layer(inputs_2)
  ```
  """

  def __init__(self, wrapped: tf.keras.layers.Layer, inner_rank: int,
               **kwargs: Mapping[Text, Any]):
    """Initialize `SquashedOuterWrapper`.

    Args:
      wrapped: The keras layer to wrap.
      inner_rank: The inner rank of inputs that will be passed to the layer.
        This value allows us to infer the outer batch dimension regardless of
        the input shape to `build` or `call`.
      **kwargs: Additional arguments for keras layer construction.

    Raises:
      ValueError: If `wrapped` has method `get_initial_state`, because
        we do not know how to handle the case of multiple inputs and
        the presence of this method typically means an RNN or RNN-like
        layer which accepts separate state tensors.
    """
    if getattr(wrapped, 'get_initial_state', None) is not None:
      raise ValueError(
          '`wrapped` has method `get_initial_state`, which means its inputs '
          'will include separate state tensors.  This is not supported by '
          '`SquashedOuterWrapper`.  wrapped: {}'.format(wrapped))
    self._inner_rank = inner_rank
    self._wrapped = wrapped
    super(SquashedOuterWrapper, self).__init__(**kwargs)

  @property
  def inner_rank(self) -> int:
    return self._inner_rank

  @property
  def wrapped(self) -> tf.keras.layers.Layer:
    return self._wrapped

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if input_shape.rank is None:
      raise ValueError(
          'inputs must have known rank; input shape: {}'.format(input_shape))
    batch_shape = input_shape[:-self.inner_rank]
    inner_shape = input_shape[-self.inner_rank:]
    if batch_shape.is_fully_defined():
      squashed_shape = (int(np.prod(batch_shape)),) + inner_shape
    else:
      squashed_shape = (None,) + inner_shape
    self.wrapped.build(squashed_shape)
    self.built = True

  def call(self, inputs, training=False):
    static_rank = inputs.shape.rank
    if static_rank is None:
      raise ValueError(
          'inputs must have known rank; inputs: {}'.format(inputs))
    squash_dims = static_rank - self.inner_rank
    bs = utils.BatchSquash(squash_dims)
    squashed_inputs = bs.flatten(inputs)
    squashed_outputs = self.wrapped(squashed_inputs, training=training)
    return bs.unflatten(squashed_outputs)

  def get_config(self):
    config = {
        'inner_rank': self.inner_rank,
        'wrapped': {
            'class_name': self.wrapped.__class__.__name__,
            'config': self.wrapped.get_config()
        }
    }
    base_config = dict(super(SquashedOuterWrapper, self).get_config())
    base_config.update(config)
    return base_config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    wrapped = tf.keras.layers.deserialize(
        config.pop('wrapped'), custom_objects=custom_objects)
    return cls(wrapped, **config)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if input_shape.rank is None:
      raise ValueError(
          'inputs must have known rank; input shape: {}'.format(input_shape))
    batch_shape = input_shape[:-self.inner_rank]
    inner_shape = input_shape[-self.inner_rank:]
    if batch_shape.is_fully_defined():
      squashed_shape = (int(np.prod(batch_shape)),) + inner_shape
    else:
      squashed_shape = (None,) + inner_shape
    squashed_output_shape = self.wrapped.compute_output_shape(squashed_shape)
    return batch_shape + squashed_output_shape[1:]

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    return self.wrapped.trainable_weights

  @property
  def non_trainable_weights(self):
    if not self.trainable:
      return self.wrapped.weights
    return self.wrapped.non_trainable_weights

  @property
  def losses(self):
    layer_losses = super(SquashedOuterWrapper, self).losses
    return self.wrapped.losses + layer_losses

  @property
  def updates(self):
    updates = self.wrapped.updates
    return updates + self._updates
