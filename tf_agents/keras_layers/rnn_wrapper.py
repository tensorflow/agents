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

"""Wrapper for tf.keras.layers.RNN subclasses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = ['RNNWrapper']


class RNNWrapper(tf.keras.layers.Layer):
  """Wraps a Keras RNN/LSTM/GRU layer to make network state more consistent."""

  def __init__(self, layer: tf.keras.layers.RNN, **kwargs):
    """Create a `RNNWrapper`.

    Args:
      layer: An instance of `tf.keras.layers.RNN` or subclasses (including
        `tf.keras.layers.{LSTM,GRU,...}`.
      **kwargs: Extra args to `Layer` parent class.

    Raises:
      TypeError: If `layer` is not a subclass of `tf.keras.layers.RNN`.
      NotImplementedError: If `layer` was created with `return_state == False`.
      NotImplementederror: If `layer` was created with
        `return_sequences == False`.
    """
    if not isinstance(layer, tf.keras.layers.RNN):
      raise TypeError(
          'layer is not a subclass of tf.keras.layers.RNN.  Layer: {}'.format(
              layer))
    layer_config = layer.get_config()
    if not layer_config.get('return_state', False):
      # This is an RNN layer that doesn't return state.
      raise NotImplementedError(
          'Provided a Keras RNN layer with return_state==False. '
          'This configuration is not supported.  Layer: {}'.format(layer))
    if not layer_config.get('return_sequences', False):
      raise NotImplementedError(
          'Provided a Keras RNN layer with return_sequences==False. '
          'This configuration is not supported.  Layer: {}'.format(layer))

    self._layer = layer
    super(RNNWrapper, self).__init__(**kwargs)

  @property
  def dtype(self) -> tf.DType:
    return self._layer.dtype

  @property
  def cell(self) -> tf.keras.layers.Layer:
    """Return the `cell` underlying the RNN layer."""
    return self._layer.cell

  @property
  def state_size(self):
    """Return the `state_size` of the cell underlying the RNN layer."""
    return self._layer.cell.state_size

  @property
  def wrapped_layer(self) -> tf.keras.layers.RNN:
    """Return the wrapped RNN layer."""
    return self._layer

  def get_config(self):
    config = {
        'layer': {
            'class_name': self._layer.__class__.__name__,
            'config': self._layer.get_config()
        }
    }
    base_config = dict(super(RNNWrapper, self).get_config())
    base_config.update(config)
    return base_config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    internal_layer = tf.keras.layers.deserialize(
        config.pop('layer'), custom_objects=custom_objects)
    layer = cls(internal_layer, **config)
    return layer

  def compute_output_shape(self, input_shape):
    return self._layer.compute_output_shape(input_shape)

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    return self._layer.trainable_weights

  @property
  def non_trainable_weights(self):
    if not self.trainable:
      return self._layer.weights
    return self._layer.non_trainable_weights

  @property
  def losses(self):
    layer_losses = super(RNNWrapper, self).losses
    return self._layer.losses + layer_losses

  @property
  def updates(self):
    updates = self._layer.updates
    return updates + self._updates

  def build(self, input_shape):
    if len(input_shape) <= 2:
      input_shape = (input_shape[0],) + (None,) + input_shape[1:]
    self._layer.build(input_shape)
    self.built = True

  def get_initial_state(self, inputs=None):
    inputs_flat = [
        tf.convert_to_tensor(x, name='input', dtype_hint=self.dtype)
        for x in tf.nest.flatten(inputs)
    ]
    has_time_axis = all(
        [x.shape.ndims is None or x.shape.ndims > 2 for x in inputs_flat])
    if not has_time_axis:
      inputs_flat = [tf.expand_dims(t, axis=1) for t in inputs_flat]
    inputs = tf.nest.pack_sequence_as(inputs, inputs_flat)
    return self._layer.get_initial_state(inputs)

  def call(self, inputs, initial_state=None, mask=None, training=False):
    """Perform the computation.

    Args:
      inputs: A tuple containing tensors in batch-major format, each shaped
        `[batch_size, n, ...]`.
      initial_state: (Optional) An initial state for the wrapped layer. If not
        provided, `get_initial_state()` is used instead.
      mask: The mask to pass down to the wrapped layer.
      training: Whether the output is being used for training.

    Returns:
      A 2-tuple `(outputs, final_state)` where:

       - `outputs` contains the outputs for all time steps of the unroll; this
         is typically a tensor shaped `[batch_size, n, ...]`.
       - `final_state` contains the final state.
    """
    inputs_flat = [
        tf.convert_to_tensor(x, name='input', dtype_hint=self.dtype)
        for x in tf.nest.flatten(inputs)
    ]
    has_time_axis = all(
        [x.shape.ndims is None or x.shape.ndims > 2 for x in inputs_flat])
    if not has_time_axis:
      inputs_flat = [tf.expand_dims(t, axis=1) for t in inputs_flat]
    inputs = tf.nest.pack_sequence_as(inputs, inputs_flat)

    # TODO(b/158804957): tf.function changes "if tensor:" to tensor bool expr.
    # pylint: disable=literal-comparison
    if initial_state is None or initial_state is () or initial_state is []:
      initial_state = self._layer.get_initial_state(inputs)
    # pylint: enable=literal-comparison

    outputs = self._layer(
        inputs, initial_state=initial_state, mask=mask, training=training)

    output, new_state = outputs[0], outputs[1:]

    # Keras RNN's outputs[1:] does not match the nest structure of its cells'
    # state_size property.  Restructure the output state to match.
    new_state = tf.nest.pack_sequence_as(
        self.state_size, tf.nest.flatten(new_state))

    if not has_time_axis:
      output = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=1), output)

    # Outputs are in output, and state is in outputs[1:]
    return output, new_state


# Register with Keras so we can do type(layer).from_config(layer.get_config())
tf.keras.utils.get_custom_objects()['RNNWrapper'] = RNNWrapper
