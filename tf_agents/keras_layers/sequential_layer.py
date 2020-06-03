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

"""Keras layer to replace the Sequential Model object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


class SequentialLayer(tf.keras.layers.Layer):
  """The SequentialLayer represents a sequence of Keras layers.

  It is a Keras Layer that can be used instead of tf.keras.layers.Sequential,
  which is actually a Keras Model.  In contrast to keras Sequential, this
  layer can be used as a pure Layer in tf.functions and when exporting
  SavedModels, without having to pre-declare input and output shapes.  In turn,
  this layer is usable as a preprocessing layer for TF Agents Networks, and
  can be exported via PolicySaver.

  Usage:
  ```python
  c = SequentialLayer([layer1, layer2, layer3])
  output = c(inputs)    # Equivalent to: output = layer3(layer2(layer1(inputs)))
  ```
  """

  def __init__(self, layers, **kwargs):
    """Create a composition.

    Args:
      layers: A list or tuple of layers to compose.
      **kwargs: Arguments to pass to `Keras` layer initializer, including
        `name`.

    Raises:
      TypeError: If any of the layers are not instances of keras `Layer`.
    """
    for layer in layers:
      if not isinstance(layer, tf.keras.layers.Layer):
        raise TypeError(
            "Expected all layers to be instances of keras Layer, but saw: '{}'"
            .format(layer))

    super(SequentialLayer, self).__init__(**kwargs)
    self.layers = copy.copy(layers)

  def compute_output_shape(self, input_shape):
    output_shape = tf.TensorShape(input_shape)
    for l in self.layers:
      output_shape = l.compute_output_shape(output_shape)
    return tf.TensorShape(output_shape)

  def compute_output_signature(self, input_signature):
    output_signature = input_signature
    for l in self.layers:
      output_signature = l.compute_output_signature(output_signature)
    return output_signature

  def build(self, input_shape=None):
    for l in self.layers:
      l.build(input_shape)
      input_shape = l.compute_output_shape(input_shape)
    self.built = True

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    weights = {}
    for l in self.layers:
      for v in l.trainable_weights:
        weights[id(v)] = v
    return list(weights.values())

  @property
  def non_trainable_weights(self):
    weights = {}
    for l in self.layers:
      for v in l.non_trainable_weights:
        weights[id(v)] = v
    return list(weights.values())

  @property
  def trainable(self):
    return all([l.trainable for l in self.layers])

  @trainable.setter
  def trainable(self, value):
    for l in self.layers:
      l.trainable = value

  @property
  def losses(self):
    values = set()
    for l in self.layers:
      values.update(l.losses)
    return list(values)

  @property
  def regularizers(self):
    values = set()
    for l in self.layers:
      values.update(l.regularizers)
    return list(values)

  def call(self, inputs, training=False):
    outputs = inputs
    for l in self.layers:
      outputs = l(outputs, training=training)
    return outputs

  def get_config(self):
    config = {}
    for i, layer in enumerate(self.layers):
      config[i] = {
          'class_name': layer.__class__.__name__,
          'config': copy.deepcopy(layer.get_config())
      }
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    layers = [
        tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
        for conf in config.values()
    ]
    return cls(layers)

# Register with Keras so we can do type(layer).from_config(layer.get_config())
tf.keras.utils.get_custom_objects()['SequentialLayer'] = SequentialLayer
