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

"""Keras layer performing the equivalent of tf.expand_dims."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


class ExpandDims(tf.keras.layers.Layer):
  """Expands dims along a particular axis.

  Arguments:
      axis: Axis to expand.  A new dim is added before this axis.
         May be a negative value.  Must not be a tensor.

  Input shape:
      `(batch_size,) + shape`

  Output shape:
      `(batch_size,) + shape + [1]`, if `axis == -1`.

      `(batch_size,) + shape[:axis + 1] + [1] + shape[axis + 1:]`,
      if `axis < -1`.

      `(batch_size,) + shape[:axis] + [1] + shape[axis:]`, if `axis >= 0`.
  """

  def __init__(self, axis, **kwargs):
    super(ExpandDims, self).__init__(**kwargs)
    self.axis = axis

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if input_shape.rank is None:
      return input_shape
    input_shape = input_shape.as_list()
    if self.axis == -1:
      output_shape = input_shape + [1]
    elif self.axis < 0:
      output_shape = (
          input_shape[:self.axis + 1] + [1] + input_shape[self.axis + 1:])
    else:
      output_shape = input_shape[:self.axis] + [1] + input_shape[self.axis:]
    return tf.TensorShape(output_shape)

  def call(self, inputs):
    if self.axis < 0:
      # Negative axis, so expand starting from the right
      return tf.expand_dims(inputs, self.axis)
    else:
      # Perform the expansion from the left, but skip the batch dimension.
      return tf.expand_dims(inputs, self.axis + 1)

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(ExpandDims, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# Register with Keras so we can do type(layer).from_config(layer.get_config())
tf.keras.utils.get_custom_objects()['ExpandDims'] = ExpandDims
