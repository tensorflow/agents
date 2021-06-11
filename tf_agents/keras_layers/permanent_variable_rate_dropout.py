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

"""A keras layer that applies dropout both in training and serving.

Add the possibility to apply a variable dropout rate, that is, the rate
parameter can be a callable.
"""

import tensorflow as tf


class PermanentVariableRateDropout(tf.keras.layers.Dropout):
  """Applies dropout both in training and serving, with variable dropout rate.

  Initialize this layer the same was as `keras.layers.Dropout`, with two notable
  differences:
  --The parameter `rate` can also be a callable.
  --The extra boolean parameter `permanent`. If set to true, dropout will be
    applied both in training and inference.
  """

  def __init__(self, rate, permanent=False, **kwargs):
    self._permanent = permanent
    super(PermanentVariableRateDropout, self).__init__(rate, **kwargs)

  def call(self, inputs, training=None):
    # If permanent, ignore training, we are keeping dropout.
    if self._permanent:
      training = True
    if training is None:
      training = tf.keras.backend.learning_phase()

    if training:
      rate = self._get_dropout_value()
      outputs = tf.nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=rate)
      return outputs
    else:
      return inputs

  def _get_dropout_value(self):
    if callable(self.rate):
      return self.rate()
    return self.rate
