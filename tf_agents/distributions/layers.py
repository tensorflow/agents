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

"""Layers for generating distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import utils


def default_fully_connected(inputs, num_elements, scope=None):
  return tf.contrib.layers.fully_connected(
      inputs, num_elements,
      biases_initializer=tf.zeros_initializer(),
      weights_initializer=tf.random_uniform_initializer(
          minval=-0.003, maxval=0.003),
      activation_fn=None,
      normalizer_fn=None,
      scope=scope)


def tanh_squash_to_spec(inputs, output_spec):
  output_means = (
      output_spec.maximum + output_spec.minimum) / 2.0
  output_magnitudes = (
      output_spec.maximum - output_spec.minimum) / 2.0

  return output_means + output_magnitudes * tf.tanh(inputs)


def passthrough(inputs, *args):
  del args
  return inputs


def factored_categorical(inputs, output_spec, outer_rank=1,
                         projection_layer=default_fully_connected):
  """Project a batch of inputs to a categorical distribution.

  Given an output spec for a single tensor discrete action, produces a
  neural net layer converting inputs to a categorical distribution
  matching the spec. The logits are derived from a fully connected linear
  layer. Each discrete action (each element of the output tensor) is sampled
  independently.

  Args:
    inputs: An input Tensor of shape [batch_size, ?].
    output_spec: An output spec (either BoundedArraySpec or BoundedTensorSpec).
    outer_rank: The number of outer dimensions of inputs to consider batch
      dimensions and to treat as batch dimensions of output distribution.
    projection_layer: Function taking in inputs, num_elements, scope and
      returning a projection of inputs to a Tensor of width num_elements.

  Returns:
    A tf.distribution.Categorical object.

  Raises:
    ValueError: If output_spec contains multiple distinct ranges or is otherwise
      invalid.
  """
  if not output_spec.is_bounded():
    raise ValueError('Input output_spec is of invalid type '
                     '%s.' % type(output_spec))
  if not output_spec.is_discrete():
    raise ValueError('Output is not discrete.')
  num_outputs = np.unique(output_spec.maximum - output_spec.minimum + 1)
  num_ranges = len(num_outputs)
  if num_ranges > 1 or np.any(num_outputs <= 0):
    raise ValueError('Single discrete output has invalid ranges: '
                     '%s' % num_outputs)
  output_shape = output_spec.shape.concatenate([num_outputs])
  batch_squash = utils.BatchSquash(outer_rank)
  inputs = batch_squash.flatten(inputs)
  logits = projection_layer(
      inputs, output_shape.num_elements(), scope='logits')
  logits = tf.reshape(logits, [-1] + output_shape.as_list())
  logits = batch_squash.unflatten(logits)
  return tfp.distributions.Categorical(logits, dtype=output_spec.dtype)


def normal(inputs,
           output_spec,
           outer_rank=1,
           projection_layer=default_fully_connected,
           mean_transform=tanh_squash_to_spec,
           std_initializer=tf.zeros_initializer(),
           std_transform=tf.exp,
           distribution_cls=tfp.distributions.Normal):
  """Project a batch of inputs to a batch of means and standard deviations.

  Given an output spec for a single tensor continuous action, produces a
  neural net layer converting inputs to a normal distribution matching
  the spec.  The mean is derived from a fully connected linear layer as
  mean_transform(layer_output, output_spec).  The std is fixed to a single
  trainable tensor (thus independent of the inputs).  Specifically, std is
  parameterized as std_transform(variable).

  Args:
    inputs: An input Tensor of shape [batch_size, ?].
    output_spec: An output spec (either BoundedArraySpec or BoundedTensorSpec).
    outer_rank: The number of outer dimensions of inputs to consider batch
      dimensions and to treat as batch dimensions of output distribution.
    projection_layer: Function taking in inputs, num_elements, scope and
      returning a projection of inputs to a Tensor of width num_elements.
    mean_transform: A function taking in layer output and the output_spec,
      returning the means.  Defaults to tanh_squash_to_spec.
    std_initializer: Initializer for std_dev variables.
    std_transform: The function applied to the trainable std variable. For
      example, tf.exp (default), tf.nn.softplus.
    distribution_cls: The distribution class to use for output distribution.
      Default is tfp.distributions.Normal.

  Returns:
    A tf.distribution.Normal object in which the standard deviation is not
      dependent on input.

  Raises:
    ValueError: If output_spec is invalid.
  """
  if not output_spec.is_bounded():
    raise ValueError('Input output_spec is of invalid type '
                     '%s.' % type(output_spec))
  if not output_spec.is_continuous():
    raise ValueError('Output is not continuous.')

  batch_squash = utils.BatchSquash(outer_rank)
  inputs = batch_squash.flatten(inputs)
  means = projection_layer(
      inputs, output_spec.shape.num_elements(), scope='means')
  stds = tf.contrib.layers.bias_add(
      tf.zeros_like(means),  # Independent of inputs.
      initializer=std_initializer,
      scope='stds',
      activation_fn=None)

  means = tf.reshape(means, [-1] + output_spec.shape.as_list())
  means = mean_transform(means, output_spec)
  means = tf.cast(means, output_spec.dtype)

  stds = tf.reshape(stds, [-1] + output_spec.shape.as_list())
  stds = std_transform(stds)
  stds = tf.cast(stds, output_spec.dtype)

  means, stds = batch_squash.unflatten(means), batch_squash.unflatten(stds)
  return distribution_cls(means, stds)
