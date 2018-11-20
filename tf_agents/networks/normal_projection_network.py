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

"""Project inputs to a normal distribution object."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import bias_layer
from tf_agents.networks import network
from tf_agents.networks import utils

import gin.tf

nest = tf.contrib.framework.nest


def tanh_squash_to_spec(inputs, spec):
  means = (spec.maximum + spec.minimum) / 2.0
  magnitudes = (spec.maximum - spec.minimum) / 2.0

  return means + magnitudes * tf.tanh(inputs)


@gin.configurable
class NormalProjectionNetwork(network.Network):
  """Generates a tfp.distribution.Normal by predicting a mean and std.

  Note: the standard deviations are independent of the input.
  """

  def __init__(self,
               output_spec,
               activation_fn=None,
               init_means_output_factor=0.1,
               std_initializer_value=0.0,
               mean_transform=tanh_squash_to_spec,
               std_transform=tf.nn.softplus,
               name='NormalProjectionNetwork'):
    """Creates an instance of NormalProjectionNetwork.

    Args:
      output_spec: An output spec (either BoundedArraySpec or
        BoundedTensorSpec).
      activation_fn: Activation function to use in dense layer.
      init_means_output_factor: Output factor for initializing action means
        weights.
      std_initializer_value: Initial value for std variables.
      mean_transform: Transform to apply to the calculated means
      std_transform: Transform to apply to the stddevs.
      name: A string representing name of the network.
    """
    super(NormalProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        observation_spec=None,
        action_spec=None,
        state_spec=(),
        name=name)

    self._output_spec = output_spec
    self._mean_transform = mean_transform
    self._std_transform = std_transform

    self._projection_layer = tf.keras.layers.Dense(
        output_spec.shape.num_elements(),
        activation=activation_fn,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=init_means_output_factor),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='normal_projection_layer')

    self._bias = bias_layer.BiasLayer(
        bias_initializer=tf.keras.initializers.Constant(
            value=std_initializer_value))

  def call(self, inputs, outer_rank):
    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)

    means = self._projection_layer(inputs)
    means = tf.reshape(means, [-1] + self._output_spec.shape.as_list())
    means = self._mean_transform(means, self._output_spec)
    means = tf.cast(means, self._output_spec.dtype)

    stds = self._bias(tf.zeros_like(means))
    stds = tf.reshape(stds, [-1] + self._output_spec.shape.as_list())
    stds = self._std_transform(stds)
    stds = tf.cast(stds, self._output_spec.dtype)

    means = batch_squash.unflatten(means)
    stds = batch_squash.unflatten(stds)

    return tfp.distributions.Normal(means, stds)
