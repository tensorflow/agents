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

# Lint as: python3
r"""Sequential Actor Network for PPO."""
import functools

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.networks import nest_map
from tf_agents.networks import sequential


def create_sequential_actor_net(fc_layer_units, action_tensor_spec):
  """Helper function for creating the actor network."""

  def create_dist(loc_and_scale):

    ndims = action_tensor_spec.shape.num_elements()
    return tfp.distributions.MultivariateNormalDiag(
        loc=loc_and_scale[..., :ndims],
        scale_diag=tf.math.softplus(loc_and_scale[..., ndims:]),
        validate_args=True)

  def means_layers():
    # TODO(b/179510447): align these parameters with Schulman 17.
    return tf.keras.layers.Dense(
        action_tensor_spec.shape.num_elements(),
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=0.1),
        name='means_projection_layer')

  def std_layers():
    # TODO(b/179510447): align these parameters with Schulman 17.
    std_kernel_initializer_scale = 0.1
    std_bias_initializer_value = np.log(np.exp(0.35) - 1)
    return tf.keras.layers.Dense(
        action_tensor_spec.shape.num_elements(),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=std_kernel_initializer_scale),
        bias_initializer=tf.keras.initializers.Constant(
            value=std_bias_initializer_value))

  dense = functools.partial(
      tf.keras.layers.Dense,
      activation=tf.nn.tanh,
      kernel_initializer=tf.keras.initializers.Orthogonal())

  return sequential.Sequential(
      [dense(num_units) for num_units in fc_layer_units] +
      [tf.keras.layers.Lambda(
          lambda x: {'loc': x, 'scale': x})] +
      [nest_map.NestMap({
          'loc': means_layers(),
          'scale': std_layers()
      })] +
      [nest_map.NestFlatten()] +
      # Concatenate the maen and standard deviation output to feed into the
      # distribution layer.
      [tf.keras.layers.Concatenate(axis=-1)] +
      # Create the output distribution from the mean and standard deviation.
      [tf.keras.layers.Lambda(create_dist)])
