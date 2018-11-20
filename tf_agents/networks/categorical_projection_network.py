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

"""Project inputs to a categorical distribution object."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils

import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class CategoricalProjectionNetwork(network.Network):
  """Generates a tfp.distribution.Categorical by predicting logits."""

  def __init__(self,
               output_spec,
               logits_init_output_factor=0.1,
               name='CategoricalProjectionNetwork'):
    """Creates an instance of CategoricalProjectionNetwork.

    Args:
      output_spec: An output spec (either BoundedArraySpec or
        BoundedTensorSpec).
      logits_init_output_factor: Output factor for initializing kernel logits
        weights.
      name: A string representing name of the network.
    """
    super(CategoricalProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        observation_spec=None,
        action_spec=None,
        state_spec=(),
        name=name)

    if not output_spec.is_bounded():
      raise ValueError(
          'output_spec must be bounded. Got: %s.' % type(output_spec))

    if not output_spec.is_discrete():
      raise ValueError(
          'output_spec must be discrete. Got: %s.' % output_spec)

    unique_num_actions = np.unique(output_spec.maximum - output_spec.minimum +
                                   1)

    if len(unique_num_actions) > 1:
      raise ValueError(
          'Projection Network requires num_actions to be equal '
          'across action dimentions. Implement a more general categorical '
          'projection if you need more flexibility.')

    self._output_spec = output_spec
    self._output_shape = output_spec.shape.concatenate([unique_num_actions])

    self._projection_layer = tf.keras.layers.Dense(
        self._output_shape.num_elements(),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=logits_init_output_factor),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits')

  def call(self, inputs, outer_rank):
    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)

    logits = self._projection_layer(inputs)
    logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
    logits = batch_squash.unflatten(logits)

    return tfp.distributions.Categorical(logits, dtype=self._output_spec.dtype)
