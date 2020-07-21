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

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec


@gin.configurable
class CategoricalProjectionNetwork(network.DistributionNetwork):
  """Generates a tfp.distribution.Categorical by predicting logits."""

  def __init__(self,
               sample_spec,
               logits_init_output_factor=0.1,
               name='CategoricalProjectionNetwork'):
    """Creates an instance of CategoricalProjectionNetwork.

    Args:
      sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape and
        dtypes of samples pulled from the output distribution.
      logits_init_output_factor: Output factor for initializing kernel logits
        weights.
      name: A string representing name of the network.
    """
    unique_num_actions = np.unique(sample_spec.maximum - sample_spec.minimum +
                                   1)
    if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
      raise ValueError('Bounds on discrete actions must be the same for all '
                       'dimensions and have at least 1 action. Projection '
                       'Network requires num_actions to be equal across '
                       'action dimensions. Implement a more general '
                       'categorical projection if you need more flexibility.')

    output_shape = sample_spec.shape.concatenate([int(unique_num_actions)])
    output_spec = self._output_distribution_spec(output_shape, sample_spec,
                                                 name)

    super(CategoricalProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        input_tensor_spec=None,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    if not tensor_spec.is_bounded(sample_spec):
      raise ValueError(
          'sample_spec must be bounded. Got: %s.' % type(sample_spec))

    if not tensor_spec.is_discrete(sample_spec):
      raise ValueError('sample_spec must be discrete. Got: %s.' % sample_spec)

    self._sample_spec = sample_spec
    self._output_shape = output_shape

    self._projection_layer = tf.keras.layers.Dense(
        self._output_shape.num_elements(),
        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=logits_init_output_factor),
        bias_initializer=tf.keras.initializers.Zeros(),
        name='logits')

  def _output_distribution_spec(self, output_shape, sample_spec, network_name):
    input_param_spec = {
        'logits':
            tensor_spec.TensorSpec(
                shape=output_shape,
                dtype=tf.float32,
                name=network_name + '_logits')
    }

    return distribution_spec.DistributionSpec(
        tfp.distributions.Categorical,
        input_param_spec,
        sample_spec=sample_spec,
        dtype=sample_spec.dtype)

  def call(self, inputs, outer_rank, training=False, mask=None):
    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)
    inputs = tf.cast(inputs, tf.float32)

    logits = self._projection_layer(inputs, training=training)
    logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
    logits = batch_squash.unflatten(logits)

    if mask is not None:
      # If the action spec says each action should be shaped (1,), add another
      # dimension so the final shape is (B, 1, A), where A is the number of
      # actions. This will make Categorical emit events shaped (B, 1) rather
      # than (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
      if mask.shape.rank < logits.shape.rank:
        mask = tf.expand_dims(mask, -2)

      # Overwrite the logits for invalid actions to a very large negative
      # number. We do not use -inf because it produces NaNs in many tfp
      # functions.
      almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
      logits = tf.compat.v2.where(
          tf.cast(mask, tf.bool), logits, almost_neg_inf)

    return self.output_spec.build_distribution(logits=logits), ()
