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

"""Project inputs to a tanh-squashed MultivariateNormalDiag distribution.

This network reproduces Soft Actor-Critic refererence implementation in:
https://github.com/rail-berkeley/softlearning/
"""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Callable, Optional, Text

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec

from tf_agents.typing import types


def tanh_squash_to_spec(inputs: types.Tensor,
                        spec: types.TensorSpec) -> types.Tensor:
  """Maps inputs with arbitrary range to range defined by spec using `tanh`."""
  means = (spec.maximum + spec.minimum) / 2.0
  magnitudes = (spec.maximum - spec.minimum) / 2.0

  return means + magnitudes * tf.tanh(inputs)


@gin.configurable
class TanhNormalProjectionNetwork(network.DistributionNetwork):
  """Generates a tanh-squashed MultivariateNormalDiag distribution.

  Note: This network uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               sample_spec: types.TensorSpec,
               activation_fn: Optional[Callable[[types.Tensor],
                                                types.Tensor]] = None,
               std_transform: Optional[Callable[[types.Tensor],
                                                types.Tensor]] = tf.exp,
               name: Text = 'TanhNormalProjectionNetwork'):
    """Creates an instance of TanhNormalProjectionNetwork.

    Args:
      sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape and
        dtypes of samples pulled from the output distribution.
      activation_fn: Activation function to use in dense layer.
      std_transform: Transformation function to apply to the stddevs.
      name: A string representing name of the network.
    """
    if len(tf.nest.flatten(sample_spec)) != 1:
      raise ValueError('Tanh Normal Projection network only supports single'
                       ' spec samples.')
    output_spec = self._output_distribution_spec(sample_spec, name)
    super(TanhNormalProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        input_tensor_spec=None,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._sample_spec = sample_spec
    self._std_transform = std_transform

    self._projection_layer = tf.keras.layers.Dense(
        sample_spec.shape.num_elements() * 2,
        activation=activation_fn,
        name='projection_layer')

  def _output_distribution_spec(self, sample_spec, network_name):
    input_param_shapes = {
        'loc': sample_spec.shape,
        'scale_diag': sample_spec.shape
    }
    input_param_spec = {
        name: tensor_spec.TensorSpec(  # pylint: disable=g-complex-comprehension
            shape=shape,
            dtype=sample_spec.dtype,
            name=network_name + '_' + name)
        for name, shape in input_param_shapes.items()
    }

    def distribution_builder(*args, **kwargs):
      distribution = tfp.distributions.MultivariateNormalDiag(*args, **kwargs)
      return distribution_utils.scale_distribution_to_spec(
          distribution, sample_spec)

    return distribution_spec.DistributionSpec(
        distribution_builder, input_param_spec, sample_spec=sample_spec)

  def call(self,
           inputs: types.NestedTensor,
           outer_rank: int,
           training: bool = False,
           mask: Optional[types.NestedTensor] = None) -> types.NestedTensor:
    if inputs.dtype != self._sample_spec.dtype:  # pytype: disable=attribute-error
      raise ValueError('Inputs to TanhNormalProjectionNetwork must match the '
                       'sample_spec.dtype.')

    if mask is not None:
      raise NotImplementedError(
          'TanhNormalProjectionNetwork does not yet implement action masking; '
          'got mask={}'.format(mask))

    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = network_utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)

    means_and_stds = self._projection_layer(inputs, training=training)
    means, stds = tf.split(means_and_stds, num_or_size_splits=2, axis=-1)
    means = tf.reshape(means, [-1] + self._sample_spec.shape.as_list())
    means = tf.cast(means, self._sample_spec.dtype)

    if self._std_transform is not None:
      stds = self._std_transform(stds)
    stds = tf.cast(stds, self._sample_spec.dtype)

    means = batch_squash.unflatten(means)
    stds = batch_squash.unflatten(stds)

    return self.output_spec.build_distribution(loc=means, scale_diag=stds), ()  # pytype: disable=bad-return-type
