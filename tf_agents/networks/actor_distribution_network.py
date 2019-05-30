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

"""Sample Keras actor network that generates distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value,
      scale_distribution=False)


@gin.configurable
class ActorDistributionNetwork(network.DistributionNetwork):
  """Creates an actor producing either Normal or Categorical distribution.

  Note: By default, this network uses `NormalProjectionNetwork` for continuous
  projection which by default uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               fc_layer_params=(200, 100),
               dropout_layer_params=None,
               conv_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               name='ActorDistributionNetwork'):
    """Creates an instance of `ActorDistributionNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      fc_layer_params: Optional list of fully_connected parameters, where each
        item is the number of units in the layer.
      dropout_layer_params: Optional list of dropout layer parameters, each item
        is the fraction of input units to drop or a dictionary of parameters
        according to the keras.Dropout documentation. The additional parameter
        `permanent', if set to True, allows to apply dropout at inference for
        approximated Bayesian inference. The dropout layers are interleaved with
        the fully connected layers; there is a dropout layer after each fully
        connected layer, except if the entry in the list is None. This list must
        have the same length of fc_layer_params, or be None.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      discrete_projection_net: Callable that generates a discrete projection
        network to be called with some hidden state and the outer_rank of the
        state.
      continuous_projection_net: Callable that generates a continuous projection
        network to be called with some hidden state and the outer_rank of the
        state.
      name: A string representing name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation.
    """

    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    mlp_layers = utils.mlp_layers(
        conv_layer_params,
        fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
        dropout_layer_params=dropout_layer_params,
        name='input_mlp')

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(ActorDistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._mlp_layers = mlp_layers
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observations, step_type, network_state):
    del step_type  # unused.
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    observations = tf.nest.flatten(observations)
    states = tf.cast(observations[0], tf.float32)

    # Reshape to only a single batch dimension for neural network functions.
    batch_squash = utils.BatchSquash(outer_rank)
    states = batch_squash.flatten(states)

    for layer in self._mlp_layers:
      states = layer(states)

    # TODO(oars): Can we avoid unflattening to flatten again
    states = batch_squash.unflatten(states)
    output_actions = tf.nest.map_structure(
        lambda proj_net: proj_net(states, outer_rank),
        self._projection_networks)
    return output_actions, network_state
