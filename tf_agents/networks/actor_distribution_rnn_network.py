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

"""Sample Keras actor network  with LSTM cells that generates distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
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
      std_bias_initializer_value=std_bias_initializer_value)


@gin.configurable
class ActorDistributionRnnNetwork(network.DistributionNetwork):
  """Creates an actor producing either Normal or Categorical distribution.

  Note: By default, this network uses `NormalProjectionNetwork` for continuous
  projection which by default uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(200, 100),
               input_dropout_layer_params=None,
               lstm_size=None,
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               rnn_construction_fn=None,
               rnn_construction_kwargs={},
               name='ActorDistributionRnnNetwork'):
    """Creates an instance of `ActorDistributionRnnNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent`, if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of input_fc_layer_params, or
        be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      dtype: The dtype to use by the convolution and fully connected layers.
      discrete_projection_net: Callable that generates a discrete projection
        network to be called with some hidden state and the outer_rank of the
        state.
      continuous_projection_net: Callable that generates a continuous projection
        network to be called with some hidden state and the outer_rank of the
        state.
      rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
        tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
        provide both rnn_construction_fn and lstm_size.
      rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
        rnn_construction_fn.

        The RNN will be constructed via:

        ```
        rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
        ```
      name: A string representing name of the network.

    Raises:
      ValueError: If 'input_dropout_layer_params' is not None.
    """
    if input_dropout_layer_params:
      raise ValueError('Dropout layer is not supported.')

    lstm_encoder = lstm_encoding_network.LSTMEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs=rnn_construction_kwargs,
        dtype=dtype,
        name=name)

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(ActorDistributionRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        output_spec=output_spec,
        name=name)

    self._lstm_encoder = lstm_encoder
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observation, step_type, network_state=(), training=False):
    state, network_state = self._lstm_encoder(
        observation, step_type=step_type, network_state=network_state,
        training=training)
    outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
    output_actions = tf.nest.map_structure(
        lambda proj_net: proj_net(state, outer_rank, training=training)[0],
        self._projection_networks)
    return output_actions, network_state
