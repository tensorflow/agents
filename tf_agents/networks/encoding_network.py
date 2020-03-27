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

# Lint as: python2, python3
"""Keras Encoding Network.

Implements a network that will generate the following layers:

  [optional]: preprocessing_layers  # preprocessing_layers
  [optional]: (Add | Concat(axis=-1) | ...)  # preprocessing_combiner
  [optional]: Conv2D # conv_layer_params
  Flatten
  [optional]: Dense  # fc_layer_params
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import gin
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal

CONV_TYPE_2D = '2d'
CONV_TYPE_1D = '1d'


def _copy_layer(layer):
  """Create a copy of a Keras layer with identical parameters.

  The new layer will not share weights with the old one.

  Args:
    layer: An instance of `tf.keras.layers.Layer`.

  Returns:
    A new keras layer.

  Raises:
    TypeError: If `layer` is not a keras layer.
    ValueError: If `layer` cannot be correctly cloned.
  """
  if not isinstance(layer, tf.keras.layers.Layer):
    raise TypeError('layer is not a keras layer: %s' % str(layer))

  # pylint:disable=unidiomatic-typecheck
  if type(layer) == tf.compat.v1.keras.layers.DenseFeatures:
    raise ValueError('DenseFeatures V1 is not supported. '
                     'Use tf.compat.v2.keras.layers.DenseFeatures instead.')
  if layer.built:
    logging.warning(
        'Beware: Copying a layer that has already been built: \'%s\'.  '
        'This can lead to subtle bugs because the original layer\'s weights '
        'will not be used in the copy.', layer.name)
  # Get a fresh copy so we don't modify an incoming layer in place.  Weights
  # will not be shared.
  return type(layer).from_config(layer.get_config())


@gin.configurable
class EncodingNetwork(network.Network):
  """Feed Forward network with CNN and FNN layers."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=None,
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               weight_decay_params=None,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='EncodingNetwork',
               conv_type=CONV_TYPE_2D):
    """Creates an instance of `EncodingNetwork`.

    Network supports calls with shape outer_rank + input_tensor_spec.shape. Note
    outer_rank must be at least 1.

    For example an input tensor spec with shape `(2, 3)` will require
    inputs with at least a batch size, the input shape is `(?, 2, 3)`.

    Input preprocessing is possible via `preprocessing_layers` and
    `preprocessing_combiner` Layers.  If the `preprocessing_layers` nest is
    shallower than `input_tensor_spec`, then the layers will get the subnests.
    For example, if:

    ```python
    input_tensor_spec = ([TensorSpec(3)] * 2, [TensorSpec(3)] * 5)
    preprocessing_layers = (Layer1(), Layer2())
    ```

    then preprocessing will call:

    ```python
    preprocessed = [preprocessing_layers[0](observations[0]),
                    preprocessing_layers[1](obsrevations[1])]
    ```

    However if

    ```python
    preprocessing_layers = ([Layer1() for _ in range(2)],
                            [Layer2() for _ in range(5)])
    ```

    then preprocessing will call:
    ```python
    preprocessed = [
      layer(obs) for layer, obs in zip(flatten(preprocessing_layers),
                                       flatten(observations))
    ]
    ```

    **NOTE** `preprocessing_layers` and `preprocessing_combiner` are not allowed
    to have already been built.  This ensures calls to `network.copy()` in the
    future always have an unbuilt, fresh set of parameters.  Furtheremore,
    a shallow copy of the layers is always created by the Network, so the
    layer objects passed to the network are never modified.  For more details
    of the semantics of `copy`, see the docstring of
    `tf_agents.networks.Network.copy`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations. All of these
        layers must not be already built.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them.  Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`. This
        layer must not be already built.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is either a length-three tuple indicating
        `(filters, kernel_size, stride)` or a length-four tuple indicating
        `(filters, kernel_size, stride, dilation_rate)`.
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
      activation_fn: Activation function, e.g. tf.keras.activations.relu.
      weight_decay_params: Optional list of weight decay parameters for the
        fully connected layers.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default variance_scaling_initializer
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.
      conv_type: string, '1d' or '2d'. Convolution layers will be 1d or 2D
        respectively

    Raises:
      ValueError: If any of `preprocessing_layers` is already built.
      ValueError: If `preprocessing_combiner` is already built.
      ValueError: If the number of dropout layer parameters does not match the
        number of fully connected layer parameters.
      ValueError: If conv_layer_params tuples do not have 3 or 4 elements each.
    """
    if preprocessing_layers is None:
      flat_preprocessing_layers = None
    else:
      flat_preprocessing_layers = [
          _copy_layer(layer) for layer in tf.nest.flatten(preprocessing_layers)
      ]
      # Assert shallow structure is the same. This verifies preprocessing
      # layers can be applied on expected input nests.
      input_nest = input_tensor_spec
      # Given the flatten on preprocessing_layers above we need to make sure
      # input_tensor_spec is a sequence for the shallow_structure check below
      # to work.
      if not nest.is_sequence(input_tensor_spec):
        input_nest = [input_tensor_spec]
      nest.assert_shallow_structure(preprocessing_layers, input_nest)

    if (len(tf.nest.flatten(input_tensor_spec)) > 1 and
        preprocessing_combiner is None):
      raise ValueError(
          'preprocessing_combiner layer is required when more than 1 '
          'input_tensor_spec is provided.')

    if preprocessing_combiner is not None:
      preprocessing_combiner = _copy_layer(preprocessing_combiner)

    if not kernel_initializer:
      kernel_initializer = tf.compat.v1.variance_scaling_initializer(
          scale=2.0, mode='fan_in', distribution='truncated_normal')

    layers = []

    if conv_layer_params:
      if conv_type == '2d':
        conv_layer_type = tf.keras.layers.Conv2D
      elif conv_type == '1d':
        conv_layer_type = tf.keras.layers.Conv1D
      else:
        raise ValueError('unsupported conv type of %s. Use 1d or 2d' % (
            conv_type))

      for config in conv_layer_params:
        if len(config) == 4:
          (filters, kernel_size, strides, dilation_rate) = config
        elif len(config) == 3:
          (filters, kernel_size, strides) = config
          dilation_rate = (1, 1) if conv_type == '2d' else (1,)
        else:
          raise ValueError(
              'only 3 or 4 elements permitted in conv_layer_params tuples')
        layers.append(
            conv_layer_type(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                dtype=dtype))

    layers.append(tf.keras.layers.Flatten())

    if fc_layer_params:
      if dropout_layer_params is None:
        dropout_layer_params = [None] * len(fc_layer_params)
      else:
        if len(dropout_layer_params) != len(fc_layer_params):
          raise ValueError('Dropout and fully connected layer parameter lists'
                           'have different lengths (%d vs. %d.)' %
                           (len(dropout_layer_params), len(fc_layer_params)))
      if weight_decay_params is None:
        weight_decay_params = [None] * len(fc_layer_params)
      else:
        if len(weight_decay_params) != len(fc_layer_params):
          raise ValueError('Weight decay and fully connected layer parameter '
                           'lists have different lengths (%d vs. %d.)' %
                           (len(weight_decay_params), len(fc_layer_params)))

      for num_units, dropout_params, weight_decay in zip(
          fc_layer_params, dropout_layer_params, weight_decay_params):
        kernal_regularizer = None
        if weight_decay is not None:
          kernal_regularizer = tf.keras.regularizers.l2(weight_decay)
        layers.append(
            tf.keras.layers.Dense(
                num_units,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernal_regularizer,
                dtype=dtype))
        if not isinstance(dropout_params, dict):
          dropout_params = {'rate': dropout_params} if dropout_params else None

        if dropout_params is not None:
          layers.append(utils.maybe_permanent_dropout(**dropout_params))

    super(EncodingNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    # Pull out the nest structure of the preprocessing layers. This avoids
    # saving the original kwarg layers as a class attribute which Keras would
    # then track.
    self._preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                                     preprocessing_layers)
    self._flat_preprocessing_layers = flat_preprocessing_layers
    self._preprocessing_combiner = preprocessing_combiner
    self._postprocessing_layers = layers
    self._batch_squash = batch_squash

  def call(self, observation, step_type=None, network_state=(), training=False):
    del step_type  # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(
          observation, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    if self._flat_preprocessing_layers is None:
      processed = observation
    else:
      processed = []
      for obs, layer in zip(
          nest.flatten_up_to(self._preprocessing_nest, observation),
          self._flat_preprocessing_layers):
        processed.append(layer(obs, training=training))
      if len(processed) == 1 and self._preprocessing_combiner is None:
        # If only one observation is passed and the preprocessing_combiner
        # is unspecified, use the preprocessed version of this observation.
        processed = processed[0]

    states = processed

    if self._preprocessing_combiner is not None:
      states = self._preprocessing_combiner(states)

    for layer in self._postprocessing_layers:
      states = layer(states, training=training)

    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)

    return states, network_state
