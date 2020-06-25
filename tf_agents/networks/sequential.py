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

"""Keras layer to replace the Sequential Model object."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import copy
import typing

import tensorflow.compat.v2 as tf

from tf_agents.keras_layers import rnn_wrapper
from tf_agents.networks import network
from tf_agents.typing import types


def _infer_specs(
    layers: typing.Sequence[tf.keras.layers.Layer],
    input_spec: types.NestedTensorSpec
) -> typing.Tuple[
    types.NestedTensorSpec,
    types.NestedTensorSpec
]:
  """Infer the state spec of a sequence of keras Layers and Networks.

  This runs `create_variables` on each layer, and identifies the
  state spec from each.  Running `create_variables` is necessary
  because this creates a `_network_state_spec` property on each
  generic (non-Network) Keras layer.

  Args:
    layers: A list of Keras layers and Network.
    input_spec: The input to the first laayer.

  Returns:
    A tuple `(output_spec, state_spec)` where `output_spec` is the output spec
    from the final layer and `state_spec` is a tuple of the state specs.
  """
  state_specs = []

  output_spec = input_spec
  for layer in layers:
    output_spec = network.create_variables(layer, output_spec)
    state_spec = network.get_state_spec(layer)
    state_specs.append(state_spec)

  state_specs = tuple(state_specs)
  return output_spec, state_specs


class Sequential(network.Network):
  """The Sequential represents a sequence of Keras layers.

  It is a Keras Layer that can be used instead of tf.keras.layers.Sequential.
  In contrast to keras Sequential, this layer can be used as a pure Layer in
  tf.functions and when exporting SavedModels, without having to pre-declare
  input and output shapes. In turn, this layer is usable as a preprocessing
  layer for TF Agents Networks, and can be exported via PolicySaver.

  Stateful Keras layers (e.g. LSTMCell, RNN, LSTM, TF-Agents DynamicUnroll)
  are all supported.  The `state_spec` of `Sequential` is a tuple whose
  length matches the number of stateful layers passed.  If no stateful layers
  or networks are passed to `Sequential` then `state_spec == ()`.

  Usage:
  ```python
  c = Sequential([layer1, layer2, layer3])
  output, next_state = c(inputs, state)
  ```
  """

  def __init__(self,
               layers: typing.Sequence[tf.keras.layers.Layer],
               input_spec: types.NestedTensorSpec = None,
               name: typing.Text = None):
    """Create a Sequential Network.

    Args:
      layers: A list or tuple of layers to compose.  Any layers that
        are subclasses of `tf.keras.layers.{RNN,LSTM,GRU,...}` are
        wrapped in `tf_agents.keras_layers.RNNWrapper`.
      input_spec: A nest of `tf.TypeSpec` representing the
        input observations.  Optional.  If not provided, this class will
        perform a best effort to identify the input spec based on the first
        `Layer` or `Network` in `layers`; but `create_variables()` may fail.
      name: (Optional.) Network name.

    Raises:
      ValueError: If `layers` is empty.
      ValueError: If `layers[0]` is a generic Keras layer (not a TF-Agents
        network) and `input_spec is None`.
      TypeError: If any of the layers are not instances of keras `Layer`.
      RuntimeError: If not `tf.executing_eagerly()`; as this is required to
        be able to create deep copies of layers in `layers`.
    """
    if not tf.executing_eagerly():
      raise RuntimeError(
          'Not executing eagerly - cannot make deep copies of `layers`.')
    if not layers:
      raise ValueError(
          '`layers` must not be empty; saw: {}'.format(layers))
    for layer in layers:
      if not isinstance(layer, tf.keras.layers.Layer):
        raise TypeError(
            'Expected all layers to be instances of keras Layer, but saw'
            ': \'{}\''.format(layer))

    layers = [
        rnn_wrapper.RNNWrapper(layer)
        if isinstance(layer, tf.keras.layers.RNN) else layer
        for layer in layers
    ]

    # Note: _infer_specs also builds the layers.  If `input_spec` is None and
    # the first layer is a generic Keras layer (not a TF-Agents Network), an
    # error will be raised.
    output_spec, state_spec = _infer_specs(layers, input_spec)

    # Now we remove all of the empty state specs so if there are no RNN layers,
    # our state spec is empty.  layer_has_state is a list of bools telling us
    # which layers have a state and which don't.
    # TODO(b/158804957): tf.function changes "s in ((),)" to a tensor bool expr.
    # pylint: disable=literal-comparison
    layer_has_state = [s is not () for s in state_spec]
    state_spec = tuple(s for s in state_spec if s is not ())
    # pylint: enable=literal-comparison
    super(Sequential, self).__init__(input_tensor_spec=input_spec,
                                     state_spec=state_spec,
                                     name=name)
    self._sequential_layers = layers
    self._layer_has_state = layer_has_state
    self._network_output_spec = output_spec

  @property
  def layers(self) -> typing.List[tf.keras.layers.Layer]:
    # Return a shallow copy so users don't modify the layers list.
    return copy.copy(self._sequential_layers)

  def copy(self, **kwargs) -> 'Sequential':
    """Make a copy of a `Sequential` instance.

    **NOTE** A copy of a `Sequential` instance always performs a deep copy
    of the underlying layers, so the new instance will not share weights
    with the original - but it will start with the same weights.

    Args:
      **kwargs: Args to override when recreating this network.  Commonly
        overridden args include 'name'.

    Returns:
      A deep copy of this network.
    """
    new_kwargs = dict(self._saved_kwargs, **kwargs)
    if 'layers' not in new_kwargs:
      new_layers = [copy.deepcopy(l) for l in self.layers]
      new_kwargs['layers'] = new_layers
    return type(self)(**new_kwargs)

  def call(self, inputs, network_state=(), **kwargs):
    if not network_state:
      network_state = ((),) * len(self.state_spec)
    next_network_state = [()] * len(self.state_spec)

    # Only Networks are expected to know about step_type; not Keras layers.
    layer_kwargs = kwargs.copy()
    layer_kwargs.pop('step_type', None)

    stateful_layer_idx = 0
    for i, layer in enumerate(self.layers):
      if isinstance(layer, network.Network):
        if self._layer_has_state[i]:
          inputs, next_network_state[stateful_layer_idx] = layer(
              inputs,
              network_state=network_state[stateful_layer_idx],
              **kwargs)
          stateful_layer_idx += 1
        else:
          inputs, _ = layer(inputs, **kwargs)
      else:
        # Generic Keras layer
        if self._layer_has_state[i]:
          # The layer maintains state.  If a state was provided at input to
          # `call`, then use it.  Otherwise ask for an initial state.
          input_state = (network_state[stateful_layer_idx]
                         or layer.get_initial_state(inputs))
          outputs = layer(inputs, input_state, **layer_kwargs)
          inputs, next_network_state[stateful_layer_idx] = outputs
          stateful_layer_idx += 1
        else:
          # Does not maintain state.
          inputs = layer(inputs, **layer_kwargs)

    return inputs, tuple(next_network_state)
