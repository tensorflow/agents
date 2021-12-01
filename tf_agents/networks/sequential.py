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

"""Keras layer to replace the Sequential Model object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from typing import Any, List, Mapping, Optional, Sequence, Text, Tuple, Union

import tensorflow as tf

from tf_agents.keras_layers import rnn_wrapper
from tf_agents.networks import network
from tf_agents.typing import types


def _infer_state_specs(
    layers: Sequence[tf.keras.layers.Layer]
) -> Tuple[types.NestedTensorSpec, List[bool]]:
  """Infer the state spec of a sequence of keras Layers and Networks.

  Args:
    layers: A list of Keras layers and Network.

  Returns:
    A tuple with `state_spec`, a tuple of the state specs of length
    `len(layers)` and a list of bools indicating if the corresponding layer
    has lists in it's state.
  """
  state_specs = []
  layer_state_is_list = []
  for layer in layers:
    spec = network.get_state_spec(layer)
    if isinstance(spec, list):
      layer_state_is_list.append(True)
      state_specs.append(tuple(spec))
    else:
      state_specs.append(spec)
      layer_state_is_list.append(False)

  return tuple(state_specs), layer_state_is_list


class Sequential(network.Network):
  """The Sequential Network represents a sequence of Keras layers.

  It is a TF-Agents network that should be used instead of
  tf.keras.layers.Sequential. In contrast to keras Sequential, this layer can be
  used as a pure Layer in tf.functions and when exporting SavedModels, without
  having to pre-declare input and output shapes. In turn, this layer is usable
  as a preprocessing layer for TF Agents Networks, and can be exported via
  PolicySaver.

  Stateful Keras layers (e.g. LSTMCell, RNN, LSTM, TF-Agents DynamicUnroll)
  are all supported.  The `state_spec` of `Sequential` is a tuple whose
  length matches the number of stateful layers passed.  If no stateful layers
  or networks are passed to `Sequential` then `state_spec == ()`. Given that
  the replay buffers do not support specs with lists due to tf.nest vs
  tf.data.nest conflicts `Sequential` will also guarantee that all specs do not
  contain lists.

  Usage:
  ```python
  c = Sequential([layer1, layer2, layer3])
  output, next_state = c(inputs, state)
  ```
  """

  def __init__(self,
               layers: Sequence[tf.keras.layers.Layer],
               input_spec: Optional[types.NestedTensorSpec] = None,
               name: Optional[Text] = None):
    """Create a Sequential Network.

    Args:
      layers: A list or tuple of layers to compose.  Any layers that
        are subclasses of `tf.keras.layers.{RNN,LSTM,GRU,...}` are
        wrapped in `tf_agents.keras_layers.RNNWrapper`.
      input_spec: (Optional.) A nest of `tf.TypeSpec` representing the
        input observations to the first layer.
      name: (Optional.) Network name.

    Raises:
      ValueError: If `layers` is empty.
      ValueError: If `layers[0]` is a generic Keras layer (not a TF-Agents
        network) and `input_spec is None`.
      TypeError: If any of the layers are not instances of keras `Layer`.
    """
    if not layers:
      raise ValueError(
          '`layers` must not be empty; saw: {}'.format(layers))
    for layer in layers:
      if not isinstance(layer, tf.keras.layers.Layer):
        raise TypeError(
            'Expected all layers to be instances of keras Layer, but saw'
            ': \'{}\''.format(layer))

    layers = [
        rnn_wrapper.RNNWrapper(layer) if isinstance(layer, tf.keras.layers.RNN)
        else layer
        for layer in layers
    ]

    state_spec, self._layer_state_is_list = _infer_state_specs(layers)

    # Now we remove all of the empty state specs so if there are no RNN layers,
    # our state spec is empty.  layer_has_state is a list of bools telling us
    # which layers have a non-empty state and which don't.
    flattened_specs = [tf.nest.flatten(s) for s in state_spec]
    layer_has_state = [bool(fs) for fs in flattened_specs]
    state_spec = tuple(
        s for s, has_state in zip(state_spec, layer_has_state) if has_state)
    super(Sequential, self).__init__(input_tensor_spec=input_spec,
                                     state_spec=state_spec,
                                     name=name)
    self._sequential_layers = layers
    self._layer_has_state = layer_has_state

  @property
  def layers(self) -> List[tf.keras.layers.Layer]:
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

    Raises:
      RuntimeError: If not `tf.executing_eagerly()`; as this is required to
        be able to create deep copies of layers in `layers`.
    """
    if not tf.executing_eagerly():
      raise RuntimeError(
          'Not executing eagerly - cannot make deep copies of `layers`.')
    new_kwargs = dict(self._saved_kwargs, **kwargs)
    if 'layers' not in kwargs:
      new_layers = [copy.deepcopy(l) for l in self.layers]
      new_kwargs['layers'] = new_layers
    return type(self)(**new_kwargs)

  def call(self, inputs, network_state=(), **kwargs):
    if not tf.is_tensor(network_state) and not network_state:
      network_state = ((),) * len(self.state_spec)
    next_network_state = [()] * len(self.state_spec)

    # Only Networks are expected to know about step_type; not Keras layers.
    layer_kwargs = kwargs.copy()
    layer_kwargs.pop('step_type', None)

    stateful_layer_idx = 0
    for i, layer in enumerate(self.layers):
      if isinstance(layer, network.Network):
        if self._layer_has_state[i]:
          input_state = network_state[stateful_layer_idx]

          if input_state is not None and self._layer_state_is_list[i]:
            input_state = list(input_state)

          inputs, next_state = layer(
              inputs,
              network_state=network_state[stateful_layer_idx],
              **kwargs)

          if self._layer_state_is_list[i]:
            next_network_state[stateful_layer_idx] = tuple(next_state)
          else:
            next_network_state[stateful_layer_idx] = next_state

          stateful_layer_idx += 1
        else:
          inputs, _ = layer(inputs, **kwargs)
      else:
        # Generic Keras layer
        if self._layer_has_state[i]:
          # The layer maintains state.  If a state was provided at input to
          # `call`, then use it.  Otherwise ask for an initial state.
          maybe_network_state = network_state[stateful_layer_idx]

          input_state = maybe_network_state

          # pylint: disable=literal-comparison
          if maybe_network_state is None:
            input_state = layer.get_initial_state(inputs)
          elif input_state is not () and self._layer_state_is_list[i]:
            input_state = list(input_state)
          # pylint: enable=literal-comparison

          outputs = layer(inputs, input_state, **layer_kwargs)
          inputs, next_state = outputs

          if self._layer_state_is_list[i]:
            next_network_state[stateful_layer_idx] = tuple(next_state)
          else:
            next_network_state[stateful_layer_idx] = next_state

          stateful_layer_idx += 1
        else:
          # Does not maintain state.
          inputs = layer(inputs, **layer_kwargs)

    return inputs, tuple(next_network_state)

  def compute_output_shape(
      self,
      input_shape: Union[List[int], Tuple[int], tf.TensorShape]) -> (
          tf.TensorShape):
    output_shape = tf.TensorShape(input_shape)
    for l in self._sequential_layers:
      output_shape = l.compute_output_shape(output_shape)
    return tf.TensorShape(output_shape)

  def compute_output_signature(
      self, input_signature: types.NestedSpec) -> types.NestedSpec:
    output_signature = input_signature
    for l in self._sequential_layers:
      output_signature = l.compute_output_signature(output_signature)
    return output_signature

  @property
  def trainable_weights(self) -> List[tf.Variable]:
    if not self.trainable:
      return []
    weights = {}
    for l in self._sequential_layers:
      for v in l.trainable_weights:
        weights[id(v)] = v
    return list(weights.values())

  @property
  def non_trainable_weights(self) -> List[tf.Variable]:
    weights = {}
    for l in self._sequential_layers:
      for v in l.non_trainable_weights:
        weights[id(v)] = v
    return list(weights.values())

  @property
  def trainable(self) -> bool:
    return any([l.trainable for l in self._sequential_layers])

  @trainable.setter
  def trainable(self, value: bool):
    for l in self._sequential_layers:
      l.trainable = value

  def get_config(self) -> Mapping[int, Mapping[str, Any]]:
    config = {}
    for i, layer in enumerate(self._sequential_layers):
      config[i] = {
          'class_name': layer.__class__.__name__,
          'config': copy.deepcopy(layer.get_config())
      }
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None) -> 'Sequential':
    layers = [
        tf.keras.layers.deserialize(conf, custom_objects=custom_objects)
        for conf in config.values()
    ]
    return cls(layers)


tf.keras.utils.get_custom_objects()['SequentialNetwork'] = Sequential
