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

"""Network layer that allows mapping multiple inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import typing

import tensorflow.compat.v2 as tf

from tf_agents.networks import network
from tf_agents.networks import sequential
from tf_agents.typing import types
from tf_agents.utils import nest_utils


def NestFlatten() -> tf.keras.layers.Layer:  # pylint: disable=invalid-name
  """Returns a Keras layer that takes a nest of inputs, and returns a list.

  Useful in combination with `NestMap` to combine processed inputs:

  ```python
  # Process inputs in dictionary {"inp1": ..., "inp2": ...}, then
  # flatten the resulting tensors into a list, and finally pass this
  # list to tf.keras.layers.Add() to sum the values element-wise.
  net = tf_agents.networks.Sequence([
    NestMap({"inp1": layer1, "inp2": layer2}),
    NestFlatten(),
    tf.keras.layers.Add(),
  ])
  combined_outputs, next_state = net({"inp1": inp1, "inp2": inp2}, state)
  ```
  """
  return tf.keras.layers.Lambda(tf.nest.flatten)


class NestMap(network.Network):
  """The `NestMap` network processes nested inputs via nested layers.

  It is a TF-Agents network that can be used to process nested inputs.

  Stateful Keras layers (e.g. LSTMCell, RNN, LSTM, TF-Agents DynamicUnroll)
  are all supported.  The `state_spec` of `NestMap` has a structure matching
  that of `nested_layers`.

  `NestMap` can be used in conjunction with `NestFlatten` and a combiner
  (e.g. `tf.keras.layers.Add` or `tf.keras.layers.Concatenate`) to process
  and aggregate in a preprocessing step.

  Usage:
  ```python
  net = NestMap({"inp1": layer1, "inp2": layer2})
  outputs, next_state = net({"inp1": inp1, "inp2": inp2}, state)
  ```
  """

  def __init__(self,
               nested_layers: types.NestedLayer,
               input_spec: typing.Optional[types.NestedTensorSpec] = None,
               name: typing.Optional[typing.Text] = None):
    """Create a Sequential Network.

    Args:
      nested_layers: A nest of layers and/or networks.  These will be used
        to process the inputs (input nest structure will have to match this
        structure).  Any layers that are subclasses of
        `tf.keras.layers.{RNN,LSTM,GRU,...}` are wrapped in
        `tf_agents.keras_layers.RNNWrapper`.
      input_spec: (Optional.)  A nest of `tf.TypeSpec` representing the
        input observations.  The structure of `input_spec` must match
        that of `nested_layers`.
      name: (Optional.) Network name.

    Raises:
      TypeError: If any of the layers are not instances of keras `Layer`.
      ValueError: If `input_spec` is provided but its nest structure does
        not match that of `nested_layers`.
      RuntimeError: If not `tf.executing_eagerly()`; as this is required to
        be able to create deep copies of layers in `layers`.
    """
    if not tf.executing_eagerly():
      raise RuntimeError(
          'Not executing eagerly - cannot make deep copies of `nested_layers`.')

    flat_nested_layers = tf.nest.flatten(nested_layers)
    for layer in flat_nested_layers:
      if not isinstance(layer, tf.keras.layers.Layer):
        raise TypeError(
            'Expected all layers to be instances of keras Layer, but saw'
            ': \'{}\''.format(layer))

    if input_spec is not None:
      nest_utils.assert_same_structure(
          nested_layers, input_spec,
          message=(
              '`nested_layers` and `input_spec` do not have matching structures'
          ))
      flat_input_spec = tf.nest.flatten(input_spec)
    else:
      flat_input_spec = [None] * len(flat_nested_layers)

    # Wrap in Sequential if necessary.
    flat_nested_layers = [
        sequential.Sequential([m], s) if not isinstance(m, network.Network)
        else m
        for (s, m) in zip(flat_input_spec, flat_nested_layers)
    ]

    flat_nested_layers_state_specs = [m.state_spec for m in flat_nested_layers]
    nested_layers = tf.nest.pack_sequence_as(nested_layers, flat_nested_layers)
    # We use flattened layers and states here instead of tf.nest.map_structure
    # for several reason.  One is that we perform several operations against
    # the layers and we want to avoid calling into tf.nest.map* multiple times.
    # But the main reason is that network states have a different *structure*
    # than the layers; e.g., `nested_layers` may just be tf.keras.layers.LSTM,
    # but the states would then have structure `[.,.]`.  Passing these in
    # as args to tf.nest.map_structure causes it to fail.  Instead we would
    # have to use nest.map_structure_up_to -- but that function is not part
    # of the public TF API.  However, if we do everything in flatland and then
    # use pack_sequence_as, we bypass the more rigid structure tests.
    state_spec = tf.nest.pack_sequence_as(
        nested_layers, flat_nested_layers_state_specs)

    super(NestMap, self).__init__(input_tensor_spec=input_spec,
                                  state_spec=state_spec,
                                  name=name)
    self._nested_layers = nested_layers

  @property
  def nested_layers(self) -> types.NestedNetwork:
    # Return a shallow copy so users don't modify the layers list.
    return tf.nest.map_structure(lambda m: m, self._nested_layers)

  def copy(self, **kwargs) -> 'NestMap':
    """Make a copy of a `NestMap` instance.

    **NOTE** A copy of a `NestMap` instance always performs a deep copy
    of the underlying layers, so the new instance will not share weights
    with the original - but it will start with the same weights.

    Args:
      **kwargs: Args to override when recreating this network.  Commonly
        overridden args include 'name'.

    Returns:
      A deep copy of this network.
    """
    new_kwargs = dict(self._saved_kwargs, **kwargs)
    if 'nested_layers' not in new_kwargs:
      new_nested_layers = [copy.deepcopy(m) for m in self._nested_layers]
      new_kwargs['nested_layers'] = new_nested_layers
    return type(self)(**new_kwargs)

  def call(self, inputs, network_state=(), **kwargs):
    nest_utils.assert_same_structure(
        self._nested_layers, inputs,
        allow_shallow_nest1=True,
        message=(
            '`self.nested_layers` and `inputs` do not have matching structures')
    )

    if network_state:
      nest_utils.assert_same_structure(
          self.state_spec, network_state,
          allow_shallow_nest1=True,
          message=(
              'network_state and state_spec do not have matching structure'))
      nested_layers_state = network_state
    else:
      nested_layers_state = tf.nest.map_structure(
          lambda _: (), self._nested_layers)

    # Here we must use map_structure_up_to because nested_layers_state has a
    # "deeper" structure than self._nested_layers.  For example, an LSTM
    # layer's state is composed of a list with two tensors.  The
    # tf.nest.map_structure function would raise an error if two
    # "incompatible" structures are passed in this way.
    def _mapper(inp, layer, state):  # pylint: disable=invalid-name
      return layer(inp, network_state=state, **kwargs)

    outputs_and_next_state = nest_utils.map_structure_up_to(
        self._nested_layers, _mapper,
        inputs, self._nested_layers, nested_layers_state)

    flat_outputs_and_next_state = nest_utils.flatten_up_to(
        self._nested_layers, outputs_and_next_state)
    flat_outputs, flat_next_state = zip(*flat_outputs_and_next_state)

    outputs = tf.nest.pack_sequence_as(
        self._nested_layers, flat_outputs)
    next_network_state = tf.nest.pack_sequence_as(
        self._nested_layers, flat_next_state)

    return outputs, next_network_state
