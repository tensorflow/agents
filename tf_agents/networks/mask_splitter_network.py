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

"""Wrapper network that handles action constraint portion of the observation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tf_agents.networks import network
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class MaskSplitterNetwork(network.DistributionNetwork):
  """Separates and passes the observation and mask to the wrapped network.

  In some environments, different sets of actions are available given different
  observations. To represent this, `env.observation` actually contains both the
  raw observation, and an action mask for this particular observation. Our
  network needs to know how to split `env.observation` into these two parts. The
  raw observation will be fed into the wrapped network, and the action mask will
  be optionally passed into the wrapped network to ensure that the network only
  outputs possible actions.

  The network uses the `splitter_fn` to separate the observation from the action
  mask (i.e. `observation, mask = splitter_fn(inputs)`). Depending on the value
  of `pass_mask_to_wrapped_network` the mask is passed into the wrapped network
  or dropped, i.e.

  ```python
  obs, mask = splitter_fn(inputs)

  wrapped_network(obs, ...)  # If pass_mask_to_wrapped_network is `False`

  wrapped_network(obs, ..., mask=mask)  # Otherwise, i.e. it is `True`.
  ```

  In each case the observation part is fed into the `wrapped_network`. It is
  expected that the input spec of wrapped network is compatible with the
  observation part of the input of the `MaskSplitterNetwork`.
  """

  def __init__(self,
               splitter_fn: types.Splitter,
               wrapped_network: network.Network,
               passthrough_mask: bool = False,
               input_tensor_spec: Optional[types.NestedTensorSpec] = None,
               name: Text = 'MaskSplitterNetwork'):
    """Initializes an instance of `MaskSplitterNetwork`.

    Args:
      splitter_fn: A function used to process observations with action
        constraints (i.e. mask).
        *Note*: The input spec of the wrapped network must be compatible with
          the network-specific half of the output of the `splitter_fn` on the
          input spec.
      wrapped_network: A `network.Network` used to process the network-specific
        part of the observation, and the mask passed as the `mask` parameter of
        the method `call` of the wrapped network.
      passthrough_mask: If it is set to `True`, the mask is fed into wrapped
        network. If it is set to `False`, the mask portion of the input is
        dropped and *not* fed into the wrapped network.
      input_tensor_spec: A `tensor_spec.TensorSpec` or a tuple of specs
        representing the input observations including the specs of the action
        constraints.
      name: A string representing name of the network.

    Raises:
      ValueError: If input_tensor_spec is not an instance of network.InputSpec.
    """
    output_spec = (
        wrapped_network.output_spec
        if isinstance(wrapped_network, network.DistributionNetwork) else None)
    super(MaskSplitterNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=wrapped_network.state_spec,
        output_spec=output_spec,
        name=name)
    # Check if the input spec without the spec of the mask is compatible with
    # the input spec of wrapped network.
    if input_tensor_spec is not None:
      input_spec_without_mask, _ = splitter_fn(input_tensor_spec)
      if wrapped_network.input_tensor_spec is not None:
        nest_utils.flatten_and_check_shape_nested_specs(
            input_spec_without_mask, wrapped_network.input_tensor_spec)

    # Store properties.
    self._wrapped_network = wrapped_network
    self._splitter_fn = splitter_fn
    self._passthrough_mask = passthrough_mask

  def call(self,
           observation,
           step_type=None,
           network_state=(),
           training=False,
           **kwargs):
    observation_without_mask, mask = self._splitter_fn(observation)
    if self._passthrough_mask:
      kwargs['mask'] = mask
    return self._wrapped_network(
        inputs=observation_without_mask,
        step_type=step_type,
        network_state=network_state,
        training=training,
        **kwargs)

  def create_variables(self, input_tensor_spec=None, **kwargs):
    # Mark the current network built.
    self.built = True

    # Build wrapped network.
    input_tensor_spec_without_mask = None
    input_tensor_spec = input_tensor_spec or self.input_tensor_spec
    if input_tensor_spec is not None:
      input_tensor_spec_without_mask, _ = self._splitter_fn(input_tensor_spec)
    return self._wrapped_network.create_variables(
        input_tensor_spec_without_mask, **kwargs)

  def copy(self, **kwargs):
    # Create a shallow copy of this network by recreating the instance using the
    # same arguments that were used to create the original instance (except
    # `kwargs` which make it possible to replace some).
    full_kwargs = dict(self._saved_kwargs, **kwargs)
    if 'wrapped_network' not in kwargs:
      # In the case of `wrapped_network` provided in `kwargs` copy is not called
      # since it is assume it is already ready to use as it is.
      full_kwargs['wrapped_network'] = self._wrapped_network.copy()
    return type(self)(**full_kwargs)
