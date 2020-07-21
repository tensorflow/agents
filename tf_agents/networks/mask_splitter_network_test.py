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

# Lint as: python3
"""Tests MaskSplitterNetwork."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.networks import mask_splitter_network
from tf_agents.networks import network
from tf_agents.networks import value_network
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import test_utils


class WrappedNetwork(network.Network):

  def call(self,
           observation,
           step_type=None,
           network_state=(),
           training=False,
           mask=None):
    del step_type, training  # Unused.
    return (observation, mask), network_state


class WrappedDistributionNetwork(network.DistributionNetwork):

  def __init__(self, **kwargs):
    super(WrappedDistributionNetwork, self).__init__(**kwargs)
    self.mask = None

  def call(self,
           observation,
           step_type=None,
           network_state=(),
           training=False,
           mask=None):
    del step_type, training  # Unused.
    self.mask = mask
    return (self.output_spec.build_distribution(logits=observation),
            network_state)


class MaskSplitterNetworkTest(test_utils.TestCase):

  def setUp(self):
    super(MaskSplitterNetworkTest, self).setUp()
    self._observation_and_mask_spec = {
        'observation': tensor_spec.BoundedTensorSpec((2,), tf.float32, 0, 5),
        'mask': tensor_spec.BoundedTensorSpec((3,), tf.int32, 0, 1),
    }
    self._observation_spec = self._observation_and_mask_spec['observation']
    self._mask_spec = self._observation_and_mask_spec['mask']
    self._state_spec = tensor_spec.BoundedTensorSpec((1,), tf.int64, 0, 10)

    def splitter_fn(observation_and_mask):
      return observation_and_mask['observation'], observation_and_mask['mask']

    self._splitter_fn = splitter_fn
    self._observation_and_mask = tensor_spec.sample_spec_nest(
        self._observation_and_mask_spec, outer_dims=(4,))
    self._network_state = tensor_spec.sample_spec_nest(
        self._state_spec, outer_dims=(4,))

    self._output_spec = distribution_spec.DistributionSpec(
        tfp.distributions.Categorical,
        self._observation_spec,
        sample_spec=tensor_spec.BoundedTensorSpec((1,), tf.int64, 0, 1),
        **tfp.distributions.Categorical(logits=[0, 5]).parameters)

  def testSimpleNetwork(self):
    # Create a wrapped network.
    wrapped_network = WrappedNetwork()

    # Create a splitter network which drops the mask (`passthrough_mask=False`).
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn, wrapped_network=wrapped_network)

    # Apply splitter network which returns the `observation` and `mask` received
    # by the `wrapped_network`.
    (observation, mask), _ = splitter_network(self._observation_and_mask)

    # Check if the wrapped network received the observation part of the input.
    self.assertAllClose(observation, self._observation_and_mask['observation'])
    # The wrapped network should *not* receive mask since the value of
    # `passthrough_mask=False` in the `splitter_network`.
    self.assertIsNone(mask)

  def testDistributionNetwork(self):
    # Create a wrapped network.
    wrapped_network = WrappedDistributionNetwork(
        input_tensor_spec=self._observation_spec,
        state_spec=(),
        output_spec=self._output_spec,
        name='WrappedDistributionNetwork')

    # Create a splitter network which drops the mask (`passthrough_mask=False`).
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn, wrapped_network=wrapped_network)

    # Apply splitter network which returns a distribution based on directly the
    # input `observation`.
    distribution, _ = splitter_network(self._observation_and_mask)

    # Check if distribution was properly created based on the input observation.
    self.assertAllClose(distribution.parameters['logits'],
                        self._observation_and_mask['observation'])
    # The wrapped network should *not* receive mask since the value of
    # `passthrough_mask=False` in the `splitter_network`.
    self.assertIsNone(wrapped_network.mask)
    # Check if the `output_spec` of the `splitter_network` is equal to the
    # `output_spec` of the `wrapped_network`.
    self.assertEqual(splitter_network.output_spec, wrapped_network.output_spec)

  def testNetworkState(self):
    # Create a wrapped network with `state_spec`.
    wrapped_network = WrappedNetwork(state_spec=self._state_spec)

    # Create a splitter network which drops the mask (`passthrough_mask=False`).
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn, wrapped_network=wrapped_network)

    # Apply the mask splitter network passing a state which is returned as the
    # output network state.
    _, network_state = splitter_network(
        self._observation_and_mask, network_state=self._network_state)

    # Check if the wrapped network received the correct network state.
    self.assertAllEqual(network_state, self._network_state)
    # Check if the `state_spec` of the `splitter_network` is equal to the
    # `state_spec` of the `wrapped_network`.
    self.assertEqual(splitter_network.state_spec, wrapped_network.state_spec)

  def testSimpleNetworkWithPassthroughMask(self):
    # Create a wrapped network.
    wrapped_network = WrappedNetwork()

    # Create a splitter network which passes the mask (`passthrough_mask=True`).
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn,
        wrapped_network=wrapped_network,
        passthrough_mask=True)  # The mask is passed through.

    # Apply splitter network which returns the `observation` and `mask` received
    # by the `wrapped_network`.
    (observation, mask), _ = splitter_network(self._observation_and_mask)

    # Check if the wrapped network received the observation part of the input.
    self.assertAllClose(observation, self._observation_and_mask['observation'])
    # Check if the wrapped network received the right mask .
    self.assertAllEqual(mask, self._observation_and_mask['mask'])

  def testDistributionNetworkWithPassthroughMask(self):
    # Create a wrapped network.
    wrapped_network = WrappedDistributionNetwork(
        input_tensor_spec=self._observation_spec,
        state_spec=(),
        output_spec=self._output_spec,
        name='WrappedDistributionNetwork')

    # Create a splitter network optionally passing the `input_tensor_spec` which
    # always applies the mask (`passthrough_mask=True`).
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn,
        wrapped_network=wrapped_network,
        passthrough_mask=True)  # The mask is passed through.

    # Apply splitter network which returns a distribution based on directly the
    # input `observation`.
    distribution, _ = splitter_network(self._observation_and_mask)

    # Check if distribution was properly created based on the input observation.
    logits = self._observation_and_mask['observation']
    self.assertAllClose(
        logits,
        distribution.parameters['logits'],
    )
    # Check if the wrapped network received the right mask .
    self.assertAllEqual(wrapped_network.mask,
                        self._observation_and_mask['mask'])
    # Check if the `output_spec` of the `splitter_network` is equal to the
    # `output_spec` of the `wrapped_network`.
    self.assertEqual(splitter_network.output_spec, wrapped_network.output_spec)

  def testNetworkStateWithPassthroughMask(self):
    # Create a wrapped network with `state_spec`.
    wrapped_network = WrappedNetwork(state_spec=self._state_spec)

    # Create a splitter network which passes the mask (`passthrough_mask=True`).
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn,
        wrapped_network=wrapped_network,
        passthrough_mask=True)  # The mask is passed through.
    no_passthrough_splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn, wrapped_network=wrapped_network)

    # Apply the mask splitter network passing a state which is returned as the
    # output network state.
    _, network_state = splitter_network(
        self._observation_and_mask, network_state=self._network_state)
    _, no_passthrough_network_state = no_passthrough_splitter_network(
        self._observation_and_mask, network_state=self._network_state)

    # Check if the wrapped network received the correct network state.
    self.assertAllEqual(network_state, self._network_state)
    # Check if the state with and without `passthrough_mask` are the same.
    self.assertAllEqual(network_state, no_passthrough_network_state)
    # Check if the `state_spec` of the `splitter_network` is equal to the
    # `state_spec` of the `wrapped_network`.
    self.assertEqual(splitter_network.state_spec, wrapped_network.state_spec)
    self.assertEqual(no_passthrough_splitter_network.state_spec,
                     wrapped_network.state_spec)

  def testCopyCreateNewInstanceOfNetworkIfNotRedefined(self):
    # Create a wrapped network.
    wrapped_network = value_network.ValueNetwork(
        self._observation_spec, fc_layer_params=(2,))

    # Create and build a `splitter_network`.
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn,
        wrapped_network=wrapped_network,
        passthrough_mask=True,
        input_tensor_spec=self._observation_and_mask_spec)
    splitter_network.create_variables()

    # Copy and build the copied network.
    copied_splitter_network = splitter_network.copy()
    copied_splitter_network.create_variables()

    # Check if the underlying wrapped network objects are different.
    self.assertIsNot(copied_splitter_network._wrapped_network,
                     splitter_network._wrapped_network)

  def testCopyUsesSameWrappedNetwork(self):
    # Create a wrapped network.
    wrapped_network = value_network.ValueNetwork(
        self._observation_spec, fc_layer_params=(2,))

    # Create and build a `splitter_network`.
    splitter_network = mask_splitter_network.MaskSplitterNetwork(
        splitter_fn=self._splitter_fn,
        wrapped_network=wrapped_network,
        passthrough_mask=True,
        input_tensor_spec=self._observation_and_mask_spec)
    splitter_network.create_variables()

    # Crate a copy of the splitter network while redefining the wrapped network.
    copied_splitter_network = splitter_network.copy(
        wrapped_network=wrapped_network)

    # Check if the underlying wrapped network objects are different.
    self.assertIs(copied_splitter_network._wrapped_network,
                  splitter_network._wrapped_network)


if __name__ == '__main__':
  test_utils.main()
