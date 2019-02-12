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

"""Helper classes for tf_agents.agents.dqn.DqnAgent testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import network


class DummyNet(network.Network):
  """Dummy network for testing DqnAgent."""

  def __init__(self, unused_observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(
        unused_observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1
    self._layers.append(
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.compat.v1.initializers.constant([[2, 1],
                                                                   [1, 1]]),
            bias_initializer=tf.compat.v1.initializers.constant([[1], [1]]),
            dtype=tf.float32))

  def call(self, inputs, unused_step_type=None, network_state=()):
    inputs = tf.cast(inputs, tf.float32)
    for layer in self.layers:
      inputs = layer(inputs)
    return inputs, network_state
