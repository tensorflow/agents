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

"""Common utility functions for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.networks import network


class KerasLayersNet(network.Network):

  def __init__(self, observation_spec, action_spec, layer, name=None):
    super(KerasLayersNet, self).__init__(
        observation_spec, state_spec=(), name=name)
    self._layer = layer

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    return self._layer(inputs), network_state
