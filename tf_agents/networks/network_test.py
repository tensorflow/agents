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

"""Tests for tf_agents.networks.network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents import specs
from tf_agents.networks import network
from tf_agents.utils import common


class BaseNetwork(network.Network):

  # pylint: disable=useless-super-delegation
  def __init__(self, v1, **kwargs):
    super(BaseNetwork, self).__init__(v1, **kwargs)
  # pylint: enable=useless-super-delegation


class MockNetwork(BaseNetwork):

  def __init__(self, param1, param2, kwarg1=2, kwarg2=3):
    self.param1 = param1
    self.param2 = param2
    self.kwarg1 = kwarg1
    self.kwarg2 = kwarg2
    super(MockNetwork, self).__init__(param1,
                                      state_spec=(),
                                      name='mock')

  def build(self, *args, **kwargs):
    self.var1 = common.create_variable('variable', trainable=False)
    self.var2 = common.create_variable('trainable_variable', trainable=True)

  def call(self, observations, step_type, network_state=None):
    return self.var1 + self.var2


class NoInitNetwork(MockNetwork):
  pass


class NetworkTest(tf.test.TestCase):

  def test_copy_works(self):
    network1 = MockNetwork(0, 1)
    network2 = network1.copy()

    self.assertNotEqual(network1, network2)
    self.assertEqual(0, network2.param1)
    self.assertEqual(1, network2.param2)
    self.assertEqual(2, network2.kwarg1)
    self.assertEqual(3, network2.kwarg2)

  def test_noinit_copy_works(self):
    network1 = NoInitNetwork(0, 1)
    network2 = network1.copy()

    self.assertNotEqual(network1, network2)
    self.assertEqual(0, network2.param1)
    self.assertEqual(1, network2.param2)
    self.assertEqual(2, network2.kwarg1)
    self.assertEqual(3, network2.kwarg2)

  def test_too_many_args_raises_appropriate_error(self):
    with self.assertRaisesRegexp(TypeError, '__init__.*given'):
      # pylint: disable=too-many-function-args
      MockNetwork(0, 1, 2, 3, 4, 5, 6)

  def test_assert_input_spec(self):
    spec = specs.TensorSpec([], tf.int32, 'action')
    net = MockNetwork(spec, 1)
    with self.assertRaises(ValueError):
      net((1, 2), 2)

  def test_create_variables(self):
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([2], tf.float32, 'action')
    net = MockNetwork(observation_spec, action_spec)
    self.assertFalse(net.built)
    with self.assertRaises(ValueError):
      net.variables  # pylint: disable=pointless-statement
    net.create_variables()
    self.assertTrue(net.built)
    self.assertLen(net.variables, 2)
    self.assertLen(net.trainable_variables, 1)

  def test_summary_no_exception(self):
    """Tests that Network.summary() does not throw an exception."""
    observation_spec = specs.TensorSpec([1], tf.float32, 'observation')
    action_spec = specs.TensorSpec([2], tf.float32, 'action')
    net = MockNetwork(observation_spec, action_spec)
    net.create_variables()
    net.summary()

if __name__ == '__main__':
  tf.test.main()
