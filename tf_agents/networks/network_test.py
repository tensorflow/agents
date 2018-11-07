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

from tf_agents.networks import network


class BaseNetwork(network.Network):

  # pylint: disable=useless-super-delegation
  def __init__(self, v1, v2, **kwargs):
    super(BaseNetwork, self).__init__(v1, v2, **kwargs)
  # pylint: enable=useless-super-delegation


class MockNetwork(BaseNetwork):

  def __init__(self, param1, param2, kwarg1=2, kwarg2=3):
    super(MockNetwork, self).__init__(param1,
                                      param2,
                                      state_spec=(),
                                      name='mock')
    self.param1 = param1
    self.param2 = param2
    self.kwarg1 = kwarg1
    self.kwarg2 = kwarg2


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


if __name__ == '__main__':
  tf.test.main()
