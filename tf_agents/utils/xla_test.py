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

"""Test for tf_agents.utils.xla."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import xla


class XLATest(tf.test.TestCase):

  def setUp(self):
    super(XLATest, self).setUp()
    tf.config.set_soft_device_placement(True)

  def testIsXLAAvailable(self):
    available = False
    try:
      self.evaluate(tf.xla.experimental.compile(lambda: tf.constant(0.0)))
      available = True
    except:  # pylint: disable=bare-except
      pass
    self.assertEqual(available, xla.is_xla_available())

  def testCompileInGraphMode(self):
    if not xla.is_xla_available():
      self.skipTest('Skipping test: XLA is not available.')

    @xla.compile_in_graph_mode
    def add(x, y):
      return x + y

    z = add(1.0, 2.0)
    self.assertAllClose(3.0, self.evaluate(z))

    @xla.compile_in_graph_mode
    def add_subtract(x, y):
      return {'add': x + y, 'sub': x - y}

    z = add_subtract(1.0, 2.0)
    self.assertAllClose({'add': 3.0, 'sub': -1.0}, self.evaluate(z))

    @xla.compile_in_graph_mode
    def add_divide(x, yz):
      return x + yz['y'] / yz['z']

    z = add_divide(1.0, {'y': 2.0, 'z': 3.0})
    self.assertAllClose(1.0 + 2.0 / 3.0, self.evaluate(z))

    if not tf.compat.v1.executing_eagerly():
      # TF2 seems to have trouble with soft device placement (both in eager and
      # tf.function mode); and here we're specifically testing what happens when
      # XLA is not available, e.g., because we didn't compile with GPU support.
      with tf.device('/gpu:0'):
        z = add_subtract(1.0, 2.0)
      self.assertAllClose({'add': 3.0, 'sub': -1.0}, self.evaluate(z))


if __name__ == '__main__':
  tf.test.main()
