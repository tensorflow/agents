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

"""Tests for tf_agents.keras_layers.inner_reshape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.keras_layers import inner_reshape
from tf_agents.utils import test_utils


class InnerReshapeTest(test_utils.TestCase):

  def testInnerReshapeSimple(self):
    layer = inner_reshape.InnerReshape([3, 4], [12])
    out = layer(np.arange(2 * 12).reshape(2, 3, 4))
    self.assertAllEqual(
        self.evaluate(out), np.arange(2 * 12).reshape(2, 12))
    out = layer(np.arange(4 * 12).reshape(2, 2, 3, 4))
    self.assertAllEqual(
        self.evaluate(out), np.arange(4 * 12).reshape(2, 2, 12))

  def testInnerReshapeUnknowns(self):
    layer = inner_reshape.InnerReshape([None, None], [-1])
    out = layer(np.arange(3 * 20).reshape(3, 4, 5))
    self.assertAllEqual(
        self.evaluate(out), np.arange(3 * 20).reshape(3, 20))
    out = layer(np.arange(6 * 20).reshape(2, 3, 4, 5))
    self.assertAllEqual(
        self.evaluate(out), np.arange(6 * 20).reshape(2, 3, 20))

  def testIncompatibleShapes(self):
    with self.assertRaisesRegex(ValueError, 'must have known rank'):
      inner_reshape.InnerReshape(tf.TensorShape(None), [1])

    with self.assertRaisesRegex(ValueError, 'Mismatched number of elements'):
      inner_reshape.InnerReshape([1, 2], [])

    with self.assertRaisesRegex(ValueError, r'Shapes.*are incompatible'):
      inner_reshape.InnerReshape([1], [1, 1])(np.ones((2, 3)))


if __name__ == '__main__':
  test_utils.main()
