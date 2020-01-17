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

"""Tests for tf_agents.utils.composite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.utils import composite


def _to_dense(st):
  return tf.scatter_nd(st.indices, st.values, st.dense_shape)


class CompositeTest(tf.test.TestCase):
  """Tests functions related to composite tensors."""

  def setUp(self):
    super(CompositeTest, self).setUp()
    self._x = tf.random.uniform([4, 5, 6], dtype=tf.int32, maxval=1000)
    self._sx = tf.sparse.reorder(
        tf.SparseTensor(
            indices=tf.random.uniform([40, 3], maxval=10, dtype=tf.int64),
            values=tf.random.uniform([40], maxval=1000, dtype=tf.int32),
            dense_shape=[10, 10, 10]))

  def testSliceFrom(self):
    from_1 = composite.slice_from(self._x, axis=1, start=1)
    from_n1 = composite.slice_from(self._x, axis=1, start=-1)
    x, from_1, from_n1 = self.evaluate((self._x, from_1, from_n1))
    self.assertAllEqual(from_1, x[:, 1:, :])
    self.assertAllEqual(from_n1, x[:, -1:, :])

    s_from_1 = _to_dense(composite.slice_from(self._sx, axis=1, start=1))
    s_from_n1 = _to_dense(composite.slice_from(self._sx, axis=1, start=-1))
    sx = _to_dense(self._sx)
    sx, s_from_1, s_from_n1 = self.evaluate((sx, s_from_1, s_from_n1))
    self.assertAllEqual(s_from_1, sx[:, 1:, :])
    self.assertAllEqual(s_from_n1, sx[:, -1:, :])

  def testSliceTo(self):
    to_1 = composite.slice_to(self._x, axis=1, end=1)
    to_n1 = composite.slice_to(self._x, axis=1, end=-1)
    x, to_1, to_n1 = self.evaluate((self._x, to_1, to_n1))
    self.assertAllEqual(to_1, x[:, :1, :])
    self.assertAllEqual(to_n1, x[:, :-1, :])

    s_from_1 = _to_dense(composite.slice_to(self._sx, axis=1, end=1))
    s_from_n1 = _to_dense(composite.slice_to(self._sx, axis=1, end=-1))
    sx = _to_dense(self._sx)
    sx, s_from_1, s_from_n1 = self.evaluate((sx, s_from_1, s_from_n1))
    self.assertAllEqual(s_from_1, sx[:, :1, :])
    self.assertAllEqual(s_from_n1, sx[:, :-1, :])

if __name__ == '__main__':
  tf.test.main()
