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

"""Tests for tf_agents.bandits.agents.multi_objective_scalarizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow.compat.v2 as tf

from tf_agents.bandits.multi_objective import multi_objective_scalarizer


class DummyScalarizer(multi_objective_scalarizer.Scalarizer):

  def call(self, multi_objectives):
    pass


class BaseScalarizerTest(tf.test.TestCase):

  def testInitialization(self):
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      DummyScalarizer(1)
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      DummyScalarizer(0)
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      DummyScalarizer(-1)

  def testValidation(self):
    scalarizer = DummyScalarizer(3)
    with self.assertRaisesRegex(ValueError,
                                'The rank of the input should be 2'):
      multi_objectives = tf.constant([1, 2, 3, 4], dtype=tf.float32)
      scalarizer(multi_objectives)
    with self.assertRaisesRegex(ValueError,
                                'The number of input objectives should be'):
      multi_objectives = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
      scalarizer(multi_objectives)


class LinearScalarizerTest(tf.test.TestCase):

  def setUp(self):
    self._scalarizer = multi_objective_scalarizer.LinearScalarizer(
        [1, 2, 3, -1])
    super(LinearScalarizerTest, self).setUp()

  def testInvalidWeights(self):
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      multi_objective_scalarizer.LinearScalarizer([])
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      multi_objective_scalarizer.LinearScalarizer([1])

  def testInvalidObjectives(self):
    with self.assertRaisesRegex(ValueError,
                                'The number of input objectives should be'):
      multi_objectives = tf.constant([[1, 2, 3]], dtype=tf.float32)
      self._scalarizer(multi_objectives)

  def testBatchSizeOne(self):
    multi_objectives = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    self.assertEqual(
        self.evaluate(self._scalarizer(multi_objectives)),
        self.evaluate(tf.constant([10], dtype=tf.float32)))

  def testBatchSizeThree(self):
    multi_objectives = tf.constant(
        [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]], dtype=tf.float32)
    self.assertAllEqual(
        self.evaluate(self._scalarizer(multi_objectives)),
        self.evaluate(tf.constant([10, 30, -10], dtype=tf.float32)))


class ChebyShevScalarizerTest(tf.test.TestCase):

  def setUp(self):
    self._scalarizer = multi_objective_scalarizer.ChebyshevScalarizer(
        [1, 2, 3, -1], [0, 1, 2, 3])
    super(ChebyShevScalarizerTest, self).setUp()

  def testInvalidWeightsAndReference(self):
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      multi_objective_scalarizer.ChebyshevScalarizer([], [])
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      multi_objective_scalarizer.ChebyshevScalarizer([1], [1])
    with self.assertRaisesRegex(ValueError, 'weights has 2 elements but'):
      multi_objective_scalarizer.ChebyshevScalarizer([1, 2], [0, 0, 0])

  def testInvalidObjectives(self):
    with self.assertRaisesRegex(ValueError,
                                'The number of input objectives should be'):
      multi_objectives = tf.constant([[1, 2, 3]], dtype=tf.float32)
      self._scalarizer(multi_objectives)

  def testBatchSizeOne(self):
    multi_objectives = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
    self.assertEqual(
        self.evaluate(self._scalarizer(multi_objectives)),
        self.evaluate(tf.constant([-1], dtype=tf.float32)))

  def testBatchSizeThree(self):
    multi_objectives = tf.constant(
        [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]], dtype=tf.float32)
    self.assertAllEqual(
        self.evaluate(self._scalarizer(multi_objectives)),
        self.evaluate(tf.constant([-1, -5, -15], dtype=tf.float32)))


class HyperVolumeScalarizerTest(tf.test.TestCase):

  def setUp(self):
    self._hv_params = [
        multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
            slope=1, offset=0)
    ] * 3
    super(HyperVolumeScalarizerTest, self).setUp()

  def testInvalidParams(self):
    with self.assertRaisesRegex(ValueError, 'nearly-zero vector'):
      multi_objective_scalarizer.HyperVolumeScalarizer([], [])
    with self.assertRaisesRegex(ValueError, 'at least two objectives'):
      multi_objective_scalarizer.HyperVolumeScalarizer([1],
                                                       [self._hv_params[0]])

  def testInvalidObjectives(self):
    with self.assertRaisesRegex(ValueError,
                                'The number of input objectives should be'):
      direction = [1, 0, 0]
      scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
          direction, self._hv_params)
      multi_objectives = tf.constant([[1, 2, 3, 4]], dtype=tf.float32)
      scalarizer(multi_objectives)

  def testDirectionWithNegativeCoordinates(self):
    with self.assertRaisesRegex(ValueError, 'has negative coordinates'):
      direction = [-1, 0, 0]
      multi_objective_scalarizer.HyperVolumeScalarizer(direction,
                                                       self._hv_params)

  def testZeroLengthDirection(self):
    with self.assertRaisesRegex(ValueError, 'nearly-zero vector'):
      direction = [0, 0, 0]
      multi_objective_scalarizer.HyperVolumeScalarizer(direction,
                                                       self._hv_params)

  def testDirectionWithWrongDimension(self):
    with self.assertRaisesRegex(ValueError, 'direction has 2 elements but'):
      direction = [1, 0]
      multi_objective_scalarizer.HyperVolumeScalarizer(direction,
                                                       self._hv_params)

  def testAxisAlignedDirection(self):
    direction = [0, 1, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    multi_objectives = tf.constant([[3, 2, 1], [6, 5, 4]], dtype=tf.float32)
    self.assertAllEqual(
        self.evaluate(scalarizer(multi_objectives)),
        self.evaluate(tf.constant([2, 5], dtype=tf.float32)))

  def testDirectionNormalization(self):
    multi_objectives = tf.constant([[3, 2, 1], [6, 5, 4]], dtype=tf.float32)
    for direction in [[1, 2, 2], [0.1, 0.2, 0.2]]:
      scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
          direction, self._hv_params)
      self.assertAllClose(
          self.evaluate(scalarizer(multi_objectives)),
          self.evaluate(tf.constant([1.5, 6], dtype=tf.float32)))

  def testNegativeObjectives(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    multi_objectives = tf.constant([[-3, 2, 1], [-6, 5, 4]], dtype=tf.float32)
    self.assertAllEqual(
        self.evaluate(scalarizer(multi_objectives)),
        self.evaluate(tf.constant([0, 0], dtype=tf.float32)))

  def testObjectiveTranformation(self):
    hv_params = copy.deepcopy(self._hv_params)
    hv_params[0] = multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
        slope=-1, offset=2)
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, hv_params)
    multi_objectives = tf.constant([[-3, 2, 1], [-6, 5, 4]], dtype=tf.float32)
    self.assertAllEqual(
        self.evaluate(scalarizer(multi_objectives)),
        self.evaluate(tf.constant([5, 8], dtype=tf.float32)))

  def testNearZeroObjectivesAndDirectionCoordinates(self):
    multi_objectives = tf.constant([[3, 0, 0]], dtype=tf.float32)

    # direction with small non-zero coordinates.
    nonzero = 2 * multi_objective_scalarizer.HyperVolumeScalarizer.ALMOST_ZERO
    direction = [1, nonzero, nonzero]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    self.assertAllClose(
        self.evaluate(scalarizer(multi_objectives)),
        self.evaluate(tf.constant([0], dtype=tf.float32)))

    # direction with coordinates that are so small in magnitude that would be
    # considered zeros.
    direction = [1, 1e-30, 1e-30]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    self.assertAllClose(
        self.evaluate(scalarizer(multi_objectives)),
        self.evaluate(tf.constant([3], dtype=tf.float32)))

    # direction with absolute zero coordinates.
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    self.assertAllClose(
        self.evaluate(scalarizer(multi_objectives)),
        self.evaluate(tf.constant([3], dtype=tf.float32)))


if __name__ == '__main__':
  tf.test.main()
