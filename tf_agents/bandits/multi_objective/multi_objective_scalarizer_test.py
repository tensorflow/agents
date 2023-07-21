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

"""Tests for tf_agents.bandits.agents.multi_objective_scalarizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow.compat.v2 as tf

from tf_agents.bandits.multi_objective import multi_objective_scalarizer


class DummyScalarizer(multi_objective_scalarizer.Scalarizer):

  def _scalarize(self, multi_objectives):
    pass

  def set_parameters(self):
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

  def testTraceProtocol(self):
    scalarizer_1 = DummyScalarizer(3)
    scalarizer_2 = DummyScalarizer(4)
    trace_type_1 = scalarizer_1.__tf_tracing_type__(None)
    trace_type_2 = scalarizer_2.__tf_tracing_type__(None)

    self.assertNotEqual(trace_type_1, trace_type_1)
    self.assertNotEqual(trace_type_1, trace_type_2)
    self.assertFalse(trace_type_1.is_subtype_of(trace_type_1))
    self.assertFalse(trace_type_1.is_subtype_of(trace_type_2))


class LinearScalarizerTest(tf.test.TestCase):

  def setUp(self):
    super(LinearScalarizerTest, self).setUp()
    self._scalarizer = multi_objective_scalarizer.LinearScalarizer(
        [1, 2, 3, -1])
    self._batch_multi_objectives = tf.constant(
        [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]], dtype=tf.float32)

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
    self.assertAllEqual(
        self.evaluate(self._scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([10, 30, -10], dtype=tf.float32)))

  def testSetParameters(self):
    # batch_size = 3, num_actions = 4.
    self._scalarizer.set_parameters(
        tf.constant([[0.1, 0.2, 0.3, -0.1], [-0.1, 0.2, 0.3, 0.1],
                     [0.1, -0.2, 0.3, 0.1]],
                    dtype=tf.float32))
    self.assertAllClose(
        self.evaluate(self._scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([1, 3.6, -1], dtype=tf.float32)))

  def testSetParametersWrongNumberOfObjectivesRaisesValueError(self):
    with self.assertRaisesRegex(
        ValueError, 'The number of objectives in scalarization parameter'):
      self._scalarizer.set_parameters(
          tf.constant([[0.1, 0.2]], dtype=tf.float32))

  def testCallWrongBatchSizeRaisesValueError(self):
    with self.assertRaisesRegex(ValueError, 'does not match the shape of'):
      self._scalarizer.set_parameters(
          tf.constant([[0.1, 0.2, 0.3, -0.1], [-0.1, 0.2, 0.3, 0.1],
                       [0.1, -0.2, 0.3, 0.1], [0.1, 0.2, -0.3, 0.1]],
                      dtype=tf.float32))
      self.evaluate(self._scalarizer(self._batch_multi_objectives))

  def testCustomTransform(self):
    # Test applying sigmoid to the 2nd and 4th metrics.
    sigmoid_metric_mask = [False, True, False, True]

    def sigmoid_metric_transform(metrics: tf.Tensor):
      batch_size = tf.shape(tf.nest.flatten(metrics)[0])[0]
      sigmoid_batch_mask = tf.reshape(
          tf.tile(sigmoid_metric_mask, [batch_size]),
          [batch_size, len(sigmoid_metric_mask)])
      return tf.where(sigmoid_batch_mask, tf.sigmoid(metrics), metrics)

    sigmoid_scalarizer = multi_objective_scalarizer.LinearScalarizer(
        [1, 2, 3, -1], sigmoid_metric_transform)

    self.assertAllClose(
        sigmoid_scalarizer(self._batch_multi_objectives),
        [10.77958, 26.995390, -9.7795804])


class ChebyShevScalarizerTest(tf.test.TestCase):

  def setUp(self):
    super(ChebyShevScalarizerTest, self).setUp()
    self._scalarizer = multi_objective_scalarizer.ChebyshevScalarizer(
        [1, 2, 3, -1], [0, 1, 2, 3])
    self._batch_multi_objectives = tf.constant(
        [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]], dtype=tf.float32)

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
    self.assertAllEqual(
        self.evaluate(self._scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([-1, -5, -15], dtype=tf.float32)))

  def testSetParameters(self):
    self._scalarizer.set_parameters(
        weights=tf.constant([[0.1, 0.2, 0.3, -0.1], [-0.1, 0.2, 0.3, 0.1],
                             [0.1, -0.2, 0.3, 0.1]],
                            dtype=tf.float32),
        reference_point=tf.constant(
            [[-5, -5, -5, -5], [0, 0, 0, 0], [1, 2, 3, 4]], dtype=tf.float32))
    self.assertAllClose(
        self.evaluate(self._scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([-0.9, -0.5, -1.8], dtype=tf.float32)))

  def testSetParametersWeightsWrongNumberOfObjectivesRaisesValueError(self):
    with self.assertRaisesRegex(
        ValueError, 'The number of objectives in scalarization parameter'):
      self._scalarizer.set_parameters(
          weights=tf.constant([[0.1, 0.2, 0.3]] * 3, dtype=tf.float32),
          reference_point=tf.constant([[-5, -5, -5, -5]] * 3, dtype=tf.float32))

  def testSetParametersReferencePointWrongNumberOfObjectivesRaisesValueError(
      self):
    with self.assertRaisesRegex(
        ValueError, 'The number of objectives in scalarization parameter'):
      self._scalarizer.set_parameters(
          weights=tf.constant([[0.1, 0.2, 0.3, -0.1]] * 3, dtype=tf.float32),
          reference_point=tf.constant([[-5, -5, -5]] * 3, dtype=tf.float32))

  def testCallWeightsWrongBatchSizeRaisesValueError(self):
    with self.assertRaisesRegex(ValueError, 'does not match the shape of'):
      self._scalarizer.set_parameters(
          weights=tf.constant([[0.1, 0.2, 0.3, -0.1], [-0.1, 0.2, 0.3, 0.1],
                               [0.1, -0.2, 0.3, 0.1], [0.1, 0.2, -0.3, 0.1]],
                              dtype=tf.float32),
          reference_point=tf.constant(
              [[-5, -5, -5, -5], [0, 0, 0, 0], [1, 2, 3, 4]], dtype=tf.float32))
      self.evaluate(self._scalarizer(self._batch_multi_objectives))

  def testCallReferencePointWrongBatchSizeRaisesValueError(self):
    with self.assertRaisesRegex(ValueError, 'does not match the shape of'):
      self._scalarizer.set_parameters(
          weights=tf.constant([[0.1, 0.2, 0.3, -0.1], [-0.1, 0.2, 0.3, 0.1],
                               [0.1, -0.2, 0.3, 0.1]],
                              dtype=tf.float32),
          reference_point=tf.constant([[-5, -5, -5, -5], [0, 0, 0, 0]],
                                      dtype=tf.float32))
      self.evaluate(self._scalarizer(self._batch_multi_objectives))


class HyperVolumeScalarizerTest(tf.test.TestCase):

  def setUp(self):
    super(HyperVolumeScalarizerTest, self).setUp()
    self._hv_params = [
        multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
            slope=1, offset=0)
    ] * 3
    self._batch_multi_objectives = tf.constant([[3, 2, 1], [6, 5, 4]],
                                               dtype=tf.float32)

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
    self.assertAllEqual(
        self.evaluate(scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([2, 5], dtype=tf.float32)))

  def testDirectionNormalization(self):
    for direction in [[1, 2, 2], [0.1, 0.2, 0.2]]:
      scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
          direction, self._hv_params)
      self.assertAllClose(
          self.evaluate(scalarizer(self._batch_multi_objectives)),
          self.evaluate(tf.constant([1.5, 6], dtype=tf.float32)))

  def testCustomTransform(self):
    transform = lambda m, s, o: m * s + o
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        [1, 0, 0], self._hv_params, transform)
    self.assertAllClose(scalarizer(self._batch_multi_objectives), [3, 6])

    transform2 = lambda m, s, o: tf.multiply(m, m) * s + o
    scalarizer2 = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0.1, 0.2, 0.2], self._hv_params, transform2)
    self.assertAllClose(scalarizer2(self._batch_multi_objectives), [1.5, 24.])

    def default_transform(metrics, slopes, offsets):
      transformed_metrics = metrics
      return transformed_metrics * slopes + offsets

    scalarizer3 = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0.1, 0.2, 0.2], self._hv_params, default_transform)
    default_scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0.1, 0.2, 0.2], self._hv_params)
    self.assertAllClose(
        self.evaluate(scalarizer3(self._batch_multi_objectives)),
        self.evaluate(default_scalarizer(self._batch_multi_objectives)))

    # Apply sigmoid to the 2nd metric.
    sigmoid_metric_mask = [False, True, False]
    batch_size = self._batch_multi_objectives.shape[0]
    sigmoid_batch_mask = tf.reshape(
        tf.tile(sigmoid_metric_mask, [batch_size]),
        [batch_size, len(sigmoid_metric_mask)])

    def sigmoid_metric_transform(metrics: tf.Tensor, slopes, offsets):
      transformed_values = tf.where(sigmoid_batch_mask, tf.sigmoid(metrics),
                                    metrics)
      return transformed_values * slopes + offsets

    sigmoid_scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0.1, 0.2, 0.2], self._hv_params, sigmoid_metric_transform)

    self.assertAllClose(
        sigmoid_scalarizer(self._batch_multi_objectives), [1.321196, 1.489961])

    sigmoid_scalarizer2 = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0, 1.0, 0], self._hv_params, sigmoid_metric_transform)
    self.assertAllClose(
        sigmoid_scalarizer2(self._batch_multi_objectives), [0.880797, 0.993307])

  def testNegativeObjectives(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    self.assertAllEqual(
        self.evaluate(scalarizer(-1.0 * self._batch_multi_objectives)),
        self.evaluate(tf.constant([0, 0], dtype=tf.float32)))

  def testObjectiveTranformation(self):
    hv_params = copy.deepcopy(self._hv_params)
    hv_params[0] = multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
        slope=-1, offset=2)
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, hv_params)
    self.assertAllEqual(
        self.evaluate(scalarizer(-1.0 * self._batch_multi_objectives)),
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

  def testSetParameters(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    scalarizer.set_parameters(
        direction=tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32),
        transform_params={
            multi_objective_scalarizer.HyperVolumeScalarizer.SLOPE_KEY:
                tf.constant([[1, 2, 1], [1, 1, 2]], dtype=tf.float32),
            multi_objective_scalarizer.HyperVolumeScalarizer.OFFSET_KEY:
                tf.constant([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
                            dtype=tf.float32)
        })
    self.assertAllClose(
        self.evaluate(scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([4.5, 8.1], dtype=tf.float32)))

  def testSetParametersDirectionOnly(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    scalarizer.set_parameters(
        direction=tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32),
        transform_params={})
    self.assertAllClose(
        self.evaluate(scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([2, 4], dtype=tf.float32)))

  def testSetParametersDirectionAndSlope(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    scalarizer.set_parameters(
        direction=tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32),
        transform_params={
            multi_objective_scalarizer.HyperVolumeScalarizer.SLOPE_KEY:
                tf.constant([[1, 2, 1], [1, 1, 2]], dtype=tf.float32)
        })
    self.assertAllClose(
        self.evaluate(scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([4, 8], dtype=tf.float32)))

  def testSetParametersDirectionAndOffset(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    scalarizer.set_parameters(
        direction=tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32),
        transform_params={
            multi_objective_scalarizer.HyperVolumeScalarizer.OFFSET_KEY:
                tf.constant([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
                            dtype=tf.float32)
        })
    self.assertAllClose(
        self.evaluate(scalarizer(self._batch_multi_objectives)),
        self.evaluate(tf.constant([2.5, 4.1], dtype=tf.float32)))

  def testSetParametersInvalidTransformationParamsKeysRaisesValueError(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(ValueError,
                                'All transform_params keys should be'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32),
          transform_params={
              'weights': tf.constant([1, 1, 1], dtype=tf.float32)
          })

  def testSetParametersDirectionWithWrongNumberOfObjectivesRaisesValueError(
      self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(
        ValueError, 'The number of objectives in scalarization parameter'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1]] * 2, dtype=tf.float32),
          transform_params={})

  def testSetParametersSlopeWithWrongNumberOfObjectivesRaisesValueError(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(
        ValueError, 'The number of objectives in scalarization parameter'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1, 0]] * 2, dtype=tf.float32),
          transform_params={
              multi_objective_scalarizer.HyperVolumeScalarizer.SLOPE_KEY:
                  tf.constant([[1, 2]] * 2, dtype=tf.float32)
          })

  def testSetParametersOffsetWithWrongNumberOfObjectivesRaisesValueError(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(
        ValueError, 'The number of objectives in scalarization parameter'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1, 0]] * 2, dtype=tf.float32),
          transform_params={
              multi_objective_scalarizer.HyperVolumeScalarizer.OFFSET_KEY:
                  tf.constant([[1, 2]] * 2, dtype=tf.float32)
          })

  def testCallDirectionWrongBatchSizeRaisesValueError(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(ValueError, 'does not match the shape of'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                                dtype=tf.float32),
          transform_params={})
      self.evaluate(scalarizer(self._batch_multi_objectives))

  def testCallSlopeWrongBatchSizeRaisesValueError(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(ValueError, 'does not match the shape of'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32),
          transform_params={
              multi_objective_scalarizer.HyperVolumeScalarizer.SLOPE_KEY:
                  tf.constant([[1, 2, 1], [2, 1, 1], [1, 1, 2]],
                              dtype=tf.float32)
          })
      self.evaluate(scalarizer(self._batch_multi_objectives))

  def testCallOffsetWrongBatchSizeRaisesValueError(self):
    direction = [1, 0, 0]
    scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        direction, self._hv_params)
    with self.assertRaisesRegex(ValueError, 'does not match the shape of'):
      scalarizer.set_parameters(
          direction=tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32),
          transform_params={
              multi_objective_scalarizer.HyperVolumeScalarizer.OFFSET_KEY:
                  tf.constant([[1, 2, 1], [2, 1, 1], [1, 1, 2]],
                              dtype=tf.float32)
          })
      self.evaluate(scalarizer(self._batch_multi_objectives))


if __name__ == '__main__':
  tf.test.main()
