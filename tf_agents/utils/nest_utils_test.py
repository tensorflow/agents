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

"""Tests for tf_agents.utils.nest_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

# We use this to build {Dict,Tuple,List}Wrappers for testing nesting code.
from tensorflow.python.training.tracking import data_structures  # pylint: disable=g-direct-tensorflow-import  # TF internal


class NestedTensorsTest(tf.test.TestCase):
  """Tests functions related to nested tensors."""

  def nest_spec(self, shape=(2, 3), dtype=tf.float32, include_sparse=True):
    spec = {
        'tensor_spec_1':
            tensor_spec.TensorSpec(shape, dtype),
        'bounded_spec_1':
            tensor_spec.BoundedTensorSpec(shape, dtype, -10, 10),
        'dict_spec': {
            'tensor_spec_2':
                tensor_spec.TensorSpec(shape, dtype),
            'bounded_spec_2':
                tensor_spec.BoundedTensorSpec(shape, dtype, -10, 10)
        },
        'tuple_spec': (
            tensor_spec.TensorSpec(shape, dtype),
            tensor_spec.BoundedTensorSpec(shape, dtype, -10, 10),
        ),
        'list_spec': [
            tensor_spec.TensorSpec(shape, dtype),
            (tensor_spec.TensorSpec(shape, dtype),
             tensor_spec.BoundedTensorSpec(shape, dtype, -10, 10)),
        ],
        'sparse_tensor_spec': tf.SparseTensorSpec(
            shape=shape, dtype=dtype)
    }
    if not include_sparse:
      del spec['sparse_tensor_spec']
    return spec

  def zeros_from_spec(self, spec, batch_size=None, extra_sizes=None):
    """Return tensors matching spec with desired additional dimensions.

    Args:
      spec: A `tf.TypeSpec`, e.g. `tf.TensorSpec` or `tf.SparseTensorSpec`.
      batch_size: The desired batch size; the size of the first dimension of
        all tensors.
      extra_sizes: An optional list of additional dimension sizes beyond the
        batch_size.

    Returns:
      A possibly nested tuple of Tensors matching the spec.
    """
    tensors = []
    extra_sizes = extra_sizes or []
    for s in tf.nest.flatten(spec):
      if isinstance(s, tf.SparseTensorSpec):
        if batch_size:
          shape = [batch_size] + extra_sizes + s.shape
          rank = 1 + len(extra_sizes) + 2
        else:
          shape = s.shape
          rank = 2
        tensors.append(
            tf.SparseTensor(
                indices=tf.zeros([7, rank], dtype=tf.int64),
                values=tf.zeros([7], dtype=s.dtype),
                dense_shape=tf.constant(shape.as_list(), dtype=tf.int64)))
      elif isinstance(s, tf.TensorSpec):
        if batch_size:
          shape = tf.TensorShape([batch_size] + extra_sizes).concatenate(
              s.shape)
        else:
          shape = s.shape
        tensors.append(tf.zeros(shape, dtype=s.dtype))
      else:
        raise TypeError('Unexpected spec type: {}'.format(s))

    return tf.nest.pack_sequence_as(spec, tensors)

  def placeholders_from_spec(self, spec):
    """Return tensors matching spec with an added unknown batch dimension.

    Args:
      spec: A `tf.TypeSpec`, e.g. `tf.TensorSpec` or `tf.SparseTensorSpec`.

    Returns:
      A possibly nested tuple of Tensors matching the spec.
    """
    tensors = []
    for s in tf.nest.flatten(spec):
      if isinstance(s, tf.SparseTensorSpec):
        raise NotImplementedError(
            'Support for SparseTensor placeholders not implemented.')
      elif isinstance(s, tf.TensorSpec):
        shape = tf.TensorShape([None]).concatenate(s.shape)
        tensors.append(tf.placeholder(dtype=s.dtype, shape=shape))
      else:
        raise TypeError('Unexpected spec type: {}'.format(s))

    return tf.nest.pack_sequence_as(spec, tensors)

  def testGetOuterShapeNotBatched(self):
    tensor = tf.zeros([2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    batch_size = nest_utils.get_outer_shape(tensor, spec)
    self.assertEqual(self.evaluate(batch_size), [])

  def testGetOuterShapeOneDim(self):
    tensor = tf.zeros([5, 2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    batch_size = nest_utils.get_outer_shape(tensor, spec)
    self.assertEqual(self.evaluate(batch_size), [5])

  def testGetOuterShapeTwoDims(self):
    tensor = tf.zeros([7, 5, 2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    batch_dim = nest_utils.get_outer_shape(tensor, spec)
    self.assertAllEqual(self.evaluate(batch_dim), [7, 5])

  def testGetOuterShapeDynamicShapeBatched(self):
    spec = tensor_spec.TensorSpec([1], dtype=tf.float32)
    tensor = tf.convert_to_tensor(value=[[0.0]] * 8)
    batch_size = self.evaluate(nest_utils.get_outer_shape(tensor, spec))
    self.assertAllEqual(batch_size, [8])

  def testGetOuterShapeDynamicShapeNotBatched(self):
    spec = tensor_spec.TensorSpec([None, 1], dtype=tf.float32)
    tensor = tf.convert_to_tensor(value=[[0.0]] * 8)
    batch_size = self.evaluate(nest_utils.get_outer_shape(tensor, spec))
    self.assertEqual(batch_size, [])

  def testGetOuterDimsSingleTensorUnbatched(self):
    tensor = tf.zeros([2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    batch_dims = nest_utils.get_outer_rank(tensor, spec)
    self.assertFalse(batch_dims)

  def testGetOuterDimsSingleTensorBatched(self):
    tensor = tf.zeros([5, 2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    batch_dims = nest_utils.get_outer_rank(tensor, spec)
    self.assertEqual(batch_dims, 1)

  def testGetOuterDimsSpecMismatchUnbatched(self):
    tensor = tf.zeros([1, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    with self.assertRaises(ValueError):
      nest_utils.get_outer_rank(tensor, spec)

  def testGetOuterDimsSpecMismatchBatched(self):
    tensor = tf.zeros([5, 1, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    with self.assertRaises(ValueError):
      nest_utils.get_outer_rank(tensor, spec)

  def testGetOuterDimsNestedTensorsUnbatched(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs)

    batch_dims = nest_utils.get_outer_rank(tensors, specs)
    self.assertFalse(batch_dims)

  def testGetOuterDimsNestedTensorsBatched(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2)

    batch_dims = nest_utils.get_outer_rank(tensors, specs)
    self.assertEqual(batch_dims, 1)

  def testGetOuterDimsNestedTensorsMixed(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2)
    tensors['tensor_spec_1'] = tf.zeros(shape)

    with self.assertRaises(ValueError):
      nest_utils.get_outer_rank(tensors, specs)

  def testGetOuterDimsNestedTensorsMultipleBatchDims(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2, extra_sizes=[2])

    batch_dims = nest_utils.get_outer_rank(tensors, specs)
    self.assertEqual(batch_dims, 2)

  def testGetOuterDimsNestedTensorsMultipleBatchDimsMixed(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2, extra_sizes=[2])

    # Tensors are ok.
    self.assertEqual(nest_utils.get_outer_rank(tensors, specs), 2)
    with self.assertRaises(ValueError):
      tensors['tensor_spec_1'] = tf.zeros_like(tensors['tensor_spec_1'][0])
      # Tensors are not ok.
      nest_utils.get_outer_rank(tensors, specs)

  def testIsBatchedSingleTensorFalse(self):
    tensor = tf.zeros([2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    is_batched = nest_utils.is_batched_nested_tensors(tensor, spec)
    self.assertFalse(is_batched)

  def testIsBatchedSingleTensorTrue(self):
    tensor = tf.zeros([5, 2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    is_batched = nest_utils.is_batched_nested_tensors(tensor, spec)
    self.assertTrue(is_batched)

  def testIsBatchedSingleTensorValueErrorUnBatched(self):
    tensor = tf.zeros([1, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    with self.assertRaises(ValueError):
      nest_utils.is_batched_nested_tensors(tensor, spec)

  def testIsBatchedSingleTensorValueErrorBatched(self):
    tensor = tf.zeros([5, 1, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    with self.assertRaises(ValueError):
      nest_utils.is_batched_nested_tensors(tensor, spec)

  def testIsBatchedNestedTensorsFalse(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs)

    is_batched = nest_utils.is_batched_nested_tensors(tensors, specs)
    self.assertFalse(is_batched)

  def testIsBatchedNestedTensorsTrue(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2)

    is_batched = nest_utils.is_batched_nested_tensors(tensors, specs)
    self.assertTrue(is_batched)

  def testIsBatchedNestedTensorsAllowExtraFields(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2)
    tensors['extra_field'] = tf.constant([1, 2, 3])
    is_batched = nest_utils.is_batched_nested_tensors(
        tensors, specs, allow_extra_fields=True)
    self.assertTrue(is_batched)

  def testIsBatchedNestedTensorsMixed(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2)
    tensors['tensor_spec_1'] = tf.zeros(shape)

    with self.assertRaises(ValueError):
      nest_utils.is_batched_nested_tensors(tensors, specs)

  def testIsBatchedNestedTensorsMultipleBatchDimsFalse(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs)

    is_batched = nest_utils.is_batched_nested_tensors(
        tensors, specs, num_outer_dims=2)
    self.assertFalse(is_batched)

  def testIsBatchedNestedTensorsMultipleBatchDimsTrue(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2, extra_sizes=[2])

    is_batched = nest_utils.is_batched_nested_tensors(
        tensors, specs, num_outer_dims=2)
    self.assertTrue(is_batched)

  def testIsBatchedNestedTensorsMultipleBatchDimsWrongBatchDimNumber(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    # Tensors only have one batch dim.
    tensors = self.zeros_from_spec(specs, batch_size=2)

    is_batched = nest_utils.is_batched_nested_tensors(tensors,
                                                      specs,
                                                      num_outer_dims=2)
    self.assertFalse(is_batched)

  def testIsBatchedNestedTensorsMultipleBatchDimsRightBatchDimNumber(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    # Tensors only have one batch dim.
    tensors = self.zeros_from_spec(specs, batch_size=2, extra_sizes=[1])

    is_batched = nest_utils.is_batched_nested_tensors(tensors,
                                                      specs,
                                                      num_outer_dims=2)
    self.assertTrue(is_batched)

  def testIsBatchedNestedTensorsMultipleBatchDimsMixed(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=2, extra_sizes=[2])

    # Tensors are ok.
    nest_utils.is_batched_nested_tensors(tensors, specs, num_outer_dims=2)
    with self.assertRaises(ValueError):
      tensors['tensor_spec_1'] = tf.zeros_like(tensors['tensor_spec_1'][0])
      # Tensors are not ok.
      nest_utils.is_batched_nested_tensors(tensors, specs, num_outer_dims=2)

  def testBatchSingleTensor(self):
    tensor = tf.zeros([2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)

    batched_tensor = nest_utils.batch_nested_tensors(tensor, spec)

    self.assertEqual(batched_tensor.shape.as_list(), [1, 2, 3])

  def testBatchedSingleTensor(self):
    tensor = tf.zeros([5, 2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)

    batched_tensor = nest_utils.batch_nested_tensors(tensor, spec)

    self.assertEqual(batched_tensor.shape.as_list(), [5, 2, 3])

  def testWrongShapeRaisesValueError(self):
    tensor = tf.zeros([3, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)

    with self.assertRaises(ValueError):
      nest_utils.batch_nested_tensors(tensor, spec)

  def testBatchNestedTensorsNoSpec(self):
    shape = [2, 3]
    batch_shape = [1] + shape
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs)
    tf.nest.assert_same_structure(tensors, specs)

    batched_tensors = nest_utils.batch_nested_tensors(tensors)

    tf.nest.assert_same_structure(specs, batched_tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), batch_shape)
    tf.nest.map_structure(assert_shapes, batched_tensors)

  def testBatchNestedTensors(self):
    shape = [2, 3]
    batch_shape = [1] + shape
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs)
    tf.nest.assert_same_structure(tensors, specs)

    batched_tensors = nest_utils.batch_nested_tensors(tensors, specs)

    tf.nest.assert_same_structure(specs, batched_tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), batch_shape)
    tf.nest.map_structure(assert_shapes, batched_tensors)

  def testBatchedNestedTensors(self):
    shape = [2, 3]
    batch_size = 5
    batch_shape = [batch_size] + shape
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=batch_size)
    tf.nest.assert_same_structure(tensors, specs)

    batched_tensors = nest_utils.batch_nested_tensors(tensors, specs)

    tf.nest.assert_same_structure(specs, batched_tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), batch_shape)
    tf.nest.map_structure(assert_shapes, batched_tensors)

  def testUnBatchSingleTensor(self):
    batched_tensor = tf.zeros([1, 2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)

    tensor = nest_utils.unbatch_nested_tensors(batched_tensor, spec)

    self.assertEqual(tensor.shape.as_list(), [2, 3])

  def testUnBatchedSingleTensor(self):
    tensor = tf.zeros([2, 3], dtype=tf.float32)
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)

    unbatched_tensor = nest_utils.unbatch_nested_tensors(tensor, spec)

    self.assertEqual(unbatched_tensor.shape.as_list(), [2, 3])

  def testUnBatchNestedTensorsNoSpec(self):
    shape = [2, 3]
    batch_size = 1

    specs = self.nest_spec(shape, include_sparse=False)
    batched_tensors = self.zeros_from_spec(specs, batch_size=batch_size)
    tf.nest.assert_same_structure(batched_tensors, specs)

    tensors = nest_utils.unbatch_nested_tensors(batched_tensors)

    tf.nest.assert_same_structure(specs, tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), shape, t)
    tf.nest.map_structure(assert_shapes, tensors)

  def testUnBatchNestedTensors(self):
    shape = [2, 3]
    batch_size = 1

    specs = self.nest_spec(shape, include_sparse=False)
    batched_tensors = self.zeros_from_spec(specs, batch_size=batch_size)
    tf.nest.assert_same_structure(batched_tensors, specs)

    tensors = nest_utils.unbatch_nested_tensors(batched_tensors, specs)

    tf.nest.assert_same_structure(specs, tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), shape, t)
    tf.nest.map_structure(assert_shapes, tensors)

  def testSplitNestedTensors(self):
    shape = [2, 3]
    batch_size = 7

    specs = self.nest_spec(shape, include_sparse=True)
    batched_tensors = self.zeros_from_spec(specs, batch_size=batch_size)
    tf.nest.assert_same_structure(batched_tensors, specs)

    tensors = nest_utils.split_nested_tensors(batched_tensors, specs,
                                              batch_size)
    self.assertEqual(batch_size, len(tensors))

    for t in tensors:
      tf.nest.assert_same_structure(specs, t)

    def assert_shapes(t):
      if not tf.executing_eagerly() and isinstance(t, tf.SparseTensor):
        # Constant value propagation in SparseTensors does not allow us to infer
        # the value of output t.shape from input's t.shape; only its rank.
        self.assertEqual(len(t.shape), 1 + len(shape))
      else:
        self.assertEqual(t.shape.as_list(), [1] + shape)
    tf.nest.map_structure(assert_shapes, tensors)

  def testSplitNestedTensorsSizeSplits(self):
    shape = [2, 3]
    batch_size = 9
    size_splits = [2, 4, 3]

    specs = self.nest_spec(shape, include_sparse=False)
    batched_tensors = self.zeros_from_spec(specs, batch_size=batch_size)
    tf.nest.assert_same_structure(batched_tensors, specs)

    tensors = nest_utils.split_nested_tensors(
        batched_tensors, specs, size_splits)
    self.assertEqual(len(tensors), len(size_splits))

    for i, tensor in enumerate(tensors):
      tf.nest.assert_same_structure(specs, tensor)
      tf.nest.map_structure(
          lambda t: self.assertEqual(t.shape.as_list()[0], size_splits[i]),  # pylint: disable=cell-var-from-loop
          tensor)

    assert_shapes = lambda t: self.assertEqual(t.shape.as_list()[1:], shape)
    tf.nest.map_structure(assert_shapes, tensors)

  def testUnstackNestedTensors(self):
    shape = [5, 8]
    batch_size = 7

    specs = self.nest_spec(shape, include_sparse=False)
    batched_tensors = self.zeros_from_spec(specs, batch_size=batch_size)
    tf.nest.assert_same_structure(batched_tensors, specs)

    tensors = nest_utils.unstack_nested_tensors(batched_tensors, specs)
    self.assertEqual(batch_size, len(tensors))

    for t in tensors:
      tf.nest.assert_same_structure(specs, t)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), shape)
    tf.nest.map_structure(assert_shapes, tensors)

  def testStackNestedTensors(self):
    shape = [5, 8]
    batch_size = 3
    batched_shape = [batch_size,] + shape

    specs = self.nest_spec(shape, include_sparse=False)
    unstacked_tensors = [self.zeros_from_spec(specs) for _ in range(batch_size)]
    stacked_tensor = nest_utils.stack_nested_tensors(unstacked_tensors)

    tf.nest.assert_same_structure(specs, stacked_tensor)
    assert_shapes = lambda tensor: self.assertEqual(tensor.shape, batched_shape)
    tf.nest.map_structure(assert_shapes, stacked_tensor)

  def testStackNestedTensorsAxis1(self):
    shape = [5, 8]
    stack_dim = 3
    stacked_shape = [5, 3, 8]

    specs = self.nest_spec(shape, include_sparse=False)
    unstacked_tensors = [self.zeros_from_spec(specs)] * stack_dim
    stacked_tensor = nest_utils.stack_nested_tensors(unstacked_tensors, axis=1)

    tf.nest.assert_same_structure(specs, stacked_tensor)
    assert_shapes = lambda tensor: self.assertEqual(tensor.shape, stacked_shape)
    tf.nest.map_structure(assert_shapes, stacked_tensor)

  def testUnBatchedNestedTensors(self, include_sparse=False):
    shape = [2, 3]

    specs = self.nest_spec(shape, include_sparse=False)
    unbatched_tensors = self.zeros_from_spec(specs)
    tf.nest.assert_same_structure(unbatched_tensors, specs)

    tensors = nest_utils.unbatch_nested_tensors(unbatched_tensors, specs)

    tf.nest.assert_same_structure(specs, tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), shape, t)
    tf.nest.map_structure(assert_shapes, tensors)

  def testFlattenMultiBatchedSingleTensor(self):
    spec = tensor_spec.TensorSpec([2, 3], dtype=tf.float32)
    tensor = self.zeros_from_spec(spec, batch_size=7, extra_sizes=[5])

    (batch_flattened_tensor,
     batch_dims) = nest_utils.flatten_multi_batched_nested_tensors(tensor, spec)

    self.assertEqual(batch_flattened_tensor.shape.as_list(), [35, 2, 3])

    self.evaluate(tf.compat.v1.global_variables_initializer())
    batch_dims_ = self.evaluate(batch_dims)
    self.assertAllEqual(batch_dims_, [7, 5])

  def testFlattenMultiBatchedNestedTensors(self):
    shape = [2, 3]
    specs = self.nest_spec(shape)
    tensors = self.zeros_from_spec(specs, batch_size=7, extra_sizes=[5])

    (batch_flattened_tensors,
     batch_dims) = nest_utils.flatten_multi_batched_nested_tensors(
         tensors, specs)

    tf.nest.assert_same_structure(specs, batch_flattened_tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), [35, 2, 3])
    tf.nest.map_structure(assert_shapes, batch_flattened_tensors)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    batch_dims_ = self.evaluate(batch_dims)
    self.assertAllEqual(batch_dims_, [7, 5])

  def testFlattenMultiBatchedNestedTensorsWithPartiallyKnownShape(self):
    if tf.executing_eagerly():
      self.skipTest('Do not check nest processing of data in eager mode. '
                    'Placeholders are not compatible with eager execution.')
    shape = [2, 3]
    specs = self.nest_spec(shape, include_sparse=False)
    tensors = self.placeholders_from_spec(specs)

    (batch_flattened_tensors,
     _) = nest_utils.flatten_multi_batched_nested_tensors(
         tensors, specs)

    tf.nest.assert_same_structure(specs, batch_flattened_tensors)
    assert_shapes = lambda t: self.assertEqual(t.shape.as_list(), [None, 2, 3])
    tf.nest.map_structure(assert_shapes, batch_flattened_tensors)


class NestedArraysTest(tf.test.TestCase):
  """Tests functions related to nested arrays."""

  def nest_spec(self, shape=(2, 3), dtype=np.float32):
    return {
        'array_spec_1':
            array_spec.ArraySpec(shape, dtype),
        'bounded_spec_1':
            array_spec.BoundedArraySpec(shape, dtype, -10, 10),
        'dict_spec': {
            'tensor_spec_2':
                array_spec.ArraySpec(shape, dtype),
            'bounded_spec_2':
                array_spec.BoundedArraySpec(shape, dtype, -10, 10)
        },
        'tuple_spec': (
            array_spec.ArraySpec(shape, dtype),
            array_spec.BoundedArraySpec(shape, dtype, -10, 10),
        ),
        'list_spec': [
            array_spec.ArraySpec(shape, dtype),
            (array_spec.ArraySpec(shape, dtype),
             array_spec.BoundedArraySpec(shape, dtype, -10, 10)),
        ],
    }

  def zeros_from_spec(self, specs, outer_dims=None):
    """Return arrays matching spec with desired additional dimensions.

    Args:
      specs: A nested array spec.
      outer_dims: An optional list of outer dimensions, e.g. batch size.

    Returns:
      A nested tuple of arrays matching the spec.
    """
    outer_dims = outer_dims or []

    def _zeros(spec):
      return np.zeros(type(spec.shape)(outer_dims) + spec.shape, spec.dtype)

    return tf.nest.map_structure(_zeros, specs)

  def testUnstackNestedArrays(self):
    shape = (5, 8)
    batch_size = 3

    specs = self.nest_spec(shape)
    batched_arrays = self.zeros_from_spec(specs, outer_dims=[batch_size])
    unbatched_arrays = nest_utils.unstack_nested_arrays(batched_arrays)
    self.assertEqual(batch_size, len(unbatched_arrays))

    for array in unbatched_arrays:
      tf.nest.assert_same_structure(specs, array)
    assert_shapes = lambda a: self.assertEqual(a.shape, shape)
    tf.nest.map_structure(assert_shapes, unbatched_arrays)

  def testUnstackNestedArraysIntoFlatItems(self):
    shape = (5, 8)
    batch_size = 3

    specs = self.nest_spec(shape)
    batched_arrays = self.zeros_from_spec(specs, outer_dims=[batch_size])
    unbatched_flat_items = nest_utils.unstack_nested_arrays_into_flat_items(
        batched_arrays)
    self.assertEqual(batch_size, len(unbatched_flat_items))

    for nested_array, flat_item in zip(
        nest_utils.unstack_nested_arrays(batched_arrays), unbatched_flat_items):
      self.assertAllEqual(flat_item, tf.nest.flatten(nested_array))
      tf.nest.assert_same_structure(specs,
                                    tf.nest.pack_sequence_as(specs, flat_item))
    assert_shapes = lambda a: self.assertEqual(a.shape, shape)
    tf.nest.map_structure(assert_shapes, unbatched_flat_items)

  def testUnstackNestedArray(self):
    shape = (5, 8)
    batch_size = 1

    specs = self.nest_spec(shape)
    batched_arrays = self.zeros_from_spec(specs, outer_dims=[batch_size])
    unbatched_arrays = nest_utils.unstack_nested_arrays(batched_arrays)
    self.assertEqual(batch_size, len(unbatched_arrays))

    for array in unbatched_arrays:
      tf.nest.assert_same_structure(specs, array)
    assert_shapes = lambda a: self.assertEqual(a.shape, shape)
    tf.nest.map_structure(assert_shapes, unbatched_arrays)

  def testStackNestedArrays(self):
    shape = (5, 8)
    batch_size = 3
    batched_shape = (batch_size,) + shape

    specs = self.nest_spec(shape)
    unstacked_arrays = [self.zeros_from_spec(specs) for _ in range(batch_size)]
    stacked_array = nest_utils.stack_nested_arrays(unstacked_arrays)

    tf.nest.assert_same_structure(specs, stacked_array)
    assert_shapes = lambda a: self.assertEqual(a.shape, batched_shape)
    tf.nest.map_structure(assert_shapes, stacked_array)

  def testGetOuterArrayShape(self):
    spec = (
        array_spec.ArraySpec([5, 8], np.float32),
        (array_spec.ArraySpec([1], np.int32),
         array_spec.ArraySpec([2, 2, 2], np.float32))
    )

    batch_size = 3
    unstacked_arrays = [self.zeros_from_spec(spec) for _ in range(batch_size)]

    outer_dims = nest_utils.get_outer_array_shape(unstacked_arrays[0], spec)
    self.assertEqual((), outer_dims)

    stacked_array = nest_utils.stack_nested_arrays(unstacked_arrays)
    outer_dims = nest_utils.get_outer_array_shape(stacked_array, spec)
    self.assertEqual((batch_size,), outer_dims)

    time_dim = [nest_utils.batch_nested_array(arr) for arr in unstacked_arrays]
    batch_time = nest_utils.stack_nested_arrays(time_dim)
    outer_dims = nest_utils.get_outer_array_shape(batch_time, spec)
    self.assertEqual((batch_size, 1), outer_dims)

  def testWhere(self):
    condition = tf.convert_to_tensor([True, False, False, True, False])
    true_output = tf.nest.map_structure(tf.convert_to_tensor,
                                        (np.array([0] * 5), np.arange(1, 6)))
    false_output = tf.nest.map_structure(tf.convert_to_tensor,
                                         (np.array([1] * 5), np.arange(6, 11)))

    result = nest_utils.where(condition, true_output, false_output)
    result = self.evaluate(result)

    expected = (np.array([0, 1, 1, 0, 1]), np.array([1, 7, 8, 4, 10]))
    self.assertAllEqual(expected, result)

  def testWhereDifferentRanks(self):
    condition = tf.convert_to_tensor([True, False, False, True, False])
    true_output = tf.nest.map_structure(
        tf.convert_to_tensor,
        (np.reshape(np.array([0] * 10),
                    (5, 2)), np.reshape(np.arange(1, 11), (5, 2))))
    false_output = tf.nest.map_structure(
        tf.convert_to_tensor,
        (np.reshape(np.array([1] * 10),
                    (5, 2)), np.reshape(np.arange(12, 22), (5, 2))))

    result = nest_utils.where(condition, true_output, false_output)
    result = self.evaluate(result)

    expected = (np.array([[0, 0], [1, 1], [1, 1], [0, 0], [1, 1]]),
                np.array([[1, 2], [14, 15], [16, 17], [7, 8], [20, 21]]))
    self.assertAllEqual(expected, result)

  def testWhereSameRankDifferentDimension(self):
    condition = tf.convert_to_tensor([True, False, True])
    true_output = (tf.convert_to_tensor([1]), tf.convert_to_tensor([2]))
    false_output = (tf.convert_to_tensor([3, 4, 5]),
                    tf.convert_to_tensor([6, 7, 8]))

    result = nest_utils.where(condition, true_output, false_output)
    result = self.evaluate(result)

    expected = (np.array([1, 4, 1]), np.array([2, 7, 2]))
    self.assertAllEqual(expected, result)


class PruneExtraKeysTest(tf.test.TestCase):

  def testPruneExtraKeys(self):
    self.assertEqual(nest_utils.prune_extra_keys({}, {'a': 1}), {})
    self.assertEqual(nest_utils.prune_extra_keys(
        {'a': 1}, {'a': 'a'}), {'a': 'a'})
    self.assertEqual(
        nest_utils.prune_extra_keys({'a': 1}, {'a': 'a', 'b': 2}), {'a': 'a'})
    self.assertEqual(
        nest_utils.prune_extra_keys([{'a': 1}], [{'a': 'a', 'b': 2}]),
        [{'a': 'a'}])
    self.assertEqual(
        nest_utils.prune_extra_keys(
            {'a': {'aa': 1, 'ab': 2}, 'b': {'ba': 1}},
            {'a': {'aa': 'aa', 'ab': 'ab', 'ac': 'ac'},
             'b': {'ba': 'ba', 'bb': 'bb'},
             'c': 'c'}),
        {'a': {'aa': 'aa', 'ab': 'ab'}, 'b': {'ba': 'ba'}})

  def testInvalidWide(self):
    self.assertEqual(nest_utils.prune_extra_keys(None, {'a': 1}), {'a': 1})
    self.assertEqual(nest_utils.prune_extra_keys({'a': 1}, {}), {})
    self.assertEqual(nest_utils.prune_extra_keys(
        {'a': 1}, {'c': 'c'}), {'c': 'c'})
    self.assertEqual(nest_utils.prune_extra_keys([], ['a']), ['a'])
    self.assertEqual(
        nest_utils.prune_extra_keys([{}, {}], [{'a': 1}]), [{'a': 1}])

  def testNamedTuple(self):

    class A(collections.namedtuple('A', ('a', 'b'))):
      pass

    self.assertEqual(
        nest_utils.prune_extra_keys(
            [A(a={'aa': 1}, b=3), {'c': 4}],
            [A(a={'aa': 'aa', 'ab': 'ab'}, b='b'), {'c': 'c', 'd': 'd'}]),
        [A(a={'aa': 'aa'}, b='b'), {'c': 'c'}])

  def testSubtypesOfListAndDict(self):

    class A(collections.namedtuple('A', ('a', 'b'))):
      pass

    # pylint: disable=invalid-name
    DictWrapper = data_structures.wrap_or_unwrap
    TupleWrapper = data_structures.wrap_or_unwrap
    # pylint: enable=invalid-name

    self.assertEqual(
        nest_utils.prune_extra_keys(
            [data_structures.ListWrapper([None, DictWrapper({'a': 3, 'b': 4})]),
             None,
             TupleWrapper((DictWrapper({'g': 5}),)),
             TupleWrapper(A(None, DictWrapper({'h': 6}))),
            ],
            [['x', {'a': 'a', 'b': 'b', 'c': 'c'}],
             'd',
             ({'g': 'g', 'gg': 'gg'},),
             A(None, {'h': 'h', 'hh': 'hh'}),
            ]),
        [data_structures.ListWrapper([
            'x', DictWrapper({'a': 'a', 'b': 'b'})]),
         'd',
         TupleWrapper((DictWrapper({'g': 'g'}),)),
         TupleWrapper(A(None, DictWrapper({'h': 'h'}),)),
        ])

  def testOrderedDict(self):
    OD = collections.OrderedDict  # pylint: disable=invalid-name

    self.assertEqual(
        nest_utils.prune_extra_keys(
            OD([('a', OD([('aa', 1), ('ab', 2)])),
                ('b', OD([('ba', 1)]))]),
            OD([('a', OD([('aa', 'aa'), ('ab', 'ab'), ('ac', 'ac')])),
                ('b', OD([('ba', 'ba'), ('bb', 'bb')])),
                ('c', 'c')])),
        OD([('a', OD([('aa', 'aa'), ('ab', 'ab')])),
            ('b', OD([('ba', 'ba')]))])
    )


if __name__ == '__main__':
  tf.test.main()
