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

"""Tests for environments.specs.tensor_spec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.environments import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
nest = tf.contrib.framework.nest

TYPE_PARAMETERS = (
    ("tf.int32", tf.int32),
    ("tf.int64", tf.int64),
    ("tf.float32", tf.float32),
    ("tf.float64", tf.float64),
    ("tf.uint8", tf.uint8),)


def example_nested_array_spec(dtype):
  return {
      "spec_1":
          array_spec.ArraySpec((2, 3), dtype),
      "bounded_spec_1":
          array_spec.BoundedArraySpec((2, 3), dtype, -10, 10),
      "bounded_array_spec_3":
          array_spec.BoundedArraySpec((2,), dtype, [-10, -10], [10, 10]),
      "dict_spec": {
          "spec_2":
              array_spec.ArraySpec((2, 3), dtype),
          "bounded_spec_2":
              array_spec.BoundedArraySpec((2, 3), dtype, -10, 10)
      },
      "tuple_spec": (
          array_spec.ArraySpec((2, 3), dtype),
          array_spec.BoundedArraySpec((2, 3), dtype, -10, 10),
      ),
      "list_spec": [
          array_spec.ArraySpec((2, 3), dtype),
          (array_spec.ArraySpec((2, 3), dtype),
           array_spec.BoundedArraySpec((2, 3), dtype, -10, 10)),
      ],
  }


def example_nested_tensor_spec(dtype):
  minval = 0 if dtype == tf.uint8 else -10
  maxval = 255 if dtype == tf.uint8 else 10
  return {
      "spec_1":
          tensor_spec.TensorSpec((2, 3), dtype),
      "bounded_spec_1":
          tensor_spec.BoundedTensorSpec((2, 3), dtype, minval, maxval),
      "bounded_array_spec_3":
          tensor_spec.BoundedTensorSpec((2), dtype, [minval, minval],
                                        [maxval, maxval]),
      "dict_spec": {
          "spec_2":
              tensor_spec.TensorSpec((2, 3), dtype),
          "bounded_spec_2":
              tensor_spec.BoundedTensorSpec((2, 3), dtype, minval, maxval)
      },
      "tuple_spec": (
          tensor_spec.TensorSpec((2, 3), dtype),
          tensor_spec.BoundedTensorSpec((2, 3), dtype, minval, maxval),
      ),
      "list_spec": [
          tensor_spec.TensorSpec((2, 3), dtype),
          (tensor_spec.TensorSpec((2, 3), dtype),
           tensor_spec.BoundedTensorSpec((2, 3), dtype, minval, maxval)),
      ],
  }


class TensorSpecTest(tf.test.TestCase):

  def testAcceptsNumpyDType(self):
    desc = tensor_spec.TensorSpec([1], np.float32)
    self.assertEqual(desc.dtype, tf.float32)

  def testWorksWithStrType(self):
    # tf.as_dtype does not handle numpy string_/unicode_ type objects and
    # requires dtype objects with type string_/unicode_.
    desc = tensor_spec.TensorSpec([1], np.dtype(np.string_))
    self.assertEqual(desc.dtype, tf.string)

    desc = tensor_spec.TensorSpec([1], tf.string)
    self.assertEqual(desc.dtype, tf.string)

  def testAcceptsTensorShape(self):
    desc = tensor_spec.TensorSpec(tf.TensorShape([1]), tf.float32)
    self.assertEqual(desc.shape, tf.TensorShape([1]))

  def testUnknownShape(self):
    desc = tensor_spec.TensorSpec(shape=None, dtype=tf.float32)
    self.assertEqual(desc.shape, tf.TensorShape(None))

  def testShapeCompatibility(self):
    unknown = tf.placeholder(tf.int64)
    partial = tf.placeholder(tf.int64, shape=[None, 1])
    full = tf.placeholder(tf.int64, shape=[2, 3])
    rank3 = tf.placeholder(tf.int64, shape=[4, 5, 6])

    desc_unknown = tensor_spec.TensorSpec(None, tf.int64)
    self.assertTrue(desc_unknown.is_compatible_with(unknown))
    self.assertTrue(desc_unknown.is_compatible_with(partial))
    self.assertTrue(desc_unknown.is_compatible_with(full))
    self.assertTrue(desc_unknown.is_compatible_with(rank3))

    desc_partial = tensor_spec.TensorSpec([2, None], tf.int64)
    self.assertTrue(desc_partial.is_compatible_with(unknown))
    self.assertTrue(desc_partial.is_compatible_with(partial))
    self.assertTrue(desc_partial.is_compatible_with(full))
    self.assertFalse(desc_partial.is_compatible_with(rank3))

    desc_full = tensor_spec.TensorSpec([2, 3], tf.int64)
    self.assertTrue(desc_full.is_compatible_with(unknown))
    self.assertFalse(desc_full.is_compatible_with(partial))
    self.assertTrue(desc_full.is_compatible_with(full))
    self.assertFalse(desc_full.is_compatible_with(rank3))

    desc_rank3 = tensor_spec.TensorSpec([4, 5, 6], tf.int64)
    self.assertTrue(desc_rank3.is_compatible_with(unknown))
    self.assertFalse(desc_rank3.is_compatible_with(partial))
    self.assertFalse(desc_rank3.is_compatible_with(full))
    self.assertTrue(desc_rank3.is_compatible_with(rank3))

  def testTypeCompatibility(self):
    floats = tf.placeholder(tf.float32, shape=[10, 10])
    ints = tf.placeholder(tf.int32, shape=[10, 10])
    desc = tensor_spec.TensorSpec(shape=(10, 10), dtype=tf.float32)
    self.assertTrue(desc.is_compatible_with(floats))
    self.assertFalse(desc.is_compatible_with(ints))

  def testName(self):
    desc = tensor_spec.TensorSpec([1], tf.float32, name="beep")
    self.assertEqual(desc.name, "beep")

  def testRepr(self):
    desc1 = tensor_spec.TensorSpec([1], tf.float32, name="beep")
    self.assertEqual(
        repr(desc1),
        "TensorSpec(shape=(1,), dtype=tf.float32, name='beep')")
    desc2 = tensor_spec.TensorSpec([1, None], tf.int32)
    self.assertEqual(
        repr(desc2),
        "TensorSpec(shape=(1, ?), dtype=tf.int32, name=None)")

  def testFromTensorSpec(self):
    spec_1 = tensor_spec.TensorSpec((1, 2), tf.int32)
    spec_2 = tensor_spec.TensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

  def testFromTensor(self):
    zero = tf.constant(0)
    spec = tensor_spec.TensorSpec.from_tensor(zero)
    self.assertEqual(spec.dtype, tf.int32)
    self.assertEqual(spec.shape, [])
    self.assertEqual(spec.name, "Const")

  def testFromPlaceholder(self):
    unknown = tf.placeholder(tf.int64, name="unknown")
    partial = tf.placeholder(tf.float32, shape=[None, 1], name="partial")
    spec_1 = tensor_spec.TensorSpec.from_tensor(unknown)
    self.assertEqual(spec_1.dtype, tf.int64)
    self.assertEqual(spec_1.shape, None)
    self.assertEqual(spec_1.name, "unknown")
    spec_2 = tensor_spec.TensorSpec.from_tensor(partial)
    self.assertEqual(spec_2.dtype, tf.float32)
    self.assertEqual(spec_2.shape.as_list(), [None, 1])
    self.assertEqual(spec_2.name, "partial")

  def testFromBoundedTensorSpec(self):
    bounded_spec = tensor_spec.BoundedTensorSpec((1, 2), tf.int32, 0, 1)
    spec = tensor_spec.TensorSpec.from_spec(bounded_spec)
    self.assertEqual(bounded_spec.shape, spec.shape)
    self.assertEqual(bounded_spec.dtype, spec.dtype)
    self.assertEqual(bounded_spec.name, spec.name)
    self.assertTrue(isinstance(spec, tensor_spec.TensorSpec))

  def testIsDiscrete(self):
    discrete_spec = tensor_spec.TensorSpec((1, 2), tf.int32)
    continuous_spec = tensor_spec.TensorSpec((1, 2), tf.float32)
    self.assertTrue(discrete_spec.is_discrete())
    self.assertFalse(continuous_spec.is_discrete())


class BoundedTensorSpecTest(tf.test.TestCase):

  def testInvalidMinimum(self):
    with self.assertRaisesRegexp(ValueError, "not compatible"):
      tensor_spec.BoundedTensorSpec((3, 5), tf.uint8, (0, 0, 0), (1, 1))

  def testInvalidMaximum(self):
    with self.assertRaisesRegexp(ValueError, "not compatible"):
      tensor_spec.BoundedTensorSpec((3, 5), tf.uint8, 0, (1, 1, 1))

  def testMinimumMaximumAttributes(self):
    spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
    self.assertEqual(type(spec.minimum), np.ndarray)
    self.assertEqual(type(spec.maximum), np.ndarray)
    self.assertAllEqual(spec.minimum, np.array(0, dtype=np.float32))
    self.assertAllEqual(spec.maximum, np.array([5, 5, 5], dtype=np.float32))

  def testNotWriteableNP(self):
    spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
    with self.assertRaisesRegexp(ValueError, "read-only"):
      spec.minimum[0] = -1
    with self.assertRaisesRegexp(ValueError, "read-only"):
      spec.maximum[0] = 100

  def testReuseSpec(self):
    spec_1 = tensor_spec.BoundedTensorSpec((1, 2), tf.int32,
                                           minimum=0, maximum=1)
    spec_2 = tensor_spec.BoundedTensorSpec(
        spec_1.shape, spec_1.dtype, spec_1.minimum, spec_1.maximum)
    self.assertEqual(spec_1, spec_2)

  def testScalarBounds(self):
    spec = tensor_spec.BoundedTensorSpec(
        (), tf.float32, minimum=0.0, maximum=1.0)

    self.assertIsInstance(spec.minimum, np.ndarray)
    self.assertIsInstance(spec.maximum, np.ndarray)

    # Sanity check that numpy compares correctly to a scalar for an empty shape.
    self.assertEqual(0.0, spec.minimum)
    self.assertEqual(1.0, spec.maximum)

    # Check that the spec doesn't fail its own input validation.
    _ = tensor_spec.BoundedTensorSpec(
        spec.shape, spec.dtype, spec.minimum, spec.maximum)

  def testFromBoundedTensorSpec(self):
    spec_1 = tensor_spec.BoundedTensorSpec((1, 2), tf.int32,
                                           minimum=0, maximum=1)
    spec_2 = tensor_spec.BoundedTensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

  def testEquality(self):
    spec_1_1 = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0,
                                             (5, 5, 5))
    spec_1_2 = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0.00000001,
                                             (5, 5, 5.00000000000000001))
    spec_2_1 = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 1,
                                             (5, 5, 5))
    spec_2_2 = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, (1, 1, 1),
                                             (5, 5, 5))
    spec_2_3 = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, (1, 1, 1),
                                             5)
    spec_3_1 = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, (2, 1, 1),
                                             (5, 5, 5))

    self.assertEqual(spec_1_1, spec_1_2)
    self.assertEqual(spec_1_2, spec_1_1)

    self.assertNotEqual(spec_1_1, spec_2_2)
    self.assertNotEqual(spec_1_1, spec_2_1)
    self.assertNotEqual(spec_2_2, spec_1_1)
    self.assertNotEqual(spec_2_1, spec_1_1)

    self.assertEqual(spec_2_1, spec_2_2)
    self.assertEqual(spec_2_2, spec_2_1)
    self.assertEqual(spec_2_2, spec_2_3)

    self.assertNotEqual(spec_1_1, spec_3_1)
    self.assertNotEqual(spec_2_1, spec_3_1)
    self.assertNotEqual(spec_2_2, spec_3_1)

  def testFromTensorSpec(self):
    spec = tensor_spec.TensorSpec((1, 2), tf.int32)
    bounded_spec = tensor_spec.BoundedTensorSpec.from_spec(spec)
    self.assertEqual(spec.shape, bounded_spec.shape)
    self.assertEqual(spec.dtype, bounded_spec.dtype)
    self.assertEqual(spec.dtype.min, bounded_spec.minimum)
    self.assertEqual(spec.dtype.max, bounded_spec.maximum)
    self.assertEqual(spec.name, bounded_spec.name)
    self.assertEquals(type(bounded_spec), tensor_spec.BoundedTensorSpec)

  def testSerialization(self):
    desc = tensor_spec.BoundedTensorSpec([1, 5], tf.float32, -1, 1, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  def testUint8IncludeMaxOfDtype(self):
    spec = tensor_spec.BoundedTensorSpec((2, 3), tf.uint8, 255, 255)
    sample = tensor_spec.sample_spec_nest(spec)
    sample_ = self.evaluate(sample)
    self.assertTrue(np.all(sample_ == 255))


@parameterized.named_parameters(*TYPE_PARAMETERS)
class BoundedTensorSpecSampleTest(tf.test.TestCase, parameterized.TestCase):

  def testIntegerSamplesIncludeUpperBound(self, dtype):
    if not dtype.is_integer:  # Only test on integer dtypes.
      return
    spec = tensor_spec.BoundedTensorSpec((2, 3), dtype, 3, 3)
    sample = tensor_spec.sample_spec_nest(spec)
    sample_ = self.evaluate(sample)
    self.assertEqual(sample_.shape, (2, 3))
    self.assertTrue(np.all(sample_ == 3))

  def testIntegerSamplesExcludeMaxOfDtype(self, dtype):
    # Exclude non integer types and uint8 (has special sampling logic).
    if not dtype.is_integer or dtype == tf.uint8:
      return
    spec = tensor_spec.BoundedTensorSpec((2, 3), dtype, dtype.max-1, dtype.max)
    sample = tensor_spec.sample_spec_nest(spec)
    sample_ = self.evaluate(sample)
    self.assertEqual(sample_.shape, (2, 3))
    self.assertTrue(np.all(sample_ == dtype.max-1))

  def testSampleWithArrayInBounds(self, dtype):
    spec = tensor_spec.BoundedTensorSpec((2, 3), dtype, (0, 0, 0), 3)
    sample = tensor_spec.sample_spec_nest(spec)
    self.assertEqual((2, 3), sample.shape)
    sample_ = self.evaluate(sample)
    self.assertEqual((2, 3), sample_.shape)
    self.assertTrue(np.all(sample_ <= 3))
    self.assertTrue(np.all(0 <= sample_))

  def testTensorSpecSample(self, dtype):
    spec = tensor_spec.TensorSpec((2, 3), dtype)
    sample = tensor_spec.sample_spec_nest(spec)
    bounded = tensor_spec.BoundedTensorSpec.from_spec(spec)

    sample_ = self.evaluate(sample)
    self.assertTrue(np.all(sample_ >= bounded.minimum),
                    (sample_.min(), sample_.max()))
    self.assertTrue(np.all(sample_ <= bounded.maximum),
                    (sample_.min(), sample_.max()))

  def testBoundedTensorSpecSample(self, dtype):
    spec = tensor_spec.BoundedTensorSpec((2, 3), dtype, 2, 7)
    sample = tensor_spec.sample_spec_nest(spec)
    sample_ = self.evaluate(sample)
    self.assertTrue(np.all(sample_ >= 2))
    self.assertTrue(np.all(sample_ <= 7))

  def testNestSample(self, dtype):
    nested_spec = example_nested_tensor_spec(dtype)
    sample = tensor_spec.sample_spec_nest(nested_spec)
    spec_1 = tensor_spec.BoundedTensorSpec.from_spec(nested_spec["spec_1"])
    bounded_spec_1 = nested_spec["bounded_spec_1"]
    sample_ = self.evaluate(sample)
    self.assertTrue(np.all(sample_["spec_1"] >= spec_1.minimum))
    self.assertTrue(np.all(sample_["spec_1"] <= spec_1.maximum))

    self.assertTrue(np.all(sample_["bounded_spec_1"] >= bounded_spec_1.minimum))
    self.assertTrue(np.all(sample_["bounded_spec_1"] <= bounded_spec_1.maximum))

    self.assertIn("spec_2", sample_["dict_spec"])
    tensor_spec_2 = sample_["dict_spec"]["spec_2"]
    self.assertTrue(np.all(tensor_spec_2 >= spec_1.minimum))
    self.assertTrue(np.all(tensor_spec_2 <= spec_1.maximum))
    self.assertIn("bounded_spec_2", sample_["dict_spec"])
    sampled_bounded_spec_2 = sample_["dict_spec"]["bounded_spec_2"]
    self.assertTrue(np.all(sampled_bounded_spec_2 >= spec_1.minimum))
    self.assertTrue(np.all(sampled_bounded_spec_2 <= spec_1.maximum))

    self.assertIn("tuple_spec", sample_)
    self.assertTrue(np.all(sample_["tuple_spec"][0] >= spec_1.minimum))
    self.assertTrue(np.all(sample_["tuple_spec"][0] <= spec_1.maximum))
    self.assertTrue(np.all(sample_["tuple_spec"][1] >= bounded_spec_1.minimum))
    self.assertTrue(np.all(sample_["tuple_spec"][1] <= bounded_spec_1.maximum))

    self.assertIn("list_spec", sample_)
    self.assertTrue(np.all(sample_["list_spec"][0] >= spec_1.minimum))
    self.assertTrue(np.all(sample_["list_spec"][0] <= spec_1.maximum))
    self.assertTrue(np.all(sample_["list_spec"][1][0] >= spec_1.minimum))
    self.assertTrue(np.all(sample_["list_spec"][1][0] <= spec_1.maximum))
    self.assertTrue(
        np.all(sample_["list_spec"][1][1] >= bounded_spec_1.minimum))
    self.assertTrue(
        np.all(sample_["list_spec"][1][1] <= bounded_spec_1.maximum))

  def testNestSampleOuterDims(self, dtype):
    # Can't add another level of parameterized args because the test class is
    # already parameterized on dtype.
    self._testNestSampleOuterDims(dtype, use_tensor=False)
    self._testNestSampleOuterDims(dtype, use_tensor=True)

  def _testNestSampleOuterDims(self, dtype, use_tensor):
    nested_spec = example_nested_tensor_spec(dtype)
    if use_tensor:
      outer_dims = tf.constant([2, 3], dtype=tf.int32)
    else:
      outer_dims = (2, 3)
    sample = tensor_spec.sample_spec_nest(nested_spec, outer_dims=outer_dims)
    bounded = tensor_spec.BoundedTensorSpec.from_spec(nested_spec["spec_1"])
    sample_ = self.evaluate(sample)
    self.assertEqual((2, 3) + tuple(nested_spec["spec_1"].shape.as_list()),
                     sample_["spec_1"].shape)
    self.assertTrue(np.all(sample_["spec_1"] >= bounded.minimum))
    self.assertTrue(np.all(sample_["spec_1"] <= bounded.maximum))

    bounded_spec_1 = nested_spec["bounded_spec_1"]
    self.assertEqual((2, 3) + tuple(bounded_spec_1.shape.as_list()),
                     sample_["bounded_spec_1"].shape)
    self.assertTrue(np.all(sample_["bounded_spec_1"] >= bounded_spec_1.minimum))
    self.assertTrue(np.all(sample_["bounded_spec_1"] <= bounded_spec_1.maximum))

    self.assertIn("spec_2", sample_["dict_spec"])
    tensor_spec_2 = sample_["dict_spec"]["spec_2"]
    self.assertEqual(
        (2, 3) + tuple(nested_spec["dict_spec"]["spec_2"].shape.as_list()),
        tensor_spec_2.shape)
    self.assertTrue(np.all(tensor_spec_2 >= bounded.minimum))
    self.assertTrue(np.all(tensor_spec_2 <= bounded.maximum))
    self.assertIn("bounded_spec_2", sample_["dict_spec"])
    sampled_bounded_spec_2 = sample_["dict_spec"]["bounded_spec_2"]
    self.assertEqual(
        (2, 3) + tuple(nested_spec["dict_spec"]["bounded_spec_2"]
                       .shape.as_list()),
        sampled_bounded_spec_2.shape)
    self.assertTrue(np.all(sampled_bounded_spec_2 >= bounded.minimum))
    self.assertTrue(np.all(sampled_bounded_spec_2 <= bounded.maximum))

    self.assertIn("tuple_spec", sample_)
    self.assertEqual(
        (2, 3) + tuple(nested_spec["tuple_spec"][0].shape.as_list()),
        sample_["tuple_spec"][0].shape)
    self.assertTrue(np.all(sample_["tuple_spec"][0] >= bounded.minimum))
    self.assertTrue(np.all(sample_["tuple_spec"][0] <= bounded.maximum))
    tuple_bounded_spec = nested_spec["tuple_spec"][1]
    self.assertEqual((2, 3) + tuple(tuple_bounded_spec.shape.as_list()),
                     sample_["tuple_spec"][1].shape)
    self.assertTrue(
        np.all(sample_["tuple_spec"][1] >= tuple_bounded_spec.minimum))
    self.assertTrue(
        np.all(sample_["tuple_spec"][1] <= tuple_bounded_spec.maximum))

    self.assertIn("list_spec", sample_)
    self.assertEqual(
        (2, 3) + tuple(nested_spec["list_spec"][0].shape.as_list()),
        sample_["list_spec"][0].shape)
    self.assertTrue(np.all(sample_["list_spec"][0] >= bounded.minimum))
    self.assertTrue(np.all(sample_["list_spec"][0] <= bounded.maximum))
    self.assertEqual(
        (2, 3) + tuple(nested_spec["list_spec"][1][0].shape.as_list()),
        sample_["list_spec"][1][0].shape)
    self.assertTrue(np.all(sample_["list_spec"][1][0] >= bounded.minimum))
    self.assertTrue(np.all(sample_["list_spec"][1][0] <= bounded.maximum))
    list_bounded_spec = nested_spec["list_spec"][1][1]
    self.assertTrue(
        np.all(sample_["list_spec"][1][1] >= list_bounded_spec.minimum))
    self.assertTrue(
        np.all(sample_["list_spec"][1][1] <= list_bounded_spec.maximum))

    def _test_batched_shape(sample_, spec_):
      self.assertSequenceEqual(sample_.shape, outer_dims + tuple(spec_.shape))
      nest.map_structure(_test_batched_shape, sample, nested_spec)


@parameterized.named_parameters(*TYPE_PARAMETERS)
class TensorSpecTypeTest(tf.test.TestCase, parameterized.TestCase):

  def testIsDiscrete(self, dtype):
    spec = tensor_spec.TensorSpec((2, 3), dtype=dtype)
    self.assertIs(spec.is_discrete(), dtype.is_integer)

  def testIsContinuous(self, dtype):
    spec = tensor_spec.TensorSpec((2, 3), dtype=dtype)
    self.assertIs(spec.is_continuous(), dtype.is_floating)

  def testExclusive(self, dtype):
    spec = tensor_spec.TensorSpec((2, 3), dtype=dtype)
    self.assertIs(spec.is_discrete() ^ spec.is_continuous(), True)


class FromSpecTest(tf.test.TestCase):

  def testFromSpec(self):
    example_array_spec = example_nested_array_spec(np.int32)
    example_tensor_spec = tensor_spec.from_spec(example_array_spec)

    flat_tensor_spec = nest.flatten(example_tensor_spec)
    expected_tensor_spec = nest.flatten(example_nested_tensor_spec(tf.int32))

    for spec, expected_spec in zip(flat_tensor_spec, expected_tensor_spec):
      self.assertEqual(type(expected_spec), type(spec))
      self.assertEqual(expected_spec, spec)

  def testFromStringSpec(self):
    spec = tensor_spec.from_spec(array_spec.ArraySpec([1], np.string_))
    self.assertEqual(tf.string, spec.dtype)


class ToPlaceholderTest(tf.test.TestCase):

  def testCreatePlaceholderFromTuple(self):
    specs = (
        tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name="act_prob"),
        tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name="value_pred")
    )
    ph = tensor_spec.to_nest_placeholder(specs)
    self.assertEqual(2, len(ph))
    self.assertEqual(ph[0].name, "act_prob:0")
    self.assertEqual(ph[0].dtype, tf.float32)
    self.assertEqual(ph[0].shape, tf.TensorShape([]))
    self.assertEqual(ph[1].name, "value_pred:0")
    self.assertEqual(ph[1].dtype, tf.float32)
    self.assertEqual(ph[1].shape, tf.TensorShape([]))

  def testCreatePlaceholderFromTimeStepSpec(self):
    obs_spec = tensor_spec.TensorSpec([2], tf.float32, "obs")
    time_step_spec = ts.time_step_spec(obs_spec)
    ph = tensor_spec.to_nest_placeholder(time_step_spec)
    self.assertIsInstance(ph, ts.TimeStep)
    self.assertEqual(ph.observation.name, "obs:0")
    self.assertEqual(ph.observation.dtype, tf.float32)
    self.assertEqual(ph.observation.shape, tf.TensorShape([2]))

  def testCreatePlaceholderWithNameScope(self):
    obs_spec = tensor_spec.TensorSpec([2], tf.float32, "obs")
    time_step_spec = ts.time_step_spec(obs_spec)
    ph = tensor_spec.to_nest_placeholder(
        time_step_spec, name_scope="action")
    self.assertEqual(ph.observation.name, "action/obs:0")


if __name__ == "__main__":
  tf.test.main()
