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

"""Tests for tf_agents.environments.specs.array_spec."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec


TYPE_PARAMETERS = (
    ("np.int8", np.int8),
    ("np.int16", np.int16),
    ("np.int32", np.int32),
    ("np.int64", np.int64),
    ("np.float16", np.float16),
    ("np.float32", np.float32),
    ("np.float64", np.float64),
    ("python float", float),
    ("python int", int),
)


def example_basic_spec():
  return array_spec.ArraySpec((1,), np.int64)


def example_nested_spec(dtype):
  """Return an example nested array spec."""
  return {
      "array_spec_1":
          array_spec.ArraySpec((2, 3), dtype),
      "bounded_spec_1":
          array_spec.BoundedArraySpec((2, 3), dtype, -10, 10),
      "dict_spec": {
          "array_spec_2": array_spec.ArraySpec((2, 3), dtype),
          "bounded_spec_2": array_spec.BoundedArraySpec((2, 3), dtype, -10, 10)
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


# These parameters will be used with every test* method in this class.
@parameterized.named_parameters(*TYPE_PARAMETERS)
class ArraySpecNestSampleTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState()
    return super(ArraySpecNestSampleTest, self).setUp()

  def testArraySpecSample(self, dtype):
    spec = array_spec.ArraySpec((2, 3), dtype)
    sample = array_spec.sample_spec_nest(spec, self.rng)

    bounded = array_spec.BoundedArraySpec.from_spec(spec)
    self.assertTrue(np.all(sample >= bounded.minimum))
    self.assertTrue(np.all(sample <= bounded.maximum))

  def testArraySpecSampleWithName(self, dtype):
    spec = array_spec.ArraySpec((2, 3), dtype, name="test_spec")
    sample = array_spec.sample_spec_nest(spec, self.rng)

    bounded = array_spec.BoundedArraySpec.from_spec(spec)
    self.assertTrue(np.all(sample >= bounded.minimum))
    self.assertTrue(np.all(sample <= bounded.maximum))
    self.assertEqual("test_spec", bounded.name)

  def testBoundedArraySpecSample(self, dtype):
    spec = array_spec.BoundedArraySpec((2, 3), dtype, -10, 10)
    sample = array_spec.sample_spec_nest(spec, self.rng)
    self.assertTrue(np.all(sample >= -10))
    self.assertTrue(np.all(sample <= 10))

  def testBoundedArraySpecSampleMultipleBounds(self, dtype):
    spec = array_spec.BoundedArraySpec((2,), dtype, [-10, 1], [10, 3])
    sample = array_spec.sample_spec_nest(spec, self.rng)
    self.assertGreaterEqual(sample[0], -10)
    self.assertLessEqual(sample[0], 10)
    self.assertGreaterEqual(sample[1], 1)
    self.assertLessEqual(sample[1], 3)

  def testBoundedArraySpecNoBounds(self, dtype):
    spec = array_spec.ArraySpec((2, 3), dtype)
    bounded_spec = array_spec.BoundedArraySpec.from_spec(spec)
    sample = array_spec.sample_spec_nest(bounded_spec, self.rng)
    tf_dtype = tf.as_dtype(spec.dtype)
    self.assertTrue(np.all(sample >= tf_dtype.min))
    self.assertTrue(np.all(sample <= tf_dtype.max))

  def testSampleTensorBoundedSpecFromArraySpecNoBounds(self, dtype):
    if dtype in [int, float]:
      return

    tf_dtype = tf.as_dtype(dtype)

    # Skip unsupported random_ops dtypes.
    # TODO(b/68706911): Add tf.float16 once bug is fixed.
    if tf_dtype not in (tf.bfloat16, tf.float32, tf.float64, tf.int32,
                        tf.int64):
      return

    spec = array_spec.ArraySpec((2, 3), dtype)
    bounded_spec = array_spec.BoundedArraySpec.from_spec(spec)

    t_spec = tensor_spec.BoundedTensorSpec.from_spec(bounded_spec)
    sample = tensor_spec.sample_spec_nest(t_spec)
    bounded = tensor_spec.BoundedTensorSpec.from_spec(t_spec)

    sample_ = self.evaluate(sample)
    self.assertTrue(
        np.all(sample_ >= bounded.minimum), (sample_.min(), sample_.max()))
    self.assertTrue(
        np.all(sample_ <= bounded.maximum), (sample_.min(), sample_.max()))

  def testNestSample(self, dtype):
    spec = example_nested_spec(dtype)
    sample = array_spec.sample_spec_nest(spec, self.rng)

    bounded = array_spec.BoundedArraySpec.from_spec(spec["array_spec_1"])
    self.assertTrue(np.all(sample["array_spec_1"] >= bounded.minimum))
    self.assertTrue(np.all(sample["array_spec_1"] <= bounded.maximum))

    self.assertTrue(np.all(sample["bounded_spec_1"] >= -10))
    self.assertTrue(np.all(sample["bounded_spec_1"] <= 10))

    self.assertIn("array_spec_2", sample["dict_spec"])
    self.assertIn("bounded_spec_2", sample["dict_spec"])

    self.assertIn("tuple_spec", sample)

    self.assertIn("list_spec", sample)
    self.assertTrue(np.all(sample["list_spec"][1][1] >= -10))
    self.assertTrue(np.all(sample["list_spec"][1][1] <= 10))

  def testNestSampleOuterDims(self, dtype):
    spec = example_nested_spec(dtype)
    outer_dims = [2, 3]
    sample = array_spec.sample_spec_nest(
        spec, self.rng, outer_dims=outer_dims)

    bounded = array_spec.BoundedArraySpec.from_spec(spec["array_spec_1"])
    self.assertTrue(np.all(sample["array_spec_1"] >= bounded.minimum))
    self.assertTrue(np.all(sample["array_spec_1"] <= bounded.maximum))

    self.assertTrue(np.all(sample["bounded_spec_1"] >= -10))
    self.assertTrue(np.all(sample["bounded_spec_1"] <= 10))

    self.assertIn("array_spec_2", sample["dict_spec"])
    self.assertIn("bounded_spec_2", sample["dict_spec"])

    self.assertIn("tuple_spec", sample)

    self.assertIn("list_spec", sample)
    self.assertTrue(np.all(sample["list_spec"][1][1] >= -10))
    self.assertTrue(np.all(sample["list_spec"][1][1] <= 10))

    def _test_batched_shape(sample_, spec_):
      self.assertSequenceEqual(sample_.shape, outer_dims + list(spec_.shape))

    tf.nest.map_structure(_test_batched_shape, sample, spec)


class CheckArraysNestTest(parameterized.TestCase):

  @parameterized.named_parameters(*TYPE_PARAMETERS)
  def testMatch(self, dtype):
    spec = example_nested_spec(dtype)
    sample = array_spec.sample_spec_nest(spec, np.random.RandomState())
    self.assertTrue(array_spec.check_arrays_nest(sample, spec))

  @parameterized.named_parameters(
      ("different keys", {"foo": np.array([1])}, {"bar": example_basic_spec()}),
      ("different types 1", {"foo": np.array([1])}, [example_basic_spec()]),
      ("different types 2", [np.array([1])], {"foo": example_basic_spec()}),
      ("different lengths", [np.array([1])], [example_basic_spec(),
                                              example_basic_spec()]),
      ("array mismatch 1",
       {"foo": np.array([1, 2])}, {"foo": example_basic_spec()}),
      ("array mismatch 2", [np.array([1, 2])], [example_basic_spec()]),
      ("not an array", "a string", example_basic_spec()),
      ("not a spec", np.array([1]), "a string"),
  )
  def testNoMatch(self, arrays, spec):
    self.assertFalse(array_spec.check_arrays_nest(arrays, spec))


class ArraySpecTest(parameterized.TestCase):

  def testShapeTypeError(self):
    with self.assertRaises(TypeError):
      array_spec.ArraySpec(32, np.int32)

  def testDtypeTypeError(self):
    with self.assertRaises(TypeError):
      array_spec.ArraySpec((1, 2, 3), "32")

  def testStringDtype(self):
    array_spec.ArraySpec((1, 2, 3), "int32")

  def testNumpyDtype(self):
    array_spec.ArraySpec((1, 2, 3), np.int32)

  def testDtype(self):
    spec = array_spec.ArraySpec((1, 2, 3), np.int32)
    self.assertEqual(np.int32, spec.dtype)

  def testShape(self):
    spec = array_spec.ArraySpec([1, 2, 3], np.int32)
    self.assertEqual((1, 2, 3), spec.shape)

  def testEqual(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int32)
    spec_2 = array_spec.ArraySpec((1, 2, 3), np.int32)
    self.assertEqual(spec_1, spec_2)

  def testNotEqualDifferentShape(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int32)
    spec_2 = array_spec.ArraySpec((1, 3, 3), np.int32)
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualDifferentDtype(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int64)
    spec_2 = array_spec.ArraySpec((1, 2, 3), np.int32)
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualOtherClass(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int32)
    spec_2 = None
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = ()
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

  def testFromArray(self):
    spec = array_spec.ArraySpec.from_array(np.array([1, 2]), "test")
    self.assertEqual(spec.shape, (2,))
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.name, "test")

  def testFromArrayWithScalar(self):
    spec = array_spec.ArraySpec.from_array(5, "test")
    self.assertEqual(spec.shape, tuple())
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.name, "test")

  def testFromArrayWithNonNumeric(self):
    self.assertRaises(ValueError, array_spec.ArraySpec.from_array, "a string")

  @parameterized.named_parameters(*TYPE_PARAMETERS)
  def testCheckArrayMatch(self, dtype):
    spec = array_spec.ArraySpec((2,), dtype)
    self.assertTrue(spec.check_array(np.array([1, 2], dtype)))

  def testCheckArrayMatchWithScalar(self):
    spec = array_spec.ArraySpec(tuple(), np.double)
    self.assertTrue(spec.check_array(5.0))

  @parameterized.named_parameters(
      ("wrong shape", np.array([1])),
      ("wrong dtype", np.array([1, 2], dtype=np.double)),
      ("not an array", "a string"))
  def testCheckArrayNoMatch(self, array):
    spec = array_spec.ArraySpec((2,), np.int64)
    self.assertFalse(spec.check_array(array))

  @parameterized.named_parameters(*TYPE_PARAMETERS)
  def testReplaceDtype(self, dtype):
    spec = array_spec.ArraySpec(tuple(), np.double).replace(dtype=dtype)
    self.assertEqual(spec.dtype, dtype)

  def testReplace(self):
    spec = array_spec.ArraySpec(tuple(), np.double)
    new_spec = spec.replace(shape=(2,))
    self.assertEqual(new_spec.shape, (2,))
    new_spec = new_spec.replace(dtype=np.int8)
    self.assertEqual(new_spec.dtype, np.int8)
    new_spec = new_spec.replace(name="name")
    self.assertEqual(new_spec.name, "name")
    exp_spec = array_spec.ArraySpec((2,), np.int8, name="name")
    self.assertEqual(exp_spec, new_spec)


class BoundedArraySpecTest(parameterized.TestCase):

  def testInvalidMinimum(self):
    with self.assertRaisesRegexp(ValueError, "not compatible"):
      array_spec.BoundedArraySpec((3, 5), np.uint8, (0, 0, 0), (1, 1))

  def testInvalidMaximum(self):
    with self.assertRaisesRegexp(ValueError, "not compatible"):
      array_spec.BoundedArraySpec((3, 5), np.uint8, 0, (1, 1, 1))

  def testMinLargerThanMax(self):
    with self.assertRaisesRegexp(ValueError, "min has values greater than max"):
      array_spec.BoundedArraySpec((3,), np.uint8, (1, 2, 3), (3, 2, 1))

  def testHandleInfLimits(self):
    spec = array_spec.BoundedArraySpec(
        (1, 2, 3),
        np.float32,
        (-np.inf, 5, -np.inf),
        (np.inf, 5, np.inf),
    )
    self.assertNotIn(np.inf, spec.minimum)
    self.assertNotIn(-np.inf, spec.minimum)

    self.assertNotIn(np.inf, spec.maximum)
    self.assertNotIn(-np.inf, spec.maximum)

    self.assertEqual(5, spec.minimum[1])
    self.assertEqual(5, spec.maximum[1])

  def testMinMaxAttributes(self):
    spec = array_spec.BoundedArraySpec((1, 2, 3), np.float32, 0, (5, 5, 5))
    self.assertEqual(type(spec.minimum), np.ndarray)
    self.assertEqual(type(spec.maximum), np.ndarray)

  def testNotWriteable(self):
    spec = array_spec.BoundedArraySpec((1, 2, 3), np.float32, 0, (5, 5, 5))
    with self.assertRaisesRegexp(ValueError, "read-only"):
      spec.minimum[0] = -1
    with self.assertRaisesRegexp(ValueError, "read-only"):
      spec.maximum[0] = 100

  def testEqualBroadcastingBounds(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=0.0, maximum=1.0)
    spec_2 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertEqual(spec_1, spec_2)

  def testNotEqualDifferentMinimum(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, -1.6], maximum=[1.0, 1.0])
    spec_2 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertNotEqual(spec_1, spec_2)

  def testReuseSpec(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=0.0, maximum=1.0)
    spec_2 = array_spec.BoundedArraySpec(spec_1.shape, spec_1.dtype,
                                         spec_1.minimum, spec_1.maximum)
    self.assertEqual(spec_1, spec_2)

  def testNotEqualOtherClass(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, -0.6], maximum=[1.0, 1.0])
    spec_2 = array_spec.ArraySpec((1, 2), np.int32)
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = None
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = ()
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

  def testNotEqualDifferentMaximum(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=0.0, maximum=2.0)
    spec_2 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertNotEqual(spec_1, spec_2)

  def testRepr(self):
    as_string = repr(
        array_spec.BoundedArraySpec(
            (1, 2), np.int32, minimum=73.0, maximum=101.0))
    self.assertIn("101", as_string)
    self.assertIn("73", as_string)

  def testFromArraySpec(self):
    spec = array_spec.ArraySpec((2, 3), np.int32)
    bounded_spec = array_spec.BoundedArraySpec.from_spec(spec)
    self.assertEqual(np.int32, bounded_spec.dtype)

    i64_info = np.iinfo(np.int32)

    self.assertEqual(i64_info.min, bounded_spec.minimum)
    self.assertEqual(i64_info.max, bounded_spec.maximum)

  def testFromBoundedArraySpec(self):
    bounded_spec = array_spec.BoundedArraySpec(
        (2, 3), np.int32, minimum=5, maximum=15, name="test_spec")
    new_spec = array_spec.BoundedArraySpec.from_spec(bounded_spec)

    self.assertEqual(bounded_spec.minimum, new_spec.minimum)
    self.assertEqual(bounded_spec.maximum, new_spec.maximum)
    self.assertEqual(bounded_spec.dtype, new_spec.dtype)
    self.assertEqual(bounded_spec.shape, new_spec.shape)
    self.assertEqual(bounded_spec.name, new_spec.name)

  def testFromArraySpecRename(self):
    bounded_spec = array_spec.BoundedArraySpec(
        (2, 3), np.int32, minimum=5, maximum=15, name="test_spec")
    new_spec = array_spec.BoundedArraySpec.from_spec(
        bounded_spec, name="rename")

    self.assertEqual(bounded_spec.minimum, new_spec.minimum)
    self.assertEqual(bounded_spec.maximum, new_spec.maximum)
    self.assertEqual(bounded_spec.dtype, new_spec.dtype)
    self.assertEqual(bounded_spec.shape, new_spec.shape)
    self.assertEqual("rename", new_spec.name)

  @parameterized.named_parameters(*TYPE_PARAMETERS)
  def testCheckArrayMatch(self, dtype):
    spec = array_spec.BoundedArraySpec((2,), dtype, minimum=5, maximum=15)
    self.assertTrue(spec.check_array(np.array([6, 7], dtype)))
    # Bounds should be inclusive.
    self.assertTrue(spec.check_array(np.array([5, 15], dtype)))

  @parameterized.named_parameters(
      ("wrong shape", np.array([1])),
      ("wrong dtype", np.array([1, 2], dtype=np.double)),
      ("not an array", "a string"),
      ("out of bounds 1", np.array([1, 10])),
      ("out of bounds 2", np.array([5, 20])))
  def testCheckArrayNoMatch(self, array):
    spec = array_spec.BoundedArraySpec((2,), np.int64, minimum=5, maximum=15)
    self.assertFalse(spec.check_array(array))

  # Tests that random sample of a complete uint8 range contains all values.
  def testSampleUint8(self):
    self.skipTest("TODO(oars): Fix this test.")
    rng = np.random.RandomState()
    spec = array_spec.BoundedArraySpec(
        (100, 10, 10), np.uint8, minimum=0, maximum=255)
    sample = array_spec.sample_bounded_spec(spec, rng)
    self.assertTupleEqual((100, 10, 10), sample.shape)
    hist, _ = np.histogram(sample, bins=256, range=(0, 255))
    self.assertTrue(np.all(hist > 0))

  # Tests that random sample of a complete int8 range contains all values. The
  # caveat is that difference of max - min is not int8.
  # TODO(oars): Fix these tests: perhaps by chance not every bin is filled?
  # Need a lot more samples (e.g. shape (100, 100, 100) to ensure they are?
  def testSampleInt8(self):
    self.skipTest("TODO(oars): Fix this test.")
    rng = np.random.RandomState()
    spec = array_spec.BoundedArraySpec(
        (100, 10, 10), np.int8, minimum=-128, maximum=127)
    sample = array_spec.sample_bounded_spec(spec, rng)
    self.assertTupleEqual((100, 10, 10), sample.shape)
    hist, _ = np.histogram(sample, bins=256, range=(-128, 127))
    self.assertTrue(np.all(hist > 0))

  # Tests that random sample from uint64 does have all values requested.
  def testSampleUint64SmallRange(self):
    self.skipTest("TODO(oars): Fix this test.")
    rng = np.random.RandomState()
    spec = array_spec.BoundedArraySpec(
        (100, 10, 10), np.uint64, minimum=0, maximum=100)
    sample = array_spec.sample_bounded_spec(spec, rng)
    self.assertTupleEqual((100, 10, 10), sample.shape)
    hist, _ = np.histogram(sample, bins=100, range=(0, 100))
    self.assertTrue(np.all(hist > 0))

  # Tests that random sample from full int64 works well. The caveat is that the
  # full range min-max cannot be represented as an int64.
  def testSampleInt64FullRange(self):
    rng = np.random.RandomState()
    spec = array_spec.BoundedArraySpec(
        (100, 10, 10),
        np.int64,
        minimum=np.iinfo(np.int64).min,
        maximum=np.iinfo(np.int64).max)
    sample = array_spec.sample_bounded_spec(spec, rng)
    self.assertTupleEqual((100, 10, 10), sample.shape)
    hist, _ = np.histogram(sample, bins=100, range=(np.iinfo(np.int64).min / 2,
                                                    np.iinfo(np.int64).max / 2))
    self.assertTrue(np.all(hist > 0))

  # Tests that random sample from full float64 does have no infs.
  def testSampleFloat64FullRange(self):
    rng = np.random.RandomState()
    spec = array_spec.BoundedArraySpec(
        (100, 10, 10), np.float64, minimum=0, maximum=100)
    sample = array_spec.sample_bounded_spec(spec, rng)
    self.assertTupleEqual((100, 10, 10), sample.shape)
    self.assertFalse(np.any(np.isinf(sample)))
    hist, _ = np.histogram(sample, bins=100, range=(0, 100))
    self.assertTrue(np.all(hist > 0))

  def testReplace(self):
    spec = array_spec.BoundedArraySpec(tuple(), np.int8, minimum=0, maximum=1)
    new_spec = spec.replace(shape=(2,))
    self.assertEqual(new_spec.shape, (2,))
    new_spec = new_spec.replace(dtype=np.int32)
    self.assertEqual(new_spec.dtype, np.int32)
    new_spec = new_spec.replace(name="name")
    self.assertEqual(new_spec.name, "name")
    new_spec = new_spec.replace(minimum=-1)
    self.assertEqual(new_spec.minimum, -1)
    new_spec = new_spec.replace(maximum=0)
    self.assertEqual(new_spec.maximum, 0)
    exp_spec = array_spec.BoundedArraySpec((2,), np.int32,
                                           minimum=-1, maximum=0, name="name")
    self.assertEqual(exp_spec, new_spec)

  @parameterized.named_parameters(*TYPE_PARAMETERS)
  def testNumValues(self, dtype):
    spec = array_spec.BoundedArraySpec(tuple(), dtype, minimum=0, maximum=9)
    num_values = spec.num_values
    if array_spec.is_discrete(spec):
      self.assertEqual(10, num_values)
    else:
      self.assertEqual(None, num_values)

  def testNumValuesVector(self):
    spec = array_spec.BoundedArraySpec((2,), np.int32, [0, 0], [1, 1])
    self.assertTrue(np.all([2, 2] == spec.num_values))
    spec = spec.replace(minimum=1)
    self.assertTrue(np.all([1, 1] == spec.num_values))
    spec = spec.replace(maximum=2)
    self.assertTrue(np.all([2, 2] == spec.num_values))


@parameterized.named_parameters(*TYPE_PARAMETERS)
class ArraySpecTypeTest(parameterized.TestCase):

  def testIsDiscrete(self, dtype):
    spec = array_spec.ArraySpec((2, 3), dtype=dtype)
    self.assertIs(tensor_spec.is_discrete(spec),
                  issubclass(np.dtype(dtype).type, np.integer))

  def testIsContinuous(self, dtype):
    spec = array_spec.ArraySpec((2, 3), dtype=dtype)
    self.assertIs(tensor_spec.is_continuous(spec),
                  issubclass(np.dtype(dtype).type, np.floating))

  def testExclusive(self, dtype):
    spec = array_spec.ArraySpec((2, 3), dtype=dtype)
    self.assertIs(
        tensor_spec.is_discrete(spec) ^ tensor_spec.is_continuous(spec), True)


if __name__ == "__main__":
  tf.test.main()
