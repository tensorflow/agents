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

"""Tests for example_encoding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.utils import example_encoding


TYPE_PARAMETERS = (
    ("np.uint8", np.uint8),
    ("np.uint16", np.uint16),
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


def example_nested_spec(dtype):
  """Return an example nested array spec."""
  low = -10
  high = 10
  if dtype in (np.uint8, np.uint16):
    low += -low
  return {
      "array_spec_1":
          array_spec.ArraySpec((2, 3), dtype),
      "bounded_spec_1":
          array_spec.BoundedArraySpec((2, 3), dtype, low, high),
      "empty_shape":
          array_spec.BoundedArraySpec((), dtype, low, high),
      "dict_spec": {
          "array_spec_2":
              array_spec.ArraySpec((2, 3), dtype),
          "bounded_spec_2":
              array_spec.BoundedArraySpec((2, 3), dtype, low, high)
      },
      "tuple_spec": (
          array_spec.ArraySpec((2, 3), dtype),
          array_spec.BoundedArraySpec((2, 3), dtype, low, high),
      ),
      "list_spec": [
          array_spec.ArraySpec((2, 3), dtype),
          (array_spec.ArraySpec((2, 3), dtype),
           array_spec.BoundedArraySpec((2, 3), dtype, low, high)),
      ],
  }


class NestExampleEncodeUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*TYPE_PARAMETERS)
  def test_serialize_deserialize(self, dtype):
    spec = example_nested_spec(dtype)
    serializer = example_encoding.get_example_serializer(spec)
    decoder = example_encoding.get_example_decoder(spec)

    sample = array_spec.sample_spec_nest(spec, np.random.RandomState(0))
    example_proto = serializer(sample)

    recovered = self.evaluate(decoder(example_proto))
    tf.nest.map_structure(np.testing.assert_almost_equal, sample, recovered)

  def test_endian_encodings(self):
    spec = {
        "a": array_spec.ArraySpec((2,), np.int16),
        "b": array_spec.ArraySpec((2,), np.int32),
        "c": array_spec.ArraySpec((2,), np.float32),
    }

    serializer = example_encoding.get_example_serializer(spec)
    decoder = example_encoding.get_example_decoder(spec)

    # Little endian encoding.
    le_sample = {
        "a": np.array([100, 25000]).astype("<i2"),
        "b": np.array([-5, 80000000]).astype("<i4"),
        "c": np.array([12.5, np.pi]).astype("<f4")
    }

    example_proto = serializer(le_sample)
    recovered = self.evaluate(decoder(example_proto))
    tf.nest.map_structure(np.testing.assert_almost_equal, le_sample, recovered)

    # Big endian encoding.
    be_sample = {
        "a": np.array([100, 25000]).astype(">i2"),
        "b": np.array([-5, 80000000]).astype(">i4"),
        "c": np.array([12.5, np.pi]).astype(">f4")
    }

    example_proto = serializer(be_sample)
    recovered = self.evaluate(decoder(example_proto))
    tf.nest.map_structure(np.testing.assert_almost_equal, be_sample, recovered)

  def test_shape_validation(self):
    with self.assertRaisesRegexp(ValueError, "is invalid"):
      example_encoding._validate_shape([1, 2, 3, -1])

    with self.assertRaisesRegexp(ValueError, "is invalid"):
      example_encoding._validate_shape([1, None, 3])

    with self.assertRaisesRegexp(ValueError, "is invalid"):
      example_encoding._validate_shape([1, 2.3, 3])


if __name__ == "__main__":
  tf.test.main()
