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

"""Tests reinforcement_learning.environments.specs without tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tf_agents import specs
from tf_agents.utils import test_utils


class SpecWithTensorflowTest(test_utils.TestCase):

  def testLoad(self):
    specs.ArraySpec([1, 2, 3], np.int32)
    specs.BoundedArraySpec([1, 2, 3], np.int32, 0, 1)
    specs.TensorSpec([1, 2, 3], np.int32)
    specs.BoundedTensorSpec([1, 2, 3], np.int32, 0, 1)

  def testFromArraySpecToTensorSpec(self):
    array_spec = specs.ArraySpec([1, 2, 3], np.int32)
    tensor_spec = specs.TensorSpec.from_spec(array_spec)
    self.assertEqual(array_spec.shape, tensor_spec.shape)
    self.assertEqual(array_spec.dtype, tensor_spec.dtype.as_numpy_dtype(0))
    self.assertEqual(array_spec.name, tensor_spec.name)
    self.assertEqual(type(tensor_spec), specs.tensor_spec.TensorSpec)

  def testFromArraySpecToBoundedTensorSpec(self):
    array_spec = specs.ArraySpec([1, 2, 3], np.int32)
    tensor_spec = specs.BoundedTensorSpec.from_spec(array_spec)
    self.assertEqual(array_spec.shape, tensor_spec.shape)
    self.assertEqual(array_spec.dtype, tensor_spec.dtype.as_numpy_dtype(0))
    self.assertEqual(array_spec.name, tensor_spec.name)
    self.assertEqual(tensor_spec.dtype.min, tensor_spec.minimum)
    self.assertEqual(tensor_spec.dtype.max, tensor_spec.maximum)
    self.assertEqual(type(tensor_spec), specs.tensor_spec.BoundedTensorSpec)

  def testFromBoundedArraySpecToBoundedTensorSpec(self):
    array_spec = specs.BoundedArraySpec([1, 2, 3], np.int32, 0, 1)
    tensor_spec = specs.BoundedTensorSpec.from_spec(array_spec)
    self.assertEqual(array_spec.shape, tensor_spec.shape)
    self.assertEqual(array_spec.dtype, tensor_spec.dtype.as_numpy_dtype(0))
    self.assertEqual(array_spec.minimum, tensor_spec.minimum)
    self.assertEqual(array_spec.maximum, tensor_spec.maximum)
    self.assertEqual(array_spec.name, tensor_spec.name)
    self.assertEqual(type(tensor_spec), specs.tensor_spec.BoundedTensorSpec)

if __name__ == '__main__':
  test_utils.main()
