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

"""Tests for utils/test_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tf_agents.utils import test_utils


class TestUtilsTest(test_utils.TestCase):

  def testBatchContainsSample(self):
    batch = np.array([[1, 2], [3, 4]])
    sample = np.array([3, 4])
    self.assertTrue(test_utils.contains(batch, [sample]))

  def testBatchDoesNotContainSample(self):
    batch = np.array([[1, 2], [3, 4]])
    sample = np.array([2, 4])
    self.assertFalse(test_utils.contains(batch, [sample]))

  def testBatchContainsBatch(self):
    batch1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    batch2 = np.array([[3, 4], [3, 4], [1, 2]])
    self.assertTrue(test_utils.contains(batch1, batch2))

  def testBatchDoesNotContainBatch(self):
    batch1 = np.array([[1, 2], [3, 4]])
    batch2 = np.array([[1, 2], [5, 6]])
    self.assertFalse(test_utils.contains(batch1, batch2))


if __name__ == '__main__':
  test_utils.main()
