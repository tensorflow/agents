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

"""Tests for tf_agents.utils.numpy_storage."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import numpy_storage

from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import  # TF internal


class NumpyStorageTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestore(self):
    arrays = numpy_storage.NumpyState()
    checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
    arrays.x = np.ones([3, 4])
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, 'ckpt')
    save_path = checkpoint.save(prefix)
    arrays.x[:] = 0.
    self.assertAllEqual(arrays.x, np.zeros([3, 4]))
    checkpoint.restore(save_path).assert_consumed()
    self.assertAllEqual(arrays.x, np.ones([3, 4]))

    second_checkpoint = tf.train.Checkpoint(
        numpy_arrays=numpy_storage.NumpyState())
    # Attributes of NumpyState objects are created automatically by restore()
    second_checkpoint.restore(save_path).assert_consumed()
    self.assertAllEqual(np.ones([3, 4]), second_checkpoint.numpy_arrays.x)


if __name__ == '__main__':
  tf.test.main()
