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

"""Common utility functions for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import numpy as np
import tensorflow as tf


FLAGS = flags.FLAGS


def contains(list1, list2):
  """Check if all items in list2 are in list1.

  This function handles the case when the parameters are lists of np.arrays
  (which wouldn't be handled by something like .issubset(...)

  Args:
    list1: List which may or may not contain list2.
    list2: List to check if included in list 1.
  Returns:
    A boolean indicating whether list2 is contained in list1.
  """
  contains_result = True
  for item2 in list2:
    contains_result = contains_result and np.any(
        [np.all(item2 == item1) for item1 in list1])
    if not contains_result:
      break
  return contains_result


def test_src_dir_path(relative_path):
  """Returns an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tf_agents root.
      e.g. "environments/config".

  Returns:
    An absolute path to the linked in runfiles.
  """
  return os.path.join(FLAGS.test_srcdir,
                      'tf_agents',
                      relative_path)


class TestCase(tf.test.TestCase):

  def setUp(self):
    super(TestCase, self).setUp()
    tf.compat.v1.enable_resource_variables()
