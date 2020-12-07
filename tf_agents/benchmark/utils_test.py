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

"""Tests for benchmark.utils."""
import os

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.benchmark import utils
from tf_agents.utils import test_utils

TEST_DATA = 'tf_agents/benchmark/test_data'


class UtilsTest(test_utils.TestCase):

  def test_extract_value(self):
    """Tests extracting data from all steps in the event log."""
    values, walltime = utils.extract_event_log_values(
        os.path.join(TEST_DATA, 'event_log_3m/events.out.tfevents.1599310762'),
        'AverageReturn')
    # Verifies all (3M) records were examined 0-3M = 301.
    self.assertLen(values, 301)
    self.assertAlmostEqual(walltime, 1152.09573, places=4)
    # Verifies event value at 3M
    self.assertAlmostEqual(values[3000000], 5950.31835, places=4)

  def test_extract_value_1m_only(self):
    """Tests extracting data from the first 1M steps in the event log."""
    values, walltime = utils.extract_event_log_values(
        os.path.join(TEST_DATA, 'event_log_3m/events.out.tfevents.1599310762'),
        'AverageReturn', 1000000)
    # Verifies only 1M records were examined 0-1M = 101.
    self.assertLen(values, 101)
    self.assertAlmostEqual(walltime, 370.61673, places=4)
    # Verifies event value at 1M.
    self.assertAlmostEqual(values[1000000], 3791.88696, places=4)

  def test_more_than_one_eventlog_per_dir(self):
    """Tests that an exception is thrown if more than one log file is found."""
    with self.assertRaises(AssertionError):
      utils.find_event_log(os.path.join(TEST_DATA, 'event_log_too_many'))

  def test_no_eventlogs_found(self):
    """Tests that an exception is thrown if no log files are found."""
    with self.assertRaises(FileNotFoundError):
      utils.find_event_log(os.path.join(TEST_DATA, 'fake_path'))


if __name__ == '__main__':
  tf.test.main()
