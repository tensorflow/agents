# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
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

# Lint as: python3
"""Tests for graph_builder.py."""
import os

from absl import flags
from absl.testing import absltest

import graph_builder

FLAGS = flags.FLAGS
TEST_DATA = 'test_data'


class GraphBuilderTest(absltest.TestCase):

  def test_extract_value(self):
    """Tests extracting data from all steps in the event log."""
    stat_builder = graph_builder.StatsBuilder([''], 'AverageReturn')
    values, walltime = stat_builder._extract_values(
        os.path.join(TEST_DATA, 'event_log_ant_eval00/'))
    # Verifies all (3M) records were examined 0-3M = 301.
    self.assertLen(values, 301)
    self.assertAlmostEqual(walltime, 1152.09573, places=4)
    # Verifies event value at 3M
    self.assertAlmostEqual(values[3000000], 5950.31835, places=4)

  def test_extract_value_1m_only(self):
    """Tests extracting data from the first 1M steps in the event log."""
    stat_builder = graph_builder.StatsBuilder([''],
                                              'AverageReturn',
                                              end_step=1000000)
    values, walltime = stat_builder._extract_values(
        os.path.join(TEST_DATA, 'event_log_ant_eval00/'))
    # Verifies only 1M records were examined 0-1M = 101.
    self.assertLen(values, 101)
    self.assertAlmostEqual(walltime, 370.61673, places=4)
    # Verifies event value at 1M.
    self.assertAlmostEqual(values[1000000], 3791.88696, places=4)

  def test_align_and_aggregate(self):
    """Tests combining data from 3 differnet logs into a single result."""
    event_log_dirs = [
        'event_log_ant_eval00', 'event_log_ant_eval01', 'event_log_ant_eval02'
    ]
    event_log_paths = [
        os.path.join(TEST_DATA, log_dir) for log_dir in event_log_dirs
    ]
    stat_builder = graph_builder.StatsBuilder(event_log_paths, 'AverageReturn')
    data_collector, _ = stat_builder._gather_data()
    agg_results = stat_builder._align_and_aggregate(data_collector)
    # Mean at step 3M.
    self.assertAlmostEqual(agg_results[-1][-1], 5674.96354, places=4)
    # Median at step 3M.
    self.assertAlmostEqual(agg_results[-1][-2], 5573.90380, places=4)

  def test_more_than_one_eventlog_per_dir(self):
    """Tests combining data from 3 differnet logs into a single result."""
    event_log_paths = [os.path.join(TEST_DATA, 'event_log_too_many')]
    stat_builder = graph_builder.StatsBuilder(event_log_paths, 'AverageReturn')
    with self.assertRaises(AssertionError):
      stat_builder._gather_data()

  def test_no_eventlogs_found(self):
    """Tests combining data from 3 differnet logs into a single result."""
    event_log_paths = [os.path.join(TEST_DATA, 'fake_path')]
    stat_builder = graph_builder.StatsBuilder(event_log_paths, 'AverageReturn')
    with self.assertRaises(FileNotFoundError):
      stat_builder._gather_data()

  def test_output_graph(self):
    """Tests outputing a graph to a file does not error out.

    There is no validation that the output graph is correct.
    """
    output_path = self.create_tempdir()
    event_log_dirs = [
        'event_log_ant_eval00', 'event_log_ant_eval01', 'event_log_ant_eval02'
    ]
    event_log_paths = [
        os.path.join(TEST_DATA, log_dir) for log_dir in event_log_dirs
    ]
    stat_builder = graph_builder.StatsBuilder(
        event_log_paths,
        'AverageReturn',
        graph_agg=graph_builder.GraphAggTypes.MEDIAN,
        output_path=output_path.full_path)
    data_collector, _ = stat_builder._gather_data()
    agg_results = stat_builder._align_and_aggregate(data_collector)
    stat_builder._output_graph(agg_results, len(event_log_dirs))


if __name__ == '__main__':
  absltest.main()
