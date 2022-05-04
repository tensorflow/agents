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

"""Tests for graph_builder.py."""
import os

from absl.testing import absltest

import graph_builder

TEST_DATA = 'test_data'


class GraphBuilderTest(absltest.TestCase):

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
