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

r"""Creates graphs and summary information from event logs.

This was created to build the graphs and supporting data for the README pages
for each agent. There are a number of FLAGS but the script is not designed to be
fully configurable. Making changes directly in the script is expected for one
off needs.

Usage examples:
  python3 graph_builder.py --eventlog=<path to event log 1> \
                           --eventlog=<path to event log 2>
"""
import csv
import enum
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tf_agents.benchmark import utils

FLAGS = flags.FLAGS

flags.DEFINE_multi_string('eventlog', None,
                          'Diretory where eventlog is stored.')
flags.DEFINE_string('output_path', '.',
                    'Path to store the graph and any other associated data.')
flags.DEFINE_string('output_prefix', 'results', 'Prefix used for artifacts')
flags.DEFINE_string('graph_title', '', 'Title for the graph.')
flags.DEFINE_string('graph_xaxis_title', 'steps', 'Title for the x-axis.')
flags.DEFINE_string('graph_yaxis_title', 'AverageReturn',
                    'Title for the y-axis or event_name is used.')

flags.DEFINE_string('event_name', 'AverageReturn', 'Name of event to track.')
flags.DEFINE_integer('end_step', None,
                     'If set, processing of the event log ends on this step.')
flags.DEFINE_boolean('show_graph', False, 'If true, show graph in a window.')


class GraphAggTypes(enum.Enum):
  """Enum of options to aggregate data when generating a graph."""
  MEAN = 'mean'
  MEDIAN = 'median'


flags.DEFINE_enum_class('graph_agg', GraphAggTypes.MEAN, GraphAggTypes,
                        'Method to aggregate data for the graph.')
Number = Union[int, float]


class StatsBuilder(object):
  """Builds graphs and other summary information from eventlogs."""

  def __init__(self,
               eventlog_dirs: List[str],
               event_tag: str,
               output_path: str = '.',
               title: str = '',
               xaxis_title: str = 'steps',
               yaxis_title: Optional[str] = None,
               graph_agg: GraphAggTypes = GraphAggTypes.MEAN,
               output_prefix: str = 'results',
               end_step: Optional[int] = None,
               show_graph: bool = False):
    """Initializes StatsBuilder class.

    Args:
      eventlog_dirs: List of paths to event log directories to process.
      event_tag: Event to extract from the logs.
      output_path: Output path for artifacts, e.g. graphs and cvs files.
      title: Title of the graph.
      xaxis_title: Title for x-axis of the graph. Defaults to "steps".
      yaxis_title: Title for the y-axis. Defaults to the `event_tag`.
      graph_agg: Aggregation for the graph.
      output_prefix: Prefix for the artifact files. Defaults to "results".
      end_step: If set, processing of the event log ends on this step.
      show_graph: If true, blocks and shows graph. Only tests in linux.

    Raises:
      ValueError: Raised if the graph_agg passed is not understood.
    """
    self.eventlog_dirs = eventlog_dirs
    self.event_tag = event_tag
    self.output_path = output_path
    self.title = title
    self.xaxis_title = xaxis_title
    self.show_graph = show_graph
    self.end_step = end_step
    if graph_agg == GraphAggTypes.MEAN:
      self.graph_agg = np.mean
    elif graph_agg == GraphAggTypes.MEDIAN:
      self.graph_agg = np.median
    else:
      raise ValueError('Unknown graph_agg:{}'.format(graph_agg))

    # Makes the output path absolute for clarity.
    self.output_dir = os.path.abspath(output_path)
    os.makedirs(self.output_dir, exist_ok=True)
    self.output_prefix = output_prefix

    if yaxis_title:
      self.yaxis_title = yaxis_title
    else:
      self.yaxis_title = event_tag

  def _gather_data(self) -> Tuple[List[Dict[int, np.generic]], List[float]]:
    """Gather data from all of the logs and add to the data_collector list.

    Returns:
      Tuple of arrays indexed by log file, e.g. data_collector[0] is all of the
      values found in the event log for the given event and walltimes[0] is the
      total time in minutes it took to get to the end_step in that event log.
    """
    data_collector, walltimes = [], []
    for eventlog_dir in self.eventlog_dirs:
      event_file = utils.find_event_log(eventlog_dir)
      logging.info('Processing event file: %s', event_file)
      data, total_time = utils.extract_event_log_values(event_file,
                                                        self.event_tag,
                                                        self.end_step)
      walltimes.append(total_time)
      data_collector.append(data)
    return data_collector, walltimes

  def _align_and_aggregate(
      self, data_collector: List[Dict[int,
                                      np.generic]]) -> List[Sequence[Number]]:
    """Combines data from multipole runs into a pivot table like structure.

    Uses the first run as the base and aligns the data for each run by rows
    with each row representing a step. If a step is not found in a run,
    the value -1 is used. No error or warning is thrown or logged.

    Args:
      data_collector: list of dicts with each dict representing a run most
        likely extracted from an event log.

    Returns:
      2d array with each row representing a step and each run represented as
      a column, e.g. step, run 1, run 2, median, and mean.

    """
    # Use the first event log's steps as the base and create aggregated data
    # at the step internals of the first event log.
    base_data = data_collector[0]
    agg_data = []
    for key, value in sorted(base_data.items()):
      entry = [key]
      values = [value]
      for data in data_collector[1:]:
        values.append(data.get(key, -1))
      mean_val = np.mean(values)
      median_val = np.median(values)

      # Combines into step, values 1..n, median, and mean.
      values.append(median_val)
      values.append(mean_val)
      entry += values
      agg_data.append(entry)
    return agg_data

  def _output_csv(self, agg_data: List[Sequence[Number]]):
    """Exports the `agg_data` as a csv.

    Args:
      agg_data: 2d array of data to export to csv.
    """
    # Outputs csv with aggregated data for each step.
    csv_path = os.path.join(self.output_path,
                            self.output_prefix + '_summary.csv')
    with open(csv_path, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(agg_data)

  def _output_graph(self, agg_data: List[Sequence[Number]], num_runs: int):
    """Builds a graph of the results and outputs to a .png.

    Args:
      agg_data: 2d array of data to be graphed.
      num_runs: Number of columns of runs in the data.
    """
    # Build data frames
    columns = ['step']
    columns.extend([str(i) for i in range(num_runs)])
    # csv contains aggregate info that will get excluded in the pd.melt.
    columns.extend(['median', 'mean'])
    print(columns)
    df = pd.DataFrame(agg_data, columns=columns)
    logging.debug('Dataframes datatypes: %s', df.dtypes)
    new_pd = pd.melt(
        df,
        id_vars='step',
        value_vars=list(df.columns[1:num_runs + 1]),
        var_name='run',
        value_name=self.yaxis_title)
    logging.info('DataFrame to graph:\n%s', new_pd)
    # Build graph
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(
        data=new_pd, x='step', y=self.yaxis_title, estimator=self.graph_agg)
    ax.set_title(self.title)
    ax.set(xlabel=self.xaxis_title)
    plt.ticklabel_format(style='plain', axis='x')
    graph_path = os.path.join(self.output_path,
                              self.output_prefix + '_graph.png')
    plt.savefig(graph_path)

  def build_artifacts(self):
    """Processes the event logs and coordinates creating the artifacts."""
    data_collector, _ = self._gather_data()
    agg_data = self._align_and_aggregate(data_collector)
    self._output_csv(agg_data)
    self._output_graph(agg_data, len(data_collector))

    if self.show_graph:
      plt.show()


def main(_):
  logging.set_verbosity(logging.INFO)

  stat_builder = StatsBuilder(
      FLAGS.eventlog,
      FLAGS.event_name,
      output_path=FLAGS.output_path,
      output_prefix=FLAGS.output_prefix,
      title=FLAGS.graph_title,
      xaxis_title=FLAGS.graph_xaxis_title,
      yaxis_title=FLAGS.graph_yaxis_title,
      graph_agg=FLAGS.graph_agg,
      end_step=FLAGS.end_step,
      show_graph=FLAGS.show_graph)

  stat_builder.build_artifacts()


if __name__ == '__main__':
  flags.mark_flag_as_required('eventlog')
  app.run(main)
