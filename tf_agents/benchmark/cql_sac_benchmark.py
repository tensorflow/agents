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

"""Executes CQL-SAC benchmarks.

Benchmarks for CQL Kumar20 are based on https://arxiv.org/abs/2006.04779.
"""
import os
import time

from absl import logging
import gin
import tensorflow as tf

from tf_agents.benchmark import utils
from tf_agents.benchmark.perfzero_benchmark import PerfZeroBenchmark
from tf_agents.examples.cql_sac.kumar20 import cql_sac_train_eval


class CqlSacKumar20Return(PerfZeroBenchmark):
  """Benchmark return tests for CQL-SAC Kumar20."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmarks for CQL-SAC Kumar20.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
        constructor forward compatible in case PerfZero provides more named
        arguments before updating the constructor.
    """
    super(CqlSacKumar20Return, self).__init__(output_dir=output_dir)

  def benchmark_halfcheetah_medium_v0(self):
    """Benchmarks MuJoCo HalfCheetah to 1M steps."""
    self.setUp()
    output_dir = self._get_test_output_dir('halfcheetah_medium_v0_02_eval')
    start_time_sec = time.time()
    gin.parse_config_file(
        'tf_agents/examples/cql_sac/kumar20/configs/mujoco_medium.gin'
    )
    cql_sac_train_eval.train_eval(
        root_dir=output_dir,
        env_name='halfcheetah-medium-v0',
        dataset_name='d4rl_mujoco_halfcheetah/v0-medium',
        num_gradient_updates=500000,  # Number of iterations.
        learner_iterations_per_call=500,
        data_shuffle_buffer_size=10000,
        data_num_shards=50,
        data_parallel_reads=500,
        data_prefetch=1000000,
        eval_interval=10000)
    wall_time_sec = time.time() - start_time_sec
    event_file = utils.find_event_log(os.path.join(output_dir, 'eval'))
    values, _ = utils.extract_event_log_values(
        event_file, 'Metrics/AverageReturn', start_step=10000)

    # Min/Max ranges are very large to only hard fail if very broken. The system
    # monitoring the results owns looking for anomalies. These numbers are based
    # on the results that we were getting in MLCompass as of 04-NOV-2021.
    # Results at 500k steps and 1M steps are similar enough to not make it worth
    # running 1M.
    metric_500k = self.build_metric(
        'average_return_at_env_step500000',
        values[500000],
        min_value=4400,
        max_value=5400)

    self.report_benchmark(
        wall_time=wall_time_sec, metrics=[metric_500k], extras={})


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
