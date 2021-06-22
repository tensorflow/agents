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

"""Executes Soft Actor Critic (SAC) benchmarks."""
import os
import time

import tensorflow as tf
from tf_agents.benchmark import utils
from tf_agents.benchmark.perfzero_benchmark import PerfZeroBenchmark
from tf_agents.experimental.examples.sac.haarnoja18 import sac_train_eval


class SacHaarnoja18Return(PerfZeroBenchmark):
  """Benchmark return tests for Soft Actor Critic (SAC) Haarnoja18."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmarks for SAC Haarnoja18.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
        constructor forward compatible in case PerfZero provides more named
        arguments before updating the constructor.
    """
    super(SacHaarnoja18Return, self).__init__(output_dir=output_dir)

  def benchmark_halfcheetah_v2(self):
    """Benchmarks MuJoCo HalfCheetah to 3M steps."""
    self.setUp()
    output_dir = self._get_test_output_dir('halfcheetah_v2')
    start_time_sec = time.time()
    # TODO(b/172017027): Use halfcheetah gin config.
    sac_train_eval.train_eval(
        output_dir,
        initial_collect_steps=10000,
        env_name='HalfCheetah-v2',
        eval_interval=50000,
        num_iterations=3000000)
    wall_time_sec = time.time() - start_time_sec
    event_file = utils.find_event_log(os.path.join(output_dir, 'eval'))
    values, _ = utils.extract_event_log_values(event_file,
                                               'Metrics/AverageReturn')

    # Min/Max ranges are very large to only hard fail if very broken. The system
    # monitoring the results owns looking for anomalies.
    metric_1m = self.build_metric(
        'average_return_at_env_step1000000',
        values[1000000],
        min_value=800,
        max_value=16000)

    metric_3m = self.build_metric(
        'average_return_at_env_step3000000',
        values[3000000],
        min_value=12000,
        max_value=16500)

    self.report_benchmark(
        wall_time=wall_time_sec, metrics=[metric_1m, metric_3m], extras={})


if __name__ == '__main__':
  tf.test.main()
