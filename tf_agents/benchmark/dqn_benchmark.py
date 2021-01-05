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

"""Executes DQN Mnih15 benchmarks."""
import os
import time

import tensorflow as tf
from tf_agents.benchmark import utils
from tf_agents.benchmark.perfzero_benchmark import PerfZeroBenchmark
from tf_agents.experimental.examples.dqn.mnih15 import dqn_train_eval_atari


class DqnMnih15Return(PerfZeroBenchmark):
  """Benchmark return tests for DQN Mnih15."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Benchmarks for DQN Mnih15.

    Args:
      output_dir: directory where to output e.g. log files.
      root_data_dir: directory under which to look for data.
      **kwargs: arbitrary named arguments. This is needed to make the
        constructor forward compatible in case PerfZero provides more named
        arguments before updating the constructor.
    """
    super(DqnMnih15Return, self).__init__(output_dir=output_dir)

  def benchmark_pong_v0_at_3M(self):
    """Benchmarks to 3M Env steps.

    This is below the 12.5M train steps (50M frames) run by the paper to
    converge. Running 12.5M at the current throughput would take more than a
    week. 1-2 days is the max duration for a remotely usable test. 3M only
    confirms we have not regressed at 3M and does not gurantee convergence to 21
    at 12.5M.
    """
    self._setup()
    output_dir = self._get_test_output_dir('pongAt3M')
    start_time_sec = time.time()
    dqn_train_eval_atari.train_eval(
        output_dir, eval_interval=10000, num_iterations=750000)
    wall_time_sec = time.time() - start_time_sec
    event_file = utils.find_event_log(os.path.join(output_dir, 'eval'))
    values, _ = utils.extract_event_log_values(
        event_file, 'AverageReturn/EnvironmentSteps')
    print('Values:{}'.format(values))
    # Min/Max ranges are very large to only hard fail if very broken. The system
    # monitoring the results owns looking for anomalies.
    metric_3m = self.build_metric(
        'average_return_at_env_step3000000',
        values[3000000],
        min_value=-14,
        max_value=21)

    self.report_benchmark(
        wall_time=wall_time_sec, metrics=[metric_3m], extras={})


if __name__ == '__main__':
  tf.test.main()
