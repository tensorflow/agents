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

"""Executes Soft Actor Critic (SAC) benchmarks."""
import time

import tensorflow as tf
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
    self._setup()
    start_time_sec = time.time()
    # TODO(b/172017027): Use halfcheetah gin config.
    sac_train_eval.train_eval(
        self._get_test_output_dir('halfcheetah_v2'),
        initial_collect_steps=10000,
        env_name='HalfCheetah-v2',
        num_iterations=3000000)
    wall_time_sec = time.time() - start_time_sec
    # TODO(b/172011457): Add train and eval final return, batch-size, and
    # batches/sec to perfzero metrics.
    self._report_benchmark(wall_time_sec)

  def _report_benchmark(self, wall_time_sec):
    """Reports benchmark results.

    Args:
      wall_time_sec: the during of the benchmark execution in seconds
    """
    metrics = []
    self.report_benchmark(wall_time=wall_time_sec, metrics=metrics, extras={})


if __name__ == '__main__':
  tf.test.main()
