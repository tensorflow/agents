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

"""Executes PPO Schulman17 benchmarks."""
import os
import time

import gin
import tensorflow as tf
from tf_agents.benchmark import utils
from tf_agents.benchmark.perfzero_benchmark import PerfZeroBenchmark
from tf_agents.experimental.examples.ppo.schulman17 import ppo_clip_train_eval


class PpoSchulman17Return(PerfZeroBenchmark):
  """Benchmark return tests for PPO Schulman17."""

  def _setup(self):
    """Setup the test by clearing gin config."""
    gin.clear_config()
    super(PpoSchulman17Return, self)._setup()

  def _tearDown(self):
    """Setup the test by making sure gin config is clear."""
    gin.clear_config()

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """Initializes class to run benchmarks for PPO Schulman17.

    Args:
      output_dir: directory where to output e.g. log files.
      root_data_dir: directory under which to look for data.
      **kwargs: arbitrary named arguments. This is needed to make the
        constructor forward compatible in case PerfZero provides more named
        arguments before updating the constructor.
    """
    super(PpoSchulman17Return, self).__init__(output_dir=output_dir)

  def benchmark_halfcheetah_v2(self):
    """Benchmarks MuJoCo HalfCheetah to 1M steps.

    HalfCheetah-v2 has a range of return values at 1M steps. The min and max
    expected values set for the test are limits for a hard failure. More
    nuanced checks are expected to be done by the regression analysis system.
    """
    self.run_benchmark('HalfCheetah-v2', 2400, 7000)

  def benchmark_invertedpendulum_v2(self):
    """Benchmarks MuJoCo InvertedPendulum to 1M steps.

    InvertedPendulum-v2 is expected to have a return of 1000 at 1M steps.
    """
    self.run_benchmark('InvertedPendulum-v2', 1000, 1000)

  def run_benchmark(self, training_env, expected_min, expected_max):
    """Run benchmark for a given environment.

    In order to execute ~1M environment steps to match the paper, we run 489
    iterations (num_iterations=489) which results in 1,001,472 environment
    steps. Each iteration results in 320 training steps and 2,048 environment
    steps. Thus 489 * 2,048 = 1,001,472 environment steps and
    489 * 320 = 156,480 training steps.

    Args:
      training_env: Name of environment to test.
      expected_min: The min expected return value.
      expected_max: The max expected return value.
    """
    self.setUp()
    output_dir = self._get_test_output_dir('training_env')
    start_time_sec = time.time()
    bindings = [
        'schulman17.train_eval_lib.train_eval.env_name= "{}"'.format(
            training_env),
        'schulman17.train_eval_lib.train_eval.eval_episodes = 100'
    ]
    gin.parse_config(bindings)
    ppo_clip_train_eval.ppo_clip_train_eval(
        output_dir, eval_interval=10000, num_iterations=489)
    wall_time_sec = time.time() - start_time_sec
    event_file = utils.find_event_log(os.path.join(output_dir, 'eval'))
    values, _ = utils.extract_event_log_values(
        event_file, 'Metrics/AverageReturn/EnvironmentSteps')

    metric_1m = self.build_metric(
        'average_return_at_env_step1000000',
        values[1001472],
        min_value=expected_min,
        max_value=expected_max)

    self.report_benchmark(
        wall_time=wall_time_sec, metrics=[metric_1m], extras={})
    self._tearDown()


if __name__ == '__main__':
  tf.test.main()
