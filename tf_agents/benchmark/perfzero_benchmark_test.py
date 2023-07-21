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

"""Tests for tf_agents.benchmark.perfzero_benchmark."""
from tf_agents.benchmark import perfzero_benchmark
from tf_agents.utils import test_utils


class PerfzeroBenchmarkTest(test_utils.TestCase):

  def test_build_metric(self):
    """Tests building metric with only required values."""
    bench = perfzero_benchmark.PerfZeroBenchmark()
    metric_name = 'metric_name'
    value = 25.93

    expected_metric = {'name': metric_name, 'value': value}
    metric = bench.build_metric(metric_name, value)
    self.assertEqual(metric, expected_metric)

  def test_build_metric_min_only(self):
    """Tests building metric with added min_value."""
    bench = perfzero_benchmark.PerfZeroBenchmark()
    metric_name = 'metric_name'
    value = 25.93
    min_value = 0.004

    expected_metric = {
        'name': metric_name,
        'value': value,
        'min_value': min_value
    }
    metric = bench.build_metric(metric_name, value, min_value=min_value)
    self.assertEqual(metric, expected_metric)

  def test_build_metric_max_only(self):
    """Tests building metric with added max_value."""
    bench = perfzero_benchmark.PerfZeroBenchmark()
    metric_name = 'metric_name'
    value = 25.93
    max_value = 89632.22

    expected_metric = {
        'name': metric_name,
        'value': value,
        'max_value': max_value
    }
    metric = bench.build_metric(metric_name, value, max_value=max_value)
    self.assertEqual(metric, expected_metric)

  def test_build_metric_min_max(self):
    """Tests building metric with min and max values."""
    bench = perfzero_benchmark.PerfZeroBenchmark()
    metric_name = 'metric_name'
    value = 25.93
    min_value = 0.004
    max_value = 59783

    expected_metric = {
        'name': metric_name,
        'value': value,
        'min_value': min_value,
        'max_value': max_value
    }
    metric = bench.build_metric(
        metric_name, value, min_value=min_value, max_value=max_value)
    self.assertEqual(metric, expected_metric)


if __name__ == '__main__':
  test_utils.main()
