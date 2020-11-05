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

"""Utils for creating PerfZero benchmarks."""
import os

from absl import flags
from absl import logging
from absl.testing import flagsaver
import tensorflow as tf

FLAGS = flags.FLAGS


class PerfZeroBenchmark(tf.test.Benchmark):
  """Common methods used in PerfZero Benchmarks.

     Handles the resetting of flags between tests. PerfZero (OSS) runs each test
     in a separate process reducing some need to reset the flags.
  """
  local_flags = None

  def __init__(self, output_dir=None):
    """Initialize class.

    Args:
      output_dir: Base directory to store all output for the test.
    """
    # MLCompass sets this value, but PerfZero OSS passes it as an arg.
    if os.getenv('BENCHMARK_OUTPUT_DIR'):
      self.output_dir = os.getenv('BENCHMARK_OUTPUT_DIR')
    elif output_dir:
      self.output_dir = output_dir
    else:
      self.output_dir = '/tmp'

  def _get_test_output_dir(self, folder_name):
    """Returns directory to store info, e.g. saved model and event log."""
    return os.path.join(self.output_dir, folder_name)

  def _setup(self):
    """Sets up and resets flags before each test."""
    logging.set_verbosity(logging.INFO)
    if PerfZeroBenchmark.local_flags is None:
      # Loads flags to get defaults to then override. List cannot be empty.
      flags.FLAGS(['foo'])
      saved_flag_values = flagsaver.save_flag_values()
      PerfZeroBenchmark.local_flags = saved_flag_values
    else:
      flagsaver.restore_flag_values(PerfZeroBenchmark.local_flags)
