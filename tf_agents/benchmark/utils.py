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

"""Utilities for running benchmarks."""
import datetime
import os
import time
from typing import Dict, Optional, Tuple

from absl import logging
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.util import event_pb2  # TF internal
from tensorflow.python.lib.io import tf_record  # TF internal
# pylint: enable=g-direct-tensorflow-import


def run_test(target_call,
             num_steps,
             strategy,
             batch_size=None,
             log_steps=100,
             num_steps_per_batch=1):
  """Run benchmark and return TimeHistory object with stats.

  Args:
    target_call: Call to execute for each step.
    num_steps: Number of steps to run.
    strategy: None or tf.distribute.DistibutionStrategy object.
    batch_size: Total batch size.
    log_steps: Interval of steps between logging of stats.
    num_steps_per_batch: Number of steps per batch. Used to account for total
      number of transitions or examples processed per iteration.

  Returns:
    TimeHistory object containing step performance stats.
  """
  history = TimeHistory(batch_size, log_steps, num_steps_per_batch)

  for _ in range(num_steps):
    history.on_batch_begin()
    if strategy:
      strategy.run(target_call)
    else:
      target_call()
    history.on_batch_end()

  return history


class BatchTimestamp(object):
  """A structure to store batch timestamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)


class TimeHistory(object):
  """Track step performance statistics."""

  def __init__(self, batch_size, log_steps, num_steps_per_batch=1):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of stats.
      num_steps_per_batch: Number of steps per batch.
    """
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.global_steps = 0
    self.num_steps_per_batch = num_steps_per_batch

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

  def on_batch_begin(self):
    self.global_steps += 1
    if self.global_steps == 1:
      self.start_time = time.time()
      self.timestamp_log.append(
          BatchTimestamp(self.global_steps, self.start_time))

  def on_batch_end(self):
    """Records elapse time of the batch and calculates examples per second."""
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      steps_per_second = self.log_steps / elapsed_time
      examples_per_second = steps_per_second * self.batch_size
      step_time = elapsed_time / self.log_steps
      self.timestamp_log.append(BatchTimestamp(self.global_steps, timestamp))
      print("BenchmarkMetric: '{{global step':{}, "
            "'steps_per_second':{:.5g}, step_time:{:.5g}, "
            "'examples_per_second':{:.3f}}}".format(self.global_steps,
                                                    steps_per_second, step_time,
                                                    examples_per_second))
      self.start_time = timestamp

  def get_average_examples_per_second(self, warmup=True):
    """Returns average examples per second so far.

    Examples per second are defined by `batch_size` * `num_steps_per_batch`

    Args:
      warmup: If true ignore first set of steps executed as determined by
        `log_steps`.

    Returns:
      Average examples per second.
    """
    return 1 / self.get_average_step_time(
        warmup=warmup) * self.batch_size * self.num_steps_per_batch

  def get_average_step_time(self, warmup=True):
    """Returns average step time (seconds) so far.

    Args:
      warmup: If true ignore first set of steps executed as determined by
        `log_steps`.

    Returns:
      Average step time in seconds.

    """
    if warmup:
      if len(self.timestamp_log) < 3:
        return -1
      elapsed = self.timestamp_log[-1].timestamp - self.timestamp_log[
          1].timestamp
      return elapsed / (self.log_steps * (len(self.timestamp_log) - 2))
    else:
      if len(self.timestamp_log) < 2:
        return -1
      elapsed = self.timestamp_log[-1].timestamp - self.timestamp_log[
          0].timestamp
      return elapsed / (self.log_steps * (len(self.timestamp_log) - 1))


def set_session_config(enable_xla=False):
  """Sets the session config."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)
    # Disable PinToHostOptimizer in grappler when enabling XLA because it
    # causes OOM and performance regression.
    tf.config.optimizer.set_experimental_options(
        {'pin_to_host_optimization': False})


def get_variable_value(agent, name):
  """Returns the value of the trainable variable with the given name."""
  policy_vars = agent.policy.variables()
  tf_vars = [v for v in policy_vars if name in v.name]
  assert tf_vars, 'Variable "{}" does not exist. Found: {}'.format(
      name, policy_vars)
  if tf.executing_eagerly() and len(tf_vars) > 1:
    var = tf_vars[0]
  else:
    assert len(tf_vars) == 1, 'More than one variable with name {}. {}'.format(
        name, [(v.name, v.shape) for v in tf_vars])
    var = tf_vars[0]
  return var.numpy() if tf.executing_eagerly() else var.eval()


def get_initial_values(agent, check_values):
  """Returns the initial values."""
  return [get_variable_value(agent, var_name) for var_name in check_values]


def check_values_changed(agent, initial_values, check_value_changes, name=None):
  """Checks that the initial values."""
  final_values = [get_variable_value(agent, var_name) for var_name in \
                    check_value_changes]
  for var_name, initial, final in zip(check_value_changes, initial_values,
                                      final_values):
    all_close = np.allclose(initial, final)
    assert not all_close, ('[{}] Variable "{}" did not change: {} -> {}'.format(
        name, var_name, initial, final))


def summary_iterator(path: str) -> event_pb2.Event:
  # `tf.data.TFRecordDataset` is not used because it requires Eager or
  # tf.function, which is a state the util should not own. b/174888476
  for record in tf_record.tf_record_iterator(path):
    yield event_pb2.Event.FromString(record)


def find_event_log(eventlog_dir: str,
                   log_file_pattern: str = 'events.out.tfevents.*') -> str:
  """Find the event log in a given folder.

  Expects to find a single log file matching the pattern provided.

  Args:
    eventlog_dir: Event log directory to search.
    log_file_pattern: Pattern to use to find the event log.

  Returns:
    Path to the event log file that was found.

  Raises:
    FileNotFoundError: If an event log is not found in the event log
      directory.
  """
  event_log_path = os.path.join(eventlog_dir, log_file_pattern)

  # In OSS tf.io.gfile.glob throws `NotFoundError` vs returning an empty
  # list. Catching `NotFoundError` and doing the check yields a consistent
  # message.
  try:
    event_files = tf.io.gfile.glob(event_log_path)
  except tf.errors.NotFoundError:
    event_files = []

  if not event_files:
    raise FileNotFoundError(f'No files found matching pattern:{event_log_path}')

  assert len(event_files) == 1, (
      'Found {} event files({}) matching "{}" pattern and expected 1.'.format(
          len(event_files), ','.join(event_files), event_log_path))

  return event_files[0]


def extract_event_log_values(
    event_file: str,
    event_tag: str,
    end_step: Optional[int] = None) -> Tuple[Dict[int, np.generic], float]:
  """Extracts the event values for the `event_tag` and total wall time.

  Args:
    event_file: Path to the event log.
    event_tag: Event to extract from the logs.
    end_step: If set, processing of the event log ends on this step.

  Returns:
    Tuple with a dict of int: np.generic (step: event value) and the total
      walltime in minutes.

  Raises:
    ValueError: If no events are found or the final step is smaller than the
      `end_step` requested.
  """
  start_step = 0
  current_step = 0
  start_time = 0
  max_wall_time = 0.0
  logging.info('Processing event file: %s', event_file)
  event_values = {}
  for summary in summary_iterator(event_file):
    current_step = summary.step
    logging.debug('Event log item: %s', summary)
    for value in summary.summary.value:
      if value.tag == event_tag:
        ndarray = tf.make_ndarray(value.tensor)
        event_values[summary.step] = ndarray.item(0)
        if current_step == start_step:
          start_time = summary.wall_time
          logging.info(
              'training start (step %d): %s', current_step,
              datetime.datetime.fromtimestamp(
                  summary.wall_time).strftime('%Y-%m-%d %H:%M:%S.%f'))
        # Avoids issue of summaries not recorded in order.
        max_wall_time = max(summary.wall_time, max_wall_time)
    if end_step and summary.step >= end_step:
      break

  if not start_time:
    raise ValueError(
        'Error: Starting event not found. Check arg event_name and '
        'warmup_steps. Possible no events were found.')

  if end_step and current_step < end_step:
    raise ValueError('Error: Final step was less than the requested end_step.')

  elapse_time = (max_wall_time - start_time) / 60
  logging.info(
      'training end (step %d): %s', current_step,
      datetime.datetime.fromtimestamp(max_wall_time).strftime(
          '%Y-%m-%d %H:%M:%S.%f'))
  logging.info('elapsed time:%dm', elapse_time)
  return event_values, elapse_time
