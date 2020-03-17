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

# Lint as: python2, python3
"""Utilities for running benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


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
