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

"""Implementation of various python metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import gin
import numpy as np
import six

from tf_agents.metrics import py_metric
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage


class NumpyDeque(numpy_storage.NumpyState):
  """Deque implementation using a numpy array as a circular buffer."""

  def __init__(self, maxlen, dtype):
    """Deque using a numpy array as a circular buffer, with FIFO evictions.

    Args:
      maxlen: Maximum length of the deque before beginning to evict the oldest
        entries. If np.inf, deque size is unlimited and the array will grow
        automatically.
      dtype: Data type of deque elements.
    """
    self._start_index = np.int64(0)
    self._len = np.int64(0)
    self._maxlen = np.array(maxlen)
    initial_len = 10 if np.isinf(self._maxlen) else self._maxlen
    self._buffer = np.zeros(shape=(initial_len,), dtype=dtype)

  def clear(self):
    self._start_index = np.int64(0)
    self._len = np.int64(0)

  def add(self, value):
    insert_idx = int((self._start_index + self._len) % self._maxlen)

    # Increase buffer size if necessary.
    if np.isinf(self._maxlen) and insert_idx >= self._buffer.shape[0]:
      self._buffer.resize((self._buffer.shape[0] * 2,))

    self._buffer[insert_idx] = value
    if self._len < self._maxlen:
      self._len += 1
    else:
      self._start_index = np.mod(self._start_index + 1, self._maxlen)

  def extend(self, values):
    for value in values:
      self.add(value)

  def __len__(self):
    return self._len

  def mean(self, dtype=None):
    if self._len == self._buffer.shape[0]:
      return np.mean(self._buffer, dtype=dtype)

    assert self._start_index == 0
    return np.mean(self._buffer[:self._len], dtype=dtype)


@six.add_metaclass(abc.ABCMeta)
class StreamingMetric(py_metric.PyStepMetric):
  """Abstract base class for streaming metrics.

  Streaming metrics keep track of the last (upto) K values of the metric in a
  Deque buffer of size K. Calling result() will return the average value of the
  items in the buffer.
  """

  def __init__(self, name='StreamingMetric', buffer_size=10, batch_size=None):
    super(StreamingMetric, self).__init__(name)
    self._buffer = NumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self._batch_size = batch_size
    self.reset()

  def reset(self):
    self._buffer.clear()
    if self._batch_size:
      self._reset(self._batch_size)

  @abc.abstractmethod
  def _reset(self, batch_size):
    """Reset stat gathering variables in child classes."""

  def add_to_buffer(self, values):
    """Appends new values to the buffer."""
    self._buffer.extend(values)

  def result(self):
    """Returns the value of this metric."""
    if self._buffer:
      return self._buffer.mean(dtype=np.float32)
    return np.array(0.0, dtype=np.float32)

  @abc.abstractmethod
  def _batched_call(self, trajectory):
    """Call with trajectory always batched."""

  def call(self, trajectory):
    if not self._batch_size:
      if trajectory.step_type.ndim == 0:
        self._batch_size = 1
      else:
        assert trajectory.step_type.ndim == 1
        self._batch_size = trajectory.step_type.shape[0]
      self.reset()
    if trajectory.step_type.ndim == 0:
      trajectory = nest_utils.batch_nested_array(trajectory)
    self._batched_call(trajectory)


@gin.configurable
class AverageReturnMetric(StreamingMetric):
  """Computes the average undiscounted reward."""

  def __init__(self, name='AverageReturn', buffer_size=10, batch_size=None):
    """Creates an AverageReturnMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.episode_return so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_return = np.float64(0)
    super(AverageReturnMetric, self).__init__(name, buffer_size=buffer_size,
                                              batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_return = np.zeros(
        shape=(batch_size,), dtype=np.float64)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_return = self._np_state.episode_return

    is_first = np.where(trajectory.is_first())
    episode_return[is_first] = 0

    episode_return += trajectory.reward

    is_last = np.where(trajectory.is_last())
    self.add_to_buffer(episode_return[is_last])


@gin.configurable
class AverageEpisodeLengthMetric(StreamingMetric):
  """Computes the average episode length."""

  def __init__(self, name='AverageEpisodeLength', buffer_size=10,
               batch_size=None):
    """Creates an AverageEpisodeLengthMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.episode_return so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_steps = np.float64(0)
    super(AverageEpisodeLengthMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_steps = np.zeros(
        shape=(batch_size,), dtype=np.float64)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_steps = self._np_state.episode_steps

    # Each non-boundary trajectory (first, mid or last) represents a step.
    episode_steps[np.where(~trajectory.is_boundary())] += 1
    self.add_to_buffer(episode_steps[np.where(trajectory.is_last())])
    episode_steps[np.where(trajectory.is_last())] = 0


@gin.configurable
class EnvironmentSteps(py_metric.PyStepMetric):
  """Counts the number of steps taken in the environment."""

  def __init__(self, name='EnvironmentSteps'):
    super(EnvironmentSteps, self).__init__(name)
    self._np_state = numpy_storage.NumpyState()
    self.reset()

  def reset(self):
    self._np_state.environment_steps = np.int64(0)

  def result(self):
    return self._np_state.environment_steps

  def call(self, trajectory):
    if trajectory.step_type.ndim == 0:
      trajectory = nest_utils.batch_nested_array(trajectory)

    new_steps = np.sum((~trajectory.is_boundary()).astype(np.int64))
    self._np_state.environment_steps += new_steps


@gin.configurable
class NumberOfEpisodes(py_metric.PyStepMetric):
  """Counts the number of episodes in the environment."""

  def __init__(self, name='NumberOfEpisodes'):
    super(NumberOfEpisodes, self).__init__(name)
    self._np_state = numpy_storage.NumpyState()
    self.reset()

  def reset(self):
    self._np_state.number_episodes = np.int64(0)

  def result(self):
    return self._np_state.number_episodes

  def call(self, trajectory):
    if trajectory.step_type.ndim == 0:
      trajectory = nest_utils.batch_nested_array(trajectory)

    completed_episodes = np.sum(trajectory.is_last().astype(np.int64))
    self._np_state.number_episodes += completed_episodes


@gin.configurable
class CounterMetric(py_metric.PyMetric):
  """Metric to track an arbitrary counter.

  This is useful for, e.g., tracking the current train/eval iteration number.

  To increment the counter, you can __call__ it (e.g. metric_obj()).
  """

  def __init__(self, name='Counter'):
    super(CounterMetric, self).__init__(name)
    self._np_state = numpy_storage.NumpyState()
    self.reset()

  def reset(self):
    self._np_state.count = np.int64(0)

  def call(self):
    self._np_state.count += 1

  def result(self):
    return self._np_state.count
