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

# Lint as: python3
"""Utility that Triggers every n calls."""

from typing import Callable

from absl import logging


class IntervalTrigger(object):
  """Triggers on every fixed interval."""

  def __init__(self, interval: int, fn: Callable[[], None], start: int = 0):
    """Constructs the IntervalTrigger.

    Args:
      interval: The triggering interval.
      fn: callable with no arguments that gets triggered.
      start: An initial value for the trigger.
    """
    self._interval = interval
    self._original_start_value = start
    self._last_trigger_value = start
    self._fn = fn

  def __call__(self, value: int, force_trigger: bool = False) -> None:
    """Maybe trigger the event based on the interval.

    Args:
      value: the value for triggering.
      force_trigger: If True, the trigger will be forced triggered unless the
        last trigger value is equal to `value`.
    """
    if self._interval <= 0:
      logging.info(
          'IntervalTrigger was not triggered because interval is set to %d',
          self._interval)
      return

    if (force_trigger and value != self._last_trigger_value) or (
        value >= self._last_trigger_value + self._interval):
      self._last_trigger_value = value
      self._fn()

  def reset(self) -> None:
    """Resets the trigger interval."""
    self._last_trigger_value = self._original_start_value
