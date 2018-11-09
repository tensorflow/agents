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

"""Timing utility for TF-Agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


class Timer(object):
  """Context manager to time blocks of code."""

  def __init__(self):
    self._accumulator = 0
    self._last = None

  def __enter__(self):
    self.start()

  def __exit__(self, *args):
    self.stop()

  def start(self):
    self._last = time.time()

  def stop(self):
    self._accumulator += time.time() - self._last

  def value(self):
    return self._accumulator

  def reset(self):
    self._accumulator = 0
