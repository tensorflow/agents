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

# Lint as: python3
"""Utility class to keep track of global training steps per second."""

import time


class StepPerSecondTracker(object):
  """Utility class for measuring steps/second."""

  def __init__(self, step):
    """Creates an instance of the StepPerSecondTracker.

    Args:
      step: `tf.Variable` holding the current value for the number of train
        steps.
    """
    self.step = step
    self.last_iteration = 0
    self.last_time = 0
    self.restart()

  def restart(self):
    self.last_iteration = self.step.numpy()
    self.last_time = time.time()

  def steps_per_second(self):
    value = ((self.step.numpy() - self.last_iteration) /
             (time.time() - self.last_time))
    return value
