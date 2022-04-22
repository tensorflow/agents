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

"""Initialization code for colab test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os
import time


def WaitForFilePath(path_pattern, timeout_sec):
  start = time.time()
  result = []
  while not result:
    if time.time() - start > timeout_sec:
      return result
    result = glob.glob(path_pattern)
    time.sleep(0.1)
  return result


def SetDisplayFromWebTest():
  """Set up display from web test.

  Colab test sets up display using xvfb for front end web test suite. We just
  ensure that DISPLAY environment variable is properly set for colab kernel
  (backend) which can be used for open gym environment rendering.
  """

  res = WaitForFilePath("/tmp/.X11-unix", 60)
  assert res

  pattern = "/tmp/.X11-unix/X*"
  res = WaitForFilePath(pattern, 60)
  assert res

  # If we find "/tmp/.X11-unix/X1", then we will set DISPLAY to be ":1".
  display = ":" + res[0][len(pattern)-1:]
  os.environ["DISPLAY"] = display
  logging.info("Set DISPLAY=%s", display)


SetDisplayFromWebTest()
