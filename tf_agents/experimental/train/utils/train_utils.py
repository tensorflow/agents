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
"""Utils for distributed training using Actor/Learner API."""

import time
from typing import Callable, Text

from absl import logging

import tensorflow.compat.v2 as tf


def create_train_step() -> tf.Variable:
  return tf.Variable(
      0,
      trainable=False,
      dtype=tf.int64,
      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
      shape=())


# TODO(b/142821173): Test train_utils `wait_for_files` functions.
def wait_for_file(file_path: Text,
                  sleep_time_secs: int = 2,
                  num_retries: int = 86400,
                  sleep: Callable[[int], None] = time.sleep) -> Text:
  """Blocks until the file at `file_path` becomes available.

  The default setting allows a fairly loose, but not infinite wait time of 2
  days for this function to block.

  Args:
    file_path: The path to the file that we are waiting for.
    sleep_time_secs: Number of time in seconds slept between retries.
    num_retries: Number of times the existence of the file is checked.
    sleep: Callable sleep function.

  Returns:
    The original `file_path`.

  Raises:
    TimeoutError: If the file does not become available during the number of
      trials.
  """
  retry = 0
  while (num_retries is None or
         retry < num_retries) and (not tf.io.gfile.exists(file_path) or
                                   tf.io.gfile.stat(file_path).length <= 0):
    logging.info('Waiting for the file to become available:\n\t%s', file_path)
    sleep(sleep_time_secs)
    retry += 1

  if (not tf.io.gfile.exists(file_path) or
      tf.io.gfile.stat(file_path).length <= 0):
    raise TimeoutError(
        'Could not find file {} after {} retries waiting {} seconds between '
        'retries.'.format(file_path, num_retries, sleep_time_secs))

  logging.info('The file %s became available, file length: %s', file_path,
               tf.io.gfile.stat(file_path).length)
  return file_path
