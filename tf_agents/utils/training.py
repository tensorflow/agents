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
"""Provide utility functions for training related operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def apply_gradients(optimizer, grads_and_vars, global_step=None):
  """Returns a tf.Operation that applies gradients and incremements the global step.

  Args:
    optimizer: An instance of `tf.compat.v1.train.Optimizer` or
      `tf.keras.optimizers.Optimizer`.
    grads_and_vars: List of `(gradient, variable)` pairs.
    global_step: An integer which corresponds to the number of batches seen by
      the graph.

  Returns:
    A `tf.Operation` that when executed, applies gradients and increments the
    global step.
  """
  with tf.control_dependencies([optimizer.apply_gradients(grads_and_vars)]):
    return global_step.assign_add(1)
