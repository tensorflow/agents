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

"""XLA utilities for TF-Agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

# Dictionary mapping a device name to a python bool.
_IS_XLA_AVAILABLE = {}


def is_xla_available():
  """Is XLA compilation available for the current device context?"""

  global _IS_XLA_AVAILABLE
  # There's unfortunately no cleaner way to get the device other than creating a
  # new op and querying it.
  with tf.name_scope("is_xla_available"):
    device = tf.constant(0.0).device
  if device not in _IS_XLA_AVAILABLE:
    try:
      # Take ourselves outside of any tf.function calls.
      with tf.init_scope():
        # Create temporary xla subgraph
        with tf.compat.v1.Graph().as_default():
          # We'll use a session so we can be compatible with both TF1 and TF2
          with tf.compat.v1.Session() as sess:
            # Check for XLA on the given device.
            with tf.device(device):
              sess.run(tf.xla.experimental.compile(lambda: tf.constant(0.0)))
    except (ValueError, tf.errors.InvalidArgumentError):
      _IS_XLA_AVAILABLE[device] = False
    else:
      _IS_XLA_AVAILABLE[device] = True
  return _IS_XLA_AVAILABLE[device]


def compile_in_graph_mode(fn):
  """Decorator for XLA compilation iff in graph mode and XLA is available.

  Example:

  ```python
  @compile_in_graph_mode
  def fn(x, y, z):
    return {'a': x + y, 'b': y * z}

  @common.function
  def calls_fn(inputs):
    return fn(inputs.x, inputs.y, inputs.z)

  # Call calls_fn().

  Args:
    fn: A callable that accepts a list of possibly nested tensor arguments.
      kwargs and inputs taking the value `None` are not supported.  Non-tensor
      arguments are treated as nest objects, and leaves are converted to
      tensors.

  Returns:
    A function that, when called, checks if XLA is compiled in and enabled
    for the current device, and that it's being built in graph mode, and
    returns an XLA-compiled version of `fn`.  If in eager mode, or XLA
    is not available, then `fn` is called directly.
  ```
  """

  @functools.wraps(fn)
  def _compiled(*args, **kwargs):
    """Helper function for optionally XLA compiling `fn`."""
    if kwargs:
      raise ValueError(
          "kwargs are not supported for functions that are XLA-compiled, "
          "but saw kwargs: {}".format(kwargs))
    args = tf.nest.map_structure(tf.convert_to_tensor, args)
    if tf.compat.v1.executing_eagerly() or not is_xla_available():
      return fn(*args)
    else:
      # The flattening/unpacking is necessary because xla compile only allows
      # flat inputs and outputs: no substructures.  But we provide support for
      # nested inputs and outputs.
      outputs_for_structure = [None]
      flat_args = tf.nest.flatten(args)
      def _fn(*flattened_args):
        unflattened_args = tf.nest.pack_sequence_as(args, flattened_args)
        fn_outputs = fn(*unflattened_args)
        outputs_for_structure[0] = fn_outputs
        return tf.nest.flatten(fn_outputs)
      outputs = tf.xla.experimental.compile(_fn, flat_args)
      return tf.nest.pack_sequence_as(outputs_for_structure[0], outputs)

  return _compiled
