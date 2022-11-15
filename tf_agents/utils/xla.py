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


def compile_method_in_graph_mode(method):
  """Decorator for XLA compilation iff in graph mode and XLA is available.

  Example:

  ```python
  class MyClass(object):
    @compile_in_graph_mode
    def method(self, x, y, z):
      return {'a': x + y, 'b': y * z}

  @common.function
  def calls_fn(inputs):
    MyClass a;
    return a.method(inputs.x, inputs.y, inputs.z)

  # Call calls_fn().

  Args:
    method: A method that accepts a list of possibly nested tensor arguments.
      kwargs and inputs taking the value `None` are not supported.  Non-tensor
      arguments are treated as nest objects, and leaves are converted to
      tensors.

  Returns:
    A function that, when called, checks if XLA is compiled in and enabled
    for the current device, and that it's being built in graph mode, and
    returns an XLA-compiled version of `method`.  If in eager mode, or XLA
    is not available, then `method` is called directly.
  ```
  """
  @functools.wraps(method)
  def _call_compiled(*args, **kwargs):
    self = args[0]
    args = args[1:]
    return _compiled(*args, _fn=method, _self=self, **kwargs)

  return _call_compiled


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
  def _call_compiled(*args, **kwargs):
    return _compiled(*args, _fn=fn, _self=None, **kwargs)

  return  _call_compiled


def _compiled(*args, _fn=None, _self=None, **kwargs):
  """Helper function for optionally XLA compiling `fn`."""
  args = tf.nest.map_structure(tf.convert_to_tensor, args)
  kwargs = tf.nest.map_structure(tf.convert_to_tensor, kwargs)
  if tf.compat.v1.executing_eagerly() or not is_xla_available():
    if _self is not None:
      return _fn(_self, *args, **kwargs)
    else:
      return _fn(*args, **kwargs)
  else:
    @tf.function(jit_compile=True)  # allow-tf-function
    def _call_fn(*args, **kwargs):
      if _self is not None:
        return _fn(_self, *args, **kwargs)
      else:
        return _fn(*args, **kwargs)

    return _call_fn(*args, **kwargs)
