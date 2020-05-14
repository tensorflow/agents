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

"""Tensorflow RL Agent RNN utilities.

This module provides helper functions that Agents can use to train
RNN-based policies.

`DynamicUnroll`

The layer `DynamicUnroll` allows an Agent to train an RNN-based policy
by running an RNN over a batch of episode chunks from a replay buffer.

The agent creates a subclass of `tf.contrib.rnn.LayerRNNCell` or a Keras RNN
cell, such as `tf.keras.layers.LSTMCell`, instances of which
which can themselves be wrappers of `RNNCell`.  Training this instance
involes passing it to `DynamicUnroll` constructor; and then pass a set of
episode tensors in the form of `inputs`.

See the unit tests in `rnn_utils_test.py` for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import common

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import tensor_shape  # TF internal
from tensorflow.python.keras import layers  # TF internal
# pylint:enable=g-direct-tensorflow-import

__all__ = ["DynamicUnroll"]


def _maybe_tensor_shape_from_tensor(shape):
  if isinstance(shape, tf.Tensor):
    return tensor_shape.as_shape(tf.get_static_value(shape))
  else:
    return shape


def _infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.

  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.

  Returns:
    dtype: inferred dtype of hidden state.

  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif tf.nest.is_nested(state):
    inferred_dtypes = [element.dtype for element in tf.nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype


def _best_effort_input_batch_size(flat_input):
  """Get static input batch size if available, with fallback to the dynamic one.

  Args:
    flat_input: An iterable of time major input Tensors of shape
      `[max_time, batch_size, ...]`.
    All inputs should have compatible batch sizes.

  Returns:
    The batch size in Python integer if available, or a scalar Tensor otherwise.

  Raises:
    ValueError: if there is any input with an invalid shape.
  """
  for input_ in flat_input:
    shape = input_.shape
    if shape.rank is None:
      continue
    if shape.rank < 2:
      raise ValueError(
          "Expected input tensor %s to have rank at least 2" % input_)
    batch_size = shape.dims[1].value
    if batch_size is not None:
      return batch_size
  # Fallback to the dynamic batch size of the first input.
  return tf.shape(input=flat_input[0])[1]


class DynamicUnroll(tf.keras.layers.Layer):
  """Process a history of sequences that are concatenated without padding.

  Given batched, batch-major `inputs`, `DynamicUnroll` unrolls
  an RNN using `cell`; at each time step it feeds a frame of `inputs` as input
  to `cell.call()`.

  Assuming all tensors in `inputs` are shaped `[batch_size, n, ...]` where
  `n` is the number of time steps, the RNN will run for exactly `n` steps.

  If `n == 1` is known statically, then only a single step is executed.
  This is done via a static unroll without using `tf.while_loop`.

  **NOTE** As the call() method requires that the user provides a mask argument,
  this Layer may not be used within a Keras Sequential model.  Instead, the user
  must manually create this layer and hook its inputs and outputs manually.
  This is the expected use case: create the layer inside the `__init__` of a
  subclass of Keras `Network`, then apply it manually inside the Network's
  `call`.
  """

  def __init__(self, cell, parallel_iterations=20, swap_memory=None,
               **kwargs):
    """Create a `DynamicUnroll` layer.

    Args:
      cell: A `tf.nn.rnn_cell.RNNCell` or Keras `RNNCell` (e.g. `LSTMCell`)
        whose `call()` method has the signature `call(input, state, ...)`.
        Each tensor in the tuple is shaped `[batch_size, ...]`.
      parallel_iterations: Parallel iterations to pass to `tf.while_loop`.
        The default value is a good trades off between memory use and
        performance.  See documentation of `tf.while_loop` for more details.
      swap_memory: Python bool.  Whether to swap memory from GPU to CPU when
        storing activations for backprop.  This may sometimes have a negligible
        performance impact, but can improve memory usage.  See documentation
        of `tf.while_loop` for more details.
      **kwargs: Additional layer arguments, such as `dtype` and `name`.

    Raises:
      TypeError: if `cell` lacks `get_initial_state`, `output_size`, or
        `state_size` property.
    """
    if getattr(cell, "get_initial_state", None) is None:
      raise TypeError("cell lacks get_initial_state method: %s" % cell)
    if getattr(cell, "output_size", None) is None:
      raise TypeError("cell lacks output_size property: %s" % cell)
    if getattr(cell, "state_size", None) is None:
      raise TypeError("cell lacks state_size property: %s" % cell)
    self.cell = cell
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory
    super(DynamicUnroll, self).__init__(**kwargs)

  def get_config(self):
    config = {
        "parallel_iterations": self.parallel_iterations,
        "swap_memory": self.swap_memory,
        "cell": {
            "class_name": self.cell.__class__.__name__,
            "config": self.cell.get_config()
        }
    }
    base_config = dict(super(DynamicUnroll, self).get_config())
    base_config.update(config)
    return base_config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    cell = layers.deserialize(config.pop("cell"), custom_objects=custom_objects)
    layer = cls(cell, **config)
    return layer

  def compute_output_shape(self, input_shape):
    return self.cell.compute_output_shape(input_shape)

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    return self.cell.trainable_weights

  @property
  def non_trainable_weights(self):
    if not self.trainable:
      return self.cell.weights
    return self.cell.non_trainable_weights

  @property
  def losses(self):
    layer_losses = super(DynamicUnroll, self).losses
    return self.cell.losses + layer_losses

  @property
  def updates(self):
    updates = self.cell.updates
    return updates + self._updates

  def build(self, input_shape):
    self.cell.build(input_shape)
    self.built = True

  def call(self, inputs, reset_mask, initial_state=None, training=False):
    """Perform the computation.

    Args:
      inputs: A tuple containing tensors in batch-major format,
        each shaped `[batch_size, n, ...]`.
      reset_mask: A `bool` matrix shaped `[batch_size, n]`, describing the
        locations for which the state will be reset to zeros.  Typically this is
        the value `time_steps.is_first()` where `time_steps` is a `TimeStep`
        containing tensors of the shape `[batch_size, n, ...]`.
        The `zero_state` of the cell will be used whenever `reset` is `True`,
        instead of either the current state or the `initial_state`.
      initial_state: (optional) An initial state for `cell`.  If not provided,
        `dtype` must be set and `cell.get_initial_state()` is used instead.
      training: Whether the output is being used for training.

    Returns:
      A 2-tuple `(outputs, final_state)` where:

       - `outputs` contains the outputs for all states of the unroll; this is
         either a tensor or nested tuple with tensors all shaped
         `[n, batch_size, ...]`,
         with structure and shape matching `cell.output_size`.
       - `final_state` contains the final state of the unroll; with structure
         and shape matching `cell.state_size`.

    Raises:
      ValueError: if static batch sizes within input tensors don't match.
      ValueError: if `initial_state` is `None` and `self.dtype` is `None`.
    """
    if not initial_state and self.dtype is None:
      raise ValueError("Must provide either dtype or initial_state")

    # Assume all inputs are batch major.  Convert to time major.
    inputs = tf.nest.map_structure(common.transpose_batch_time, inputs)
    inputs_flat = tf.nest.flatten(inputs)
    inputs_static_shapes = tuple(x.shape for x in inputs_flat)
    batch_size = _best_effort_input_batch_size(inputs_flat)
    const_batch_size = tensor_shape.dimension_value(inputs_static_shapes[0][1])

    # reset_mask is batch major.  Convert to time major.
    reset_mask = tf.transpose(a=reset_mask)

    for shape in inputs_static_shapes:
      got_batch_size = tensor_shape.dimension_value(shape[1])
      if const_batch_size is None:
        const_batch_size = got_batch_size
      if got_batch_size is not None and const_batch_size != got_batch_size:
        raise ValueError(
            "batch_size is not the same for all the elements in the input. "
            "Saw values %s and %s" % (const_batch_size, got_batch_size))

    if not initial_state:
      dtype = self.dtype
      initial_state = zero_state = self.cell.get_initial_state(
          batch_size=batch_size, dtype=self.dtype)
    else:
      dtype = _infer_state_dtype(self.dtype, initial_state)
      zero_state = self.cell.get_initial_state(
          batch_size=batch_size, dtype=dtype)

    # Try to get the iteration count statically; if that's not possible,
    # access it dynamically at runtime.
    iterations = tensor_shape.dimension_value(inputs_flat[0].shape[0])
    iterations = iterations or tf.shape(input=inputs_flat[0])[0]

    if not tf.is_tensor(iterations) and iterations == 1:
      # Take exactly one time step
      return _static_unroll_single_step(
          self.cell,
          inputs,
          reset_mask,
          state=initial_state,
          zero_state=zero_state,
          training=training)
    else:
      return _dynamic_unroll_multi_step(
          self.cell,
          inputs,
          reset_mask,
          initial_state=initial_state,
          zero_state=zero_state,
          dtype=dtype,
          parallel_iterations=self.parallel_iterations,
          swap_memory=self.swap_memory,
          iterations=iterations,
          const_batch_size=const_batch_size,
          training=training)


def _maybe_reset_state(reset, s_zero, s):
  if not isinstance(s, tf.TensorArray) and s.shape.rank > 0:
    return tf.compat.v1.where(reset, s_zero, s)
  else:
    return s


def _static_unroll_single_step(cell,
                               inputs,
                               reset_mask,
                               state,
                               zero_state,
                               training):
  """Helper for dynamic_unroll which runs a single step."""
  def _squeeze(t):
    if not isinstance(t, tf.TensorArray) and t.shape.rank > 0:
      return tf.squeeze(t, [0])
    else:
      return t

  # Remove time dimension.
  inputs = tf.nest.map_structure(_squeeze, inputs)
  reset_mask = _squeeze(reset_mask)

  state = tf.nest.map_structure(
      lambda s, s_zero: _maybe_reset_state(reset_mask, s_zero, s), state,
      zero_state)

  outputs, final_state = cell(inputs, state, training=training)
  outputs = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1), outputs)

  return (outputs, final_state)


def _dynamic_unroll_multi_step(cell,
                               inputs,
                               reset_mask,
                               initial_state,
                               zero_state,
                               dtype,
                               parallel_iterations,
                               swap_memory,
                               iterations,
                               const_batch_size,
                               training):
  """Helper for dynamic_unroll which uses a tf.while_loop."""

  # Convert all inputs to TensorArrays
  def ta_and_unstack(x):
    return (tf.TensorArray(dtype=x.dtype,
                           size=iterations,
                           element_shape=x.shape[1:])
            .unstack(x))

  inputs_tas = tf.nest.map_structure(ta_and_unstack, inputs)
  reset_mask_ta = ta_and_unstack(reset_mask)

  # Create a TensorArray for each output
  def create_output_ta(s):
    return tf.TensorArray(
        dtype=_infer_state_dtype(dtype, initial_state),
        size=iterations,
        element_shape=(tf.TensorShape([const_batch_size])
                       .concatenate(_maybe_tensor_shape_from_tensor(s))))

  output_tas = tf.nest.map_structure(create_output_ta, cell.output_size)

  def pred(time, *unused_args):
    return time < iterations

  def body(time, state, output_tas):
    """Internal while_loop body.

    Args:
      time: time
      state: rnn state @ time
      output_tas: output tensorarrays

    Returns:
      - time + 1
      - state: rnn state @ time + 1
      - output_tas: output tensorarrays with values written @ time
      - masks_ta: optional mask tensorarray with mask written @ time
    """
    input_ = tf.nest.map_structure(lambda ta: ta.read(time), inputs_tas)
    is_reset = reset_mask_ta.read(time)
    state = tf.nest.map_structure(
        lambda s_zero, s: _maybe_reset_state(is_reset, s_zero, s), zero_state,
        state)

    outputs, next_state = cell(input_, state, training=training)

    output_tas = tf.nest.map_structure(lambda ta, x: ta.write(time, x),
                                       output_tas, outputs)

    return (time + 1, next_state, output_tas)

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with tf.compat.v1.variable_scope(
      tf.compat.v1.get_variable_scope()) as varscope:
    if (not tf.executing_eagerly() and varscope.caching_device is None):
      varscope.set_caching_device(lambda op: op.device)

    _, final_state, output_tas = (
        tf.while_loop(
            cond=pred,
            body=body,
            loop_vars=(tf.constant(0, name="time"), initial_state, output_tas),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            maximum_iterations=iterations))

  outputs = tf.nest.map_structure(lambda ta: ta.stack(), output_tas)

  if isinstance(iterations, int):
    # TensorArray.stack() doesn't set a static value for dimension 0,
    # even if the size is known. Set the shapes here.
    iterations_shape = tf.TensorShape([iterations])
    tf.nest.map_structure(
        lambda t: t.set_shape(iterations_shape.concatenate(t.shape[1:])),
        outputs)

  # Convert everything back to batch major
  outputs = tf.nest.map_structure(common.transpose_batch_time, outputs)

  return (outputs, final_state)
