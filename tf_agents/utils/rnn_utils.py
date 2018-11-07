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

dynamic_unroll

The function dynamic_unroll allows an Agent to train an RNN-based policy
by running an RNN over a batch of episode chunks from a replay buffer.

The agent creates a subclass of `tf.contrib.rnn.LayerRNNCell`, instances of
which can themselves contain instances of `RNNCell`.  Training this instance
involes passing it to `dynamic_unroll` with a set of episode tensors in
the form of `time_steps`, `policy_states`, and `actions`.  Each cell `call()`
will have access to the current replay frame's actions and current and next
frames' time_step and policy_state.  Backprop through the final outputs and
state of the RNN is supported, and can be used to update the parameters of
a new policy.

See the unit tests in `rnn_utils_test.py` for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import tensor_shape  # TF internal

# pylint: disable=invalid-name
LayerRNNCell = tf.contrib.rnn.LayerRNNCell
# pylint: enable=invalid-name

nest = tf.contrib.framework.nest


def _maybe_tensor_shape_from_tensor(shape):
  if isinstance(shape, tf.Tensor):
    return tensor_shape.as_shape(tf.contrib.util.constant_value(shape))
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
  elif nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in nest.flatten(state)]
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


def _range_with_reset_mask_scan(reset_mask):
  """Like range_with_reset_mask, but almost always slower.

  This function may be the one we use for XLA/TPU computation, but for now
  it is simply here to clarify how the real function works.

  Args:
    reset_mask: See documentation of `range_with_reset_mask`.

  Returns:
    ranges: See documentation of `range_with_reset_mask`.
  """
  reset_mask = tf.cast(reset_mask, tf.bool)
  batch_size = tf.shape(reset_mask)[0]
  initializer = tf.zeros((batch_size,), dtype=tf.int64)
  return tf.transpose(
      tf.scan(
          lambda a, x: tf.where(x, initializer, a + 1),
          tf.transpose(reset_mask),
          initializer=initializer - 1))  # Start with -1s for counter.


def range_with_reset_mask(reset_mask):
  """Converts a reset mask to ranges that restart counting after each reset.

  Example usage:

  ```python
  reset_mask = np.array(  # This could be a bool array too
      [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 1]])
  expected = np.array(
      [[0, 1, 0, 1, 2, 3, 0, 1, 2, 3],
       [0, 1, 2, 0, 1, 2, 0, 1, 0, 1],
       [0, 0, 0, 0, 1, 2, 3, 4, 5, 0]])

  assertAllEqual(expected, session.run(range_with_reset_mask(reset_mask)))
  ```

  Args:
    reset_mask: matrix shaped `[batch_size, n]`, any numeric or bool dtype.
      For convenience, numeric matrices are treated as reset matrices:
      zero values are treated as no reset, non-zero values are treated as reset.

  Returns:
    ranges: int64 matrix shaped `[batch_size, n]` containing ranges (along rows)
      that reset to 0 wherever `reset_mask` is `True` / nonzero.

  Raises:
    InvalidArgument: (at runtime) if `n == 0`.
  """
  reset_mask = tf.convert_to_tensor(reset_mask)
  shape = tf.to_int64(tf.shape(reset_mask))
  indices = tf.where(reset_mask)

  def _with_resets():
    """Perform the calculation for non-empty resets.

    Example calculation:

    ```python
    reset_mask = [[0, 1, 1], [1, 0, 1]]
    indices = [[0, 1], [0, 2], [1, 0], [1, 2]]
    indices_diff = concat(([[1, -1]], [[0, 1], [1, -2], [0, 2]]))
    col_ix_diff_clean = where([1, 0, 1, 0] > 0, [1, 2, 0, 2], [-1, 1, -2, 2])
                      = [1, 1, 0, 2]
    col_ix_reset_count = [[0, 1, 1], [0, 0, 2]]
    counter = [[1, 1, 1], [1, 1, 1]] - [[0, 1, 1], [0, 0, 2]]
            = [[1, 0, 0], [1, 1, -1]]
    ranges_after_reset = [[1, 1, 1], [1, 2, 1]] - 1
                       = [[0, 0, 0], [0, 1, 0]]

    Returns:
      ranges_after_reset
    """
    # Diff of indices (identify distances between resets); row column is
    # negative at row transitions; second column contains column distances.
    indices_diff = tf.concat(
        ([[1, -1]], indices[1:] - indices[:-1]),
        axis=0)
    # Replace row transition diff values with the col ix at that location.
    col_ix_diff_clean = tf.where(
        indices_diff[:, 0] > 0, indices[:, 1], indices_diff[:, 1])
    # Store these diffs at the mask locations.
    col_ix_reset_count = tf.scatter_nd(indices, col_ix_diff_clean, shape)
    # Create a counter that adds 1 for every col, but resets by the
    # "between reset" count at the reset locations.
    counter = tf.ones_like(reset_mask, dtype=tf.int64) - col_ix_reset_count
    # Accumulate the counts to get a 1-based column counter that resets
    # wherever reset_mask != 0.  Then convert to a 0-based range.
    ranges_after_reset = tf.cumsum(counter, axis=1) - 1
    return ranges_after_reset

  def _with_no_resets():
    batch_size = shape[0]
    n = shape[1]
    ranges_row = tf.expand_dims(tf.range(n, dtype=tf.int64), 0)
    return tf.tile(ranges_row, tf.stack([batch_size, 1]))

  return tf.cond(tf.size(indices) > 0, _with_resets, _with_no_resets)


def dynamic_unroll(cell,
                   inputs,
                   reset_mask,
                   initial_state=None,
                   dtype=None,
                   parallel_iterations=20,
                   swap_memory=None,
                   mask_fn=None):
  """Process a history of sequences that are concatenated without padding.

  Given batched, batch-major `inputs`, `dynamic_unroll` unrolls
  an RNN using `cell`; at each time step it feeds a frame of `inputs` as input
  to `cell.call()`.

  Assuming all tensors in `inputs` are shaped `[batch_size, n, ...]` where
  `n` is the number of time steps, the RNN will run for exactly `n` steps.

  If `n == 1` is known statically, then only a single step is executed.
  This is done via a static unroll without using `tf.while_loop`.

  Args:
    cell: A `tf.nn.rnn_cell.RNNCell` or Keras `RNNCell` (e.g. `LSTMCell`)
      whose `call()` method has the signature `call(input, state, ...)`.
      Each tensor in the tuple is shaped `[batch_size, ...]`.
    inputs: A tuple containing tensors in time-major format,
      each shaped `[batch_size, n, ...]`.
    reset_mask: A `bool` matrix shaped `[batch_size, n]`, describing the
      locations for which the state will be reset to zeros.  Typically this is
      the value `time_steps.is_first()` where `time_steps` is a `TimeStep`
      containing tensors of the shape `[batch_size, n, ...]`.  The `zero_state`
      of the cell will be used whenever `reset` is `True`, instead of either
      the current state or the `initial_state`.
    initial_state: (optional) An initial state for `cell`.  If not provided,
      `dtype` must be set and `cell.get_initial_state()` is used instead.
    dtype: (optional) dtype argument for `cell.get_initial_state()` if
      `initial_state` is not provided.
    parallel_iterations: Parallel iterations to pass to `tf.while_loop`.
      The default value is a good trades off between memory use and performance.
      See documentation of `tf.while_loop` for more details.
    swap_memory: Python bool.  Whether to swap memory from GPU to CPU when
      storing activations for backprop.  This may sometimes have a negligible
      performance impact, but can improve memory usage.  See documentation
      of `tf.while_loop` for more details.
    mask_fn: (optional) A function with the signature
      `m = mask_fn(step_time, episode_time)`, where:

      - `step_time` is a scalar `int32`: current unroll index.
      - `episode_time` is a vec `int32` shaped `[batch_size]`: the number
        of steps taken since either the beginning of unroll or the last
        occurrence of `StepTime.LAST`, whichever is more recent to the current
        time_step.

      And the return tensor `m` is expected to be a `float32` vector shaped
      `[batch_size]`, taking on values between `0.0` and `1.0`.

  Returns:
    A 3-tuple `(outputs, final_state, mask)` where:

     - `outputs` contains the outputs for all states of the unroll; this is
       either a tensor or nested tuple with tensors all shaped
       `[n, batch_size, ...]`,
       with structure and shape matching `cell.output_size`.
     - `final_state` contains the final state of the unroll; with structure
       and shape matching `cell.state_size`.
     - `masks` is a `[n, batch_size]` `float32` tensor containing stacked
       results of each call to `mask_fn`.  If `mask_fn` was `None`, then
       `masks` is `None`.

  Raises:
    ValueError: if dtype or initial_state is not provided.
    ValueError: if static batch sizes within input tensors don't match.
    TypeError: if cell lacks `get_initial_state`, `output_size`, or `state_size`
      property.
    TypeError: if mask_fn is provided but not callable.
  """
  with tf.name_scope("dynamic_unroll"):
    if getattr(cell, "get_initial_state", None) is None:
      raise TypeError("cell lacks get_initial_state method: %s" % cell)
    if getattr(cell, "output_size", None) is None:
      raise TypeError("cell lacks output_size property: %s" % cell)
    if getattr(cell, "state_size", None) is None:
      raise TypeError("cell lacks state_size property: %s" % cell)
    if mask_fn and not callable(mask_fn):
      raise TypeError("mask_fn is not callable: %s" % mask_fn)
    if initial_state is None and dtype is None:
      raise ValueError("Must provide either dtype or initial_state")

    # Assume all inputs are batch major.  Convert to time major.
    inputs = nest.map_structure(tf.contrib.rnn.transpose_batch_time, inputs)
    inputs_flat = nest.flatten(inputs)
    inputs_static_shapes = tuple(x.shape for x in inputs_flat)
    batch_size = tf.contrib.rnn.best_effort_input_batch_size(inputs_flat)
    const_batch_size = inputs_static_shapes[0][1].value

    # reset_mask is batch major.  Convert to time major.
    reset_mask = tf.transpose(reset_mask)

    for shape in inputs_static_shapes:
      got_batch_size = shape[1].value
      if const_batch_size is None:
        const_batch_size = got_batch_size
      if got_batch_size is not None and const_batch_size != got_batch_size:
        raise ValueError(
            "batch_size is not the same for all the elements in the input. "
            "Saw values %s and %s" % (const_batch_size, got_batch_size))

    if initial_state is None:
      initial_state = zero_state = cell.get_initial_state(
          batch_size=batch_size, dtype=dtype)
    else:
      dtype = _infer_state_dtype(dtype, initial_state)
      zero_state = cell.get_initial_state(batch_size=batch_size, dtype=dtype)

    # Try to get the iteration count statically; if that's not possible,
    # access it dynamically at runtime.
    iterations = inputs_flat[0].shape[0].value or tf.shape(inputs_flat[0])[0]

    if not tf.contrib.framework.is_tensor(iterations) and iterations == 1:
      # Take exactly one time step
      return _static_unroll_single_step(
          cell,
          inputs,
          reset_mask,
          state=initial_state,
          zero_state=zero_state,
          mask_fn=mask_fn,
          batch_size=batch_size)
    else:
      return _dynamic_unroll_multi_step(
          cell,
          inputs,
          reset_mask,
          initial_state=initial_state,
          zero_state=zero_state,
          dtype=dtype,
          mask_fn=mask_fn,
          parallel_iterations=parallel_iterations,
          swap_memory=swap_memory,
          iterations=iterations,
          batch_size=batch_size,
          const_batch_size=const_batch_size)


def _maybe_reset_state(reset, s_zero, s):
  if not isinstance(s, tf.TensorArray) and s.shape.ndims > 0:
    return tf.where(reset, s_zero, s)
  else:
    return s


def _static_unroll_single_step(cell,
                               inputs,
                               reset_mask,
                               state,
                               zero_state,
                               mask_fn,
                               batch_size):
  """Helper for dynamic_unroll which runs a single step."""
  def _squeeze(t):
    if not isinstance(t, tf.TensorArray) and t.shape.ndims > 0:
      return tf.squeeze(t, [0])
    else:
      return t

  # Remove time dimension.
  inputs = nest.map_structure(_squeeze, inputs)
  reset_mask = _squeeze(reset_mask)

  state = nest.map_structure(
      lambda s, s_zero: _maybe_reset_state(reset_mask, s_zero, s),
      state,
      zero_state)

  outputs, final_state = cell(inputs, state)
  outputs = nest.map_structure(lambda t: tf.expand_dims(t, 1), outputs)

  if mask_fn:
    time_since_reset = tf.zeros((batch_size,), dtype=tf.int32)
    mask = mask_fn(0, time_since_reset)
  else:
    mask = None

  return (outputs, final_state, mask)


def _dynamic_unroll_multi_step(cell,
                               inputs,
                               reset_mask,
                               initial_state,
                               zero_state,
                               dtype,
                               mask_fn,
                               parallel_iterations,
                               swap_memory,
                               iterations,
                               batch_size,
                               const_batch_size):
  """Helper for dynamic_unroll which uses a tf.while_loop."""

  # Convert all inputs to TensorArrays
  def ta_and_unstack(x):
    return (tf.TensorArray(dtype=x.dtype,
                           size=iterations,
                           element_shape=x.shape[1:])
            .unstack(x))

  inputs_tas = nest.map_structure(ta_and_unstack, inputs)
  reset_mask_ta = ta_and_unstack(reset_mask)

  # Create a TensorArray for each output
  def create_output_ta(s):
    return tf.TensorArray(
        dtype=_infer_state_dtype(dtype, initial_state),
        size=iterations,
        element_shape=(tf.TensorShape([const_batch_size])
                       .concatenate(_maybe_tensor_shape_from_tensor(s))))

  output_tas = nest.map_structure(create_output_ta, cell.output_size)

  if mask_fn:
    masks_ta = tf.TensorArray(
        dtype=tf.float32,
        size=iterations,
        element_shape=tf.TensorShape([const_batch_size]))
  else:
    masks_ta = ()

  def pred(time, *unused_args):
    return time < iterations

  def body(time, time_since_reset, state, output_tas, masks_ta):
    """Internal while_loop body.

    Args:
      time: time
      time_since_reset: time since last prev_time_steps.is_first() == true.
        (only accurate / valid when mask_fn is not None).
      state: rnn state @ time
      output_tas: output tensorarrays
      masks_ta: optional mask tensorarray

    Returns:
      - time + 1
      - time_since_reset (next value)
      - state: rnn state @ time + 1
      - output_tas: output tensorarrays with values written @ time
      - masks_ta: optional mask tensorarray with mask written @ time
    """
    input_ = nest.map_structure(lambda ta: ta.read(time), inputs_tas)
    is_reset = reset_mask_ta.read(time)
    state = nest.map_structure(
        lambda s_zero, s: _maybe_reset_state(is_reset, s_zero, s),
        zero_state,
        state)

    outputs, next_state = cell(input_, state)

    output_tas = nest.map_structure(
        lambda ta, x: ta.write(time, x), output_tas, outputs)

    if mask_fn:
      time_since_reset = tf.where(
          is_reset,
          tf.zeros_like(time_since_reset),
          time_since_reset + 1,
          name="time_since_reset")
      masks_ta = masks_ta.write(time, mask_fn(time, time_since_reset))

    return (time + 1, time_since_reset, next_state, output_tas, masks_ta)

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with tf.variable_scope(tf.get_variable_scope()) as varscope:
    if (not tf.contrib.eager.executing_eagerly()
        and varscope.caching_device is None):
      varscope.set_caching_device(lambda op: op.device)

    _, _, final_state, output_tas, masks_ta = (
        tf.while_loop(
            pred,
            body,
            (tf.constant(0, name="time"),
             tf.zeros((batch_size,), dtype=tf.int32, name="time_since_reset"),
             initial_state,
             output_tas,
             masks_ta),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            maximum_iterations=iterations))

  outputs = nest.map_structure(lambda ta: ta.stack(), output_tas)

  if mask_fn:
    mask = masks_ta.stack()
  else:
    mask = None

  if isinstance(iterations, int):
    # TensorArray.stack() doesn't set a static value for dimension 0,
    # even if the size is known. Set the shapes here.
    iterations_shape = tf.TensorShape([iterations])
    for tensor in nest.flatten(outputs) + ([mask] if mask_fn else []):
      tensor.set_shape(iterations_shape.concatenate(tensor.shape[1:]))

  # Convert everything back to batch major
  outputs = nest.map_structure(tf.contrib.rnn.transpose_batch_time,
                               outputs)
  if mask is not None:
    mask = tf.transpose(mask)

  return (outputs, final_state, mask)
