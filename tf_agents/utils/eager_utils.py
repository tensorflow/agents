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

"""Common utilities for TF-Agents.

Example of usage:

  ```python
  from tf_agents.utils import eager_utils

  @eager_utils.run_in_graph_and_eager_modes
  def loss_fn(x, y):
    v = tf.get_variable('v', initializer=tf.ones_initializer(), shape=())
    return v + x - y

  with tfe.graph_mode():
    # loss and train_step are Tensors/Ops in the graph
    loss_op = loss_fn(inputs, labels)
    train_step_op = eager_utils.create_train_step(loss_op, optimizer)
    # Compute the loss and apply gradients to the variables using the optimizer.
    with tf.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      for _ in range(num_train_steps):
        loss_value = sess.run(train_step_op)

  with tfe.eager_mode():
    # loss and train_step are lambda functions that can be called.
    loss = loss_fn(inputs, labels)
    train_step = eager_utils.create_train_step(loss, optimizer)
    # Compute the loss and apply gradients to the variables using the optimizer.
    for _ in range(num_train_steps):
      loss_value = train_step()
  ```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import inspect
from absl import logging

import numpy as np
import six
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import common

from tensorflow.python.util import tf_decorator  # pylint:disable=g-direct-tensorflow-import  # TF internal

_USE_GLOBAL_STEP = 0


def has_self_cls_arg(func_or_method):
  """Checks if it is method which takes self/cls as the first argument."""
  if isinstance(func_or_method, staticmethod):
    return False
  if inspect.ismethod(func_or_method):
    return True
  if isinstance(func_or_method, classmethod):
    return True
  if six.PY2:
    arg_names = inspect.getargspec(func_or_method).args
  else:
    arg_names = list(inspect.signature(func_or_method).parameters)
  if arg_names and arg_names[0] in ('self', 'cls'):
    return True
  return False


def is_unbound(method):
  """Checks if it is an unbounded method."""
  return not (hasattr(method, '__self__') and method.__self__)


class Future(object):
  """Converts a function or class method call into a future callable."""

  def __init__(self, func_or_method, *args, **kwargs):
    self._func_or_method = func_or_method
    self._args = args
    self._kwargs = copy.copy(kwargs)
    getargspec = inspect.getargspec if six.PY2 else inspect.getfullargspec
    arg_names = getargspec(func_or_method).args
    self._arg_names = arg_names
    self._self = None
    if has_self_cls_arg(func_or_method):
      # Skip the first arg_name self/cls
      self._arg_names = arg_names[1:]
      if is_unbound(func_or_method):
        # For unbound methods we require self/cls as args[0].
        if not args:
          raise ValueError(
              'func_or_method is unbond, but not class/instance provided.')
        else:
          self._self = args[0]
          self._args = args[1:]

  def __call__(self, *args, **kwargs):
    """If *args/**kwargs are given they would replace those given at init.

    Args:
      *args: List of extra arguments.
      **kwargs: Dict of extra keyword arguments.
    Returns:
      The result of func_or_method(*args, **kwargs).
    """
    # By default use the init args.
    call_args = args or self._args
    call_kwargs = copy.copy(self._kwargs)
    for arg_name in self._arg_names[:len(args)]:
      # Remove any original kwargs replaced by new positional args.
      call_kwargs.pop(arg_name, None)
    call_kwargs.update(kwargs)
    if self._self:
      return self._func_or_method(self._self, *call_args, **call_kwargs)
    return self._func_or_method(*call_args, **call_kwargs)


def future_in_eager_mode(func_or_method):
  """Decorator that allow a function/method to run in graph and in eager modes.

  When applied in graph mode it calls the function and return its outputs.
  When applied in eager mode it returns a lambda function that when called
  returns the outputs.

  ```python
  @eager_utils.future_in_eager_mode
  def loss_fn(x):
    v = tf.get_variable('v', initializer=tf.ones_initializer(), shape=())
    return v + x

  with context.graph_mode():
    loss_op = loss_fn(inputs)
    loss_value = sess.run(loss_op)

  with context.eager_mode():
    loss = loss_fn(inputs)
    # Now loss is a Future callable.
    loss_value = loss()

  Args:
    func_or_method: A function or method to decorate.

  Returns:
    Either the output ops of the function/method or a Future (lambda function).
  """
  if not callable(func_or_method):
    raise TypeError('func_or_method must be callable.')

  def decorator(*args, **kwargs):
    if tf.executing_eagerly():
      return Future(func_or_method, *args, **kwargs)
    else:
      return func_or_method(*args, **kwargs)

  return tf_decorator.make_decorator(func_or_method, decorator)


def add_variables_summaries(grads_and_vars, step):
  """Add summaries for variables.

  Args:
    grads_and_vars: A list of (gradient, variable) pairs.
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_vars'):
    for _, var in grads_and_vars:
      if isinstance(var, tf.IndexedSlices):
        var_values = var.values
      else:
        var_values = var
      var_name = var.name.replace(':', '_')
      tf.compat.v2.summary.histogram(
          name=var_name + '_value', data=var_values, step=step)
      tf.compat.v2.summary.scalar(
          name=var_name + '_value_norm',
          data=tf.linalg.global_norm([var_values]),
          step=step)


def add_gradients_summaries(grads_and_vars, step):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    step: Variable to use for summaries.
  """
  with tf.name_scope('summarize_grads'):
    for grad, var in grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
        var_name = var.name.replace(':', '_')
        tf.compat.v2.summary.histogram(
            name=var_name + '_gradient', data=grad_values, step=step)
        tf.compat.v2.summary.scalar(
            name=var_name + '_gradient_norm',
            data=tf.linalg.global_norm([grad_values]),
            step=step)
      else:
        logging.info('Var %s has no gradient', var.name)


def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, max_norm)
        grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = tf.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars


def clip_gradient_norms_fn(max_norm):
  """Returns a `transform_grads_fn` function for gradient clipping."""
  def clip_norms(gradients_to_variables):
    return clip_gradient_norms(gradients_to_variables, max_norm)
  return clip_norms


def create_train_step(loss,
                      optimizer,
                      global_step=_USE_GLOBAL_STEP,
                      total_loss_fn=None,
                      update_ops=None,
                      variables_to_train=None,
                      transform_grads_fn=None,
                      summarize_gradients=False,
                      gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                      aggregation_method=None,
                      check_numerics=True):
  """Creates a train_step that evaluates the gradients and returns the loss.

  Args:
    loss: A (possibly nested tuple of) `Tensor` or function representing
      the loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.train.get_or_create_global_step() is used.
    total_loss_fn: Function to call on loss value to access the final
     item to minimize.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    transform_grads_fn: A function which takes a single argument, a list of
      gradient to variable pairs (tuples), performs any requested gradient
      updates, such as gradient clipping or multipliers, and returns the updated
      list.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    check_numerics: Whether or not we apply check_numerics.

  Returns:
    In graph mode: A (possibly nested tuple of) `Tensor` that when evaluated,
      calculates the current loss, computes the gradients, applies the
      optimizer, and returns the current loss.
    In eager mode: A lambda function that when is called, calculates the loss,
      then computes and applies the gradients and returns the original
      loss values.
  Raises:
    ValueError: if loss is not callable.
    RuntimeError: if resource variables are not enabled.
  """
  if total_loss_fn is None:
    total_loss_fn = lambda x: x
  if not callable(total_loss_fn):
    raise ValueError('`total_loss_fn` should be a function.')
  if not common.resource_variables_enabled():
    raise RuntimeError(common.MISSING_RESOURCE_VARIABLES_ERROR)
  if not tf.executing_eagerly():
    if callable(loss):
      loss = loss()
    if callable(variables_to_train):
      variables_to_train = variables_to_train()
    # Calculate loss first, then calculate train op, then return the original
    # loss conditioned on executing the train op.
    with tf.control_dependencies(tf.nest.flatten(loss)):
      loss = tf.nest.map_structure(
          lambda t: tf.identity(t, 'loss_pre_train'), loss)
    train_op = create_train_op(
        total_loss_fn(loss),
        optimizer,
        global_step=global_step,
        update_ops=update_ops,
        variables_to_train=variables_to_train,
        transform_grads_fn=transform_grads_fn,
        summarize_gradients=summarize_gradients,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        check_numerics=check_numerics)

    with tf.control_dependencies([train_op]):
      return tf.nest.map_structure(
          lambda t: tf.identity(t, 'loss_post_train'), loss)

  if global_step is _USE_GLOBAL_STEP:
    global_step = tf.compat.v1.train.get_or_create_global_step()

  if not callable(loss):
    raise ValueError('`loss` should be a function in eager mode.')

  if not isinstance(loss, Future):
    logging.warning('loss should be an instance of eager_utils.Future')

  with tf.GradientTape() as tape:
    loss_value = loss()
    total_loss_value = total_loss_fn(loss_value)
  if variables_to_train is None:
    variables_to_train = tape.watched_variables()
  elif callable(variables_to_train):
    variables_to_train = variables_to_train()
  variables_to_train = tf.nest.flatten(variables_to_train)
  grads = tape.gradient(total_loss_value, variables_to_train)
  grads_and_vars = zip(grads, variables_to_train)

  if transform_grads_fn:
    grads_and_vars = transform_grads_fn(grads_and_vars)

  if summarize_gradients:
    with tf.name_scope('summarize_grads'):
      add_gradients_summaries(grads_and_vars, global_step)

  if check_numerics:
    with tf.name_scope('train_op'):
      tf.debugging.check_numerics(total_loss_value, 'Loss is inf or nan')

  optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  return loss_value


def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    transform_grads_fn=None,
                    summarize_gradients=False,
                    gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                    aggregation_method=None,
                    check_numerics=True):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.train.get_or_create_global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    transform_grads_fn: A function which takes a single argument, a list of
      gradient to variable pairs (tuples), performs any requested gradient
      updates, such as gradient clipping or multipliers, and returns the updated
      list.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    check_numerics: Whether or not we apply check_numerics.

  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  if global_step is _USE_GLOBAL_STEP:
    global_step = tf.compat.v1.train.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(
      tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    tf.compat.v1.logging.warning(
        'update_ops in create_train_op does not contain all the '
        'update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with tf.control_dependencies(update_ops):
      barrier = tf.no_op(name='update_barrier')
    with tf.control_dependencies([barrier]):
      total_loss = tf.identity(total_loss)

  if variables_to_train is None:
    # Default to tf.trainable_variables()
    variables_to_train = tf.compat.v1.trainable_variables()
  else:
    # Make sure that variables_to_train are in tf.trainable_variables()
    trainable_variables = tf.compat.v1.trainable_variables()
    for v in variables_to_train:
      assert v.trainable or v in trainable_variables

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method)

  if transform_grads_fn:
    grads = transform_grads_fn(grads)

  # Summarize gradients.
  if summarize_gradients:
    with tf.name_scope('summarize_grads'):
      add_gradients_summaries(grads, global_step)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with tf.name_scope('train_op'):
    # Make sure total_loss is valid.
    if check_numerics:
      total_loss = tf.debugging.check_numerics(total_loss,
                                               'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    with tf.control_dependencies([grad_updates]):
      train_op = tf.identity(total_loss, name='train_op')

  return train_op


def np_function(func=None, output_dtypes=None):
  """Decorator that allow a numpy function to be used in Eager and Graph modes.

  Similar to `tf.py_func` and `tf.py_function` but it doesn't require defining
  the inputs or the dtypes of the outputs a priori.

  In Eager mode it would convert the tf.Tensors to np.arrays before passing to
  `func` and then convert back the outputs from np.arrays to tf.Tensors.

  In Graph mode it would create different tf.py_function for each combination
  of dtype of the inputs and cache them for reuse.

  NOTE: In Graph mode: if `output_dtypes` is not provided then `func` would
  be called with `np.ones()` to infer the output dtypes, and therefore `func`
  should be stateless.

  ```python
  Instead of doing:

  def sum(x):
    return np.sum(x)
  inputs = tf.constant([3, 4])
  outputs = tf.py_function(sum, inputs, Tout=[tf.int64])

  inputs = tf.constant([3., 4.])
  outputs = tf.py_function(sum, inputs, Tout=[tf.float32])

  Do:
  @eager_utils.np_function
  def sum(x):
    return np.sum(x)

  inputs = tf.constant([3, 4])
  outputs = sum(inputs)  # Infers that Tout is tf.int64

  inputs = tf.constant([3., 4.])
  outputs = sum(inputs)  # Infers that Tout is tf.float32

  # Output dtype is always float32 for valid input dtypes.
  @eager_utils.np_function(output_dtypes=np.float32)
  def mean(x):
    return np.mean(x)

  # Output dtype depends on the input dtype.
  @eager_utils.np_function(output_dtypes=lambda x: (x, x))
  def repeat(x):
    return x, x

  with context.graph_mode():
    outputs = sum(tf.constant([3, 4]))
    outputs2 = sum(tf.constant([3., 4.]))
    sess.run(outputs) # np.array(7)
    sess.run(outputs2) # np.array(7.)

  with context.eager_mode():
    inputs = tf.constant([3, 4])
    outputs = sum(tf.constant([3, 4])) # tf.Tensor([7])
    outputs = sum(tf.constant([3., 4.])) # tf.Tensor([7.])

  ```
  Args:
    func: A numpy function, that takes numpy arrays as inputs and return numpy
      arrays as outputs.
    output_dtypes: Optional list of dtypes or a function that maps input dtypes
      to output dtypes. Examples: output_dtypes=[tf.float32],
      output_dtypes=lambda x: x (outputs have the same dtype as inputs).
      If it is not provided in Graph mode the `func` would be called to infer
      the output dtypes.
  Returns:
    A wrapped function that can be used with TF code.
  """
  def decorated(func):
    """Decorated func."""
    dtype_map = {}
    def wrapper(*args, **kwargs):
      """Wrapper to add nested input and outputs support."""
      func_with_kwargs = functools.partial(func, **kwargs)
      def func_flat_outputs(*args):
        return tf.nest.flatten(func_with_kwargs(*args))

      def compute_output_dtypes(*args):
        """Calls the func to compute output dtypes."""
        result = func(*args, **kwargs)
        return tf.nest.map_structure(lambda x: x.dtype, result)

      if tf.executing_eagerly():
        result = func_with_kwargs(
            *tf.nest.map_structure(lambda x: x.numpy(), args))
        convert = lambda x: x if x is None else tf.convert_to_tensor(value=x)
        return tf.nest.map_structure(convert, result)
      else:
        input_dtypes = tuple([x.dtype for x in tf.nest.flatten(args)])
        if input_dtypes not in dtype_map:
          if output_dtypes is None:
            dummy_args = tf.nest.map_structure(
                lambda x: np.ones(x.shape, x.dtype.as_numpy_dtype), args)
            dtype_map[input_dtypes] = compute_output_dtypes(*dummy_args)
          elif isinstance(output_dtypes, (list, tuple)):
            # output_dtypes define the output dtypes.
            dtype_map[input_dtypes] = output_dtypes
          else:
            try:
              # See if output_dtypes define the output dtype directly.
              tf.as_dtype(output_dtypes)
              dtype_map[input_dtypes] = output_dtypes
            except TypeError:
              if callable(output_dtypes):
                # output_dtypes is mapping from input_dtypes to output_dtypes.
                dtype_map[input_dtypes] = output_dtypes(*input_dtypes)
              else:
                raise ValueError(
                    'output_dtypes not a list of dtypes or a callable.')

      flat_output_dtypes = tf.nest.flatten(dtype_map[input_dtypes])
      flat_outputs = tf.py_function(func_flat_outputs,
                                    inp=args,
                                    Tout=flat_output_dtypes)
      return tf.nest.pack_sequence_as(dtype_map[input_dtypes], flat_outputs)

    return tf_decorator.make_decorator(func, wrapper)
  # This code path is for the `foo = np_function(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the decorator
  # @np_function(...)
  # def foo(...):
  return decorated


def dataset_iterator(dataset):
  """Constructs a `Dataset` iterator.

  The method used to construct the iterator is conditioned on whether Graph mode
  is enabled. `dataset_iterator` and `get_next` are useful when we need to
  construct an iterator and iterate through it inside a `tensorflow.function`.

  Args:
    dataset: a `tf.data.Dataset`.
  Returns:
    A `tf.data.Iterator` if Graph mode is enabled; a tf.data.EagerIterator if
    in eager mode.
  """
  if tf.executing_eagerly():
    return iter(dataset)
  try:
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  except ValueError:
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  return iterator


def get_next(iterator):
  """Returns the next element in a `Dataset` iterator.

  The syntax used to retrieve the next item is conditioned on whether Graph mode
  is enabled. `dataset_iterator` and `get_next` are useful when we need to
  construct an iterator and iterate through it inside a `tensorflow.function`.

  Args:
    iterator: a `tf.data.Iterator` if in Graph mode; a `tf.data.EagerIterator`
      if in eager mode.
  Returns:
    A `tf.data.Iterator` if Graph mode is enabled; a Python iterator if in eager
    mode.
  """
  if tf.executing_eagerly():
    return next(iterator)
  return iterator.get_next()
