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

"""Common utilities for TF-Agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import logging

import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils

from tensorflow.core.protobuf import struct_pb2  # pylint:disable=g-direct-tensorflow-import  # TF internal
from tensorflow.python.saved_model import nested_structure_coder  # pylint:disable=g-direct-tensorflow-import  # TF internal

MISSING_RESOURCE_VARIABLES_ERROR = """
Resource variables are not enabled.  Please enable them by adding the following
code to your main() method:
  tf.compat.v1.enable_resource_variables()
For unit tests, subclass `tf_agents.utils.test_utils.TestCase`.
"""


def resource_variables_enabled():
  return tf.compat.v1.resource_variables_enabled()


def function(*args, **kwargs):
  """Wrapper for tf.function with TF Agents-specific customizations.

  Example:

  ```python
  @common.function()
  def my_eager_code(x, y):
    ...
  ```

  Args:
    *args: Args for tf.function.
    **kwargs: Keyword args for tf.function.

  Returns:
    A tf.function wrapper.
  """
  autograph = kwargs.pop('autograph', False)
  experimental_relax_shapes = kwargs.pop('experimental_relax_shapes', True)
  return tf.function(  # allow-tf-function
      *args,
      autograph=autograph,
      experimental_relax_shapes=experimental_relax_shapes,
      **kwargs)


def has_eager_been_enabled():
  """Returns true iff in TF2 or in TF1 with eager execution enabled."""
  with tf.init_scope():
    return tf.executing_eagerly()


def function_in_tf1(*args, **kwargs):
  """Wrapper that returns common.function if using TF1.

  This allows for code that assumes autodeps is available to be written once,
  in the same way, for both TF1 and TF2.

  Usage:

  ```python
  train = function_in_tf1()(agent.train)
  loss = train(experience)
  ```

  Args:
    *args: Arguments for common.function.
    **kwargs: Keyword arguments for common.function.

  Returns:
    A callable that wraps a function.
  """

  def maybe_wrap(fn):
    """Helper function."""
    # We're in TF1 mode and want to wrap in common.function to get autodeps.
    wrapped = [None]
    @functools.wraps(fn)
    def with_check_resource_vars(*fn_args, **fn_kwargs):
      """Helper function for calling common.function."""
      if has_eager_been_enabled():
        # We're either in eager mode or in tf.function mode (no in-between); so
        # autodep-like behavior is already expected of fn.
        return fn(*fn_args, **fn_kwargs)
      if not resource_variables_enabled():
        raise RuntimeError(MISSING_RESOURCE_VARIABLES_ERROR)
      if wrapped[0] is None:
        wrapped[0] = function(*args, **kwargs)(fn)
      return wrapped[0](*fn_args, **fn_kwargs)  # pylint: disable=not-callable
    return with_check_resource_vars
  return maybe_wrap


def create_variable(name,
                    initial_value=0,
                    shape=(),
                    dtype=tf.int64,
                    use_local_variable=False,
                    trainable=False,
                    initializer=None,
                    unique_name=True):
  """Create a variable."""
  if has_eager_been_enabled():
    if initializer is None:
      if shape:
        initial_value = tf.constant(initial_value, shape=shape, dtype=dtype)
      else:
        initial_value = tf.convert_to_tensor(initial_value, dtype=dtype)
    else:
      if callable(initializer):
        initial_value = lambda: initializer(shape)
      else:
        initial_value = initializer
    return tf.compat.v2.Variable(
        initial_value, trainable=trainable, dtype=dtype, name=name)
  collections = [tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
  if use_local_variable:
    collections = [tf.compat.v1.GraphKeys.LOCAL_VARIABLES]
  if initializer is None:
    initializer = tf.compat.v1.initializers.constant(initial_value, dtype=dtype)
    if shape is None:
      shape = tf.convert_to_tensor(initial_value).shape
  if unique_name:
    name = tf.compat.v1.get_default_graph().unique_name(name)
  return tf.compat.v1.get_variable(
      name=name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      collections=collections,
      use_resource=True,
      trainable=trainable)


def soft_variables_update(source_variables,
                          target_variables,
                          tau=1.0,
                          sort_variables_by_name=False):
  """Performs a soft/hard update of variables from the source to the target.

  For each variable v_t in target variables and its corresponding variable v_s
  in source variables, a soft update is:
  v_t = (1 - tau) * v_t + tau * v_s

  When tau is 1.0 (the default), then it does a hard update:
  v_t = v_s

  Args:
    source_variables: list of source variables.
    target_variables: list of target variables.
    tau: A float scalar in [0, 1]. When tau is 1.0 (the default), we do a hard
      update.
    sort_variables_by_name: A bool, when True would sort the variables by name
      before doing the update.

  Returns:
    An operation that updates target variables from source variables.
  Raises:
    ValueError: if tau is not in [0, 1].
  """
  if tau < 0 or tau > 1:
    raise ValueError('Input `tau` should be in [0, 1].')
  updates = []

  op_name = 'soft_variables_update'
  if tau == 0.0 or not source_variables or not target_variables:
    return tf.no_op(name=op_name)
  if sort_variables_by_name:
    source_variables = sorted(source_variables, key=lambda x: x.name)
    target_variables = sorted(target_variables, key=lambda x: x.name)
  for (v_s, v_t) in zip(source_variables, target_variables):
    v_t.shape.assert_is_compatible_with(v_s.shape)
    if tau == 1.0:
      update = v_t.assign(v_s)
    else:
      update = v_t.assign((1 - tau) * v_t + tau * v_s)
    updates.append(update)
  return tf.group(*updates, name=op_name)


def join_scope(parent_scope, child_scope):
  """Joins a parent and child scope using `/`, checking for empty/none.

  Args:
    parent_scope: (string) parent/prefix scope.
    child_scope: (string) child/suffix scope.

  Returns:
    joined scope: (string) parent and child scopes joined by /.
  """
  if not parent_scope:
    return child_scope
  if not child_scope:
    return parent_scope
  return '/'.join([parent_scope, child_scope])


def index_with_actions(q_values, actions, multi_dim_actions=False):
  """Index into q_values using actions.

  Note: this supports multiple outer dimensions (e.g. time, batch etc).

  Args:
    q_values: A float tensor of shape [outer_dim1, ... outer_dimK, action_dim1,
      ..., action_dimJ].
    actions: An int tensor of shape [outer_dim1, ... outer_dimK]    if
      multi_dim_actions=False [outer_dim1, ... outer_dimK, J] if
      multi_dim_actions=True I.e. in the multidimensional case,
      actions[outer_dim1, ... outer_dimK] is a vector [actions_1, ...,
      actions_J] where each element actions_j is an action in the range [0,
      num_actions_j). While in the single dimensional case, actions[outer_dim1,
      ... outer_dimK] is a scalar.
    multi_dim_actions: whether the actions are multidimensional.
    # TODO(kbanoop): Add an optional action_spec for validation.

  Returns:
    A [outer_dim1, ... outer_dimK] tensor of q_values for the given actions.

  Raises:
    ValueError: If actions have unknown rank.
  """
  if actions.shape.ndims is None:
    raise ValueError('actions should have known rank.')
  batch_dims = actions.shape.ndims
  if multi_dim_actions:
    # In the multidimensional case, the last dimension of actions indexes the
    # vector of actions for each batch, so exclude it from the batch dimensions.
    batch_dims -= 1

  outer_shape = tf.shape(input=actions)
  batch_indices = tf.meshgrid(
      *[tf.range(outer_shape[i]) for i in range(batch_dims)], indexing='ij')
  batch_indices = [
      tf.expand_dims(batch_index, -1) for batch_index in batch_indices
  ]
  if not multi_dim_actions:
    actions = tf.expand_dims(actions, -1)
  action_indices = tf.concat(batch_indices + [actions], -1)
  return tf.gather_nd(q_values, action_indices)


def periodically(body, period, name='periodically'):
  """Periodically performs the tensorflow op in `body`.

  The body tensorflow op will be executed every `period` times the periodically
  op is executed. More specifically, with `n` the number of times the op has
  been executed, the body will be executed when `n` is a non zero positive
  multiple of `period` (i.e. there exist an integer `k > 0` such that
  `k * period == n`).

  If `period` is `None`, it will not perform any op and will return a
  `tf.no_op()`.

  If `period` is 1, it will just execute the body, and not create any counters
  or conditionals.

  Args:
    body: callable that returns the tensorflow op to be performed every time an
      internal counter is divisible by the period. The op must have no output
      (for example, a tf.group()).
    period: inverse frequency with which to perform the op.
    name: name of the variable_scope.

  Raises:
    TypeError: if body is not a callable.

  Returns:
    An op that periodically performs the specified op.
  """
  if tf.executing_eagerly():
    if isinstance(period, tf.Variable):
      return Periodically(body, period, name)
    return EagerPeriodically(body, period)
  else:
    return Periodically(body, period, name)()


class Periodically(tf.Module):
  """Periodically performs the ops defined in `body`."""

  def __init__(self, body, period, name='periodically'):
    """Periodically performs the ops defined in `body`.

    The body tensorflow op will be executed every `period` times the
    periodically op is executed. More specifically, with `n` the number of times
    the op has been executed, the body will be executed when `n` is a non zero
    positive multiple of `period` (i.e. there exist an integer `k > 0` such that
    `k * period == n`).

    If `period` is `None`, it will not perform any op and will return a
    `tf.no_op()`.

    If `period` is 1, it will just execute the body, and not create any counters
    or conditionals.

    Args:
      body: callable that returns the tensorflow op to be performed every time
        an internal counter is divisible by the period. The op must have no
        output (for example, a tf.group()).
      period: inverse frequency with which to perform the op. It can be a Tensor
        or a Variable.
      name: name of the object.

    Raises:
      TypeError: if body is not a callable.

    Returns:
      An op that periodically performs the specified op.
    """
    super(Periodically, self).__init__(name=name)
    if not callable(body):
      raise TypeError('body must be callable.')
    self._body = body
    self._period = period
    self._counter = create_variable(self.name + '/counter', 0)

  def __call__(self):

    def call(strategy=None):
      del strategy  # unused
      if self._period is None:
        return tf.no_op()
      if self._period == 1:
        return self._body()
      period = tf.cast(self._period, self._counter.dtype)
      remainder = tf.math.mod(self._counter.assign_add(1), period)
      return tf.cond(
          pred=tf.equal(remainder, 0), true_fn=self._body, false_fn=tf.no_op)

    # TODO(b/129083817) add an explicit unit test to ensure correct behavior
    ctx = tf.distribute.get_replica_context()
    if ctx:
      return tf.distribute.get_replica_context().merge_call(call)
    else:
      return call()


class EagerPeriodically(object):
  """EagerPeriodically performs the ops defined in `body`.

  Only works in Eager mode.
  """

  def __init__(self, body, period):
    """EagerPeriodically performs the ops defined in `body`.

    Args:
      body: callable that returns the tensorflow op to be performed every time
        an internal counter is divisible by the period. The op must have no
        output (for example, a tf.group()).
      period: inverse frequency with which to perform the op. Must be a simple
        python int/long.

    Raises:
      TypeError: if body is not a callable.

    Returns:
      An op that periodically performs the specified op.
    """
    if not callable(body):
      raise TypeError('body must be callable.')
    self._body = body
    self._period = period
    self._counter = 0

  def __call__(self):
    if self._period is None:
      return tf.no_op()
    if self._period == 1:
      return self._body()
    self._counter += 1
    if self._counter % self._period == 0:
      self._body()


def clip_to_spec(value, spec):
  """Clips value to a given bounded tensor spec.

  Args:
    value: (tensor) value to be clipped.
    spec: (BoundedTensorSpec) spec containing min. and max. values for clipping.

  Returns:
    clipped_value: (tensor) `value` clipped to be compatible with `spec`.
  """
  return tf.clip_by_value(value, spec.minimum, spec.maximum)


def spec_means_and_magnitudes(action_spec):
  """Get the center and magnitude of the ranges in action spec."""
  action_means = tf.nest.map_structure(
      lambda spec: (spec.maximum + spec.minimum) / 2.0, action_spec)
  action_magnitudes = tf.nest.map_structure(
      lambda spec: (spec.maximum - spec.minimum) / 2.0, action_spec)
  return tf.cast(
      action_means, dtype=tf.float32), tf.cast(
          action_magnitudes, dtype=tf.float32)


def scale_to_spec(tensor, spec):
  """Shapes and scales a batch into the given spec bounds.

  Args:
    tensor: A [batch x n] tensor with values in the range of [-1, 1].
    spec: (BoundedTensorSpec) to use for scaling the action.

  Returns:
    A batch scaled the given spec bounds.
  """
  tensor = tf.reshape(tensor, [-1] + spec.shape.as_list())

  # Scale the tensor.
  means, magnitudes = spec_means_and_magnitudes(spec)
  tensor = means + magnitudes * tensor

  # Set type.
  return tf.cast(tensor, spec.dtype)


def ornstein_uhlenbeck_process(initial_value,
                               damping=0.15,
                               stddev=0.2,
                               seed=None,
                               scope='ornstein_uhlenbeck_noise'):
  """An op for generating noise from a zero-mean Ornstein-Uhlenbeck process.

  The Ornstein-Uhlenbeck process is a process that generates temporally
  correlated noise via a random walk with damping. This process describes
  the velocity of a particle undergoing brownian motion in the presence of
  friction. This can be useful for exploration in continuous action environments
  with momentum.

  The temporal update equation is:
  `x_next = (1 - damping) * x + N(0, std_dev)`

  Args:
    initial_value: Initial value of the process.
    damping: The rate at which the noise trajectory is damped towards the mean.
      We must have 0 <= damping <= 1, where a value of 0 gives an undamped
      random walk and a value of 1 gives uncorrelated Gaussian noise. Hence in
      most applications a small non-zero value is appropriate.
    stddev: Standard deviation of the Gaussian component.
    seed: Seed for random number generation.
    scope: Scope of the variables.

  Returns:
    An op that generates noise.
  """
  if tf.executing_eagerly():
    return OUProcess(initial_value, damping, stddev, seed, scope)
  else:
    return OUProcess(initial_value, damping, stddev, seed, scope)()


class OUProcess(tf.Module):
  """A zero-mean Ornstein-Uhlenbeck process."""

  def __init__(self,
               initial_value,
               damping=0.15,
               stddev=0.2,
               seed=None,
               scope='ornstein_uhlenbeck_noise'):
    """A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.

    The Ornstein-Uhlenbeck process is a process that generates temporally
    correlated noise via a random walk with damping. This process describes
    the velocity of a particle undergoing brownian motion in the presence of
    friction. This can be useful for exploration in continuous action
    environments with momentum.

    The temporal update equation is:
    `x_next = (1 - damping) * x + N(0, std_dev)`

    Args:
      initial_value: Initial value of the process.
      damping: The rate at which the noise trajectory is damped towards the
        mean. We must have 0 <= damping <= 1, where a value of 0 gives an
        undamped random walk and a value of 1 gives uncorrelated Gaussian noise.
        Hence in most applications a small non-zero value is appropriate.
      stddev: Standard deviation of the Gaussian component.
      seed: Seed for random number generation.
      scope: Scope of the variables.
    """
    super(OUProcess, self).__init__()
    self._damping = damping
    self._stddev = stddev
    self._seed = seed
    with tf.name_scope(scope):
      self._x = tf.compat.v2.Variable(
          initial_value=initial_value, trainable=False)

  def __call__(self):
    noise = tf.random.normal(
        shape=self._x.shape,
        stddev=self._stddev,
        dtype=self._x.dtype,
        seed=self._seed)
    return self._x.assign((1. - self._damping) * self._x + noise)


def log_probability(distributions, actions, action_spec):
  """Computes log probability of actions given distribution.

  Args:
    distributions: A possibly batched tuple of distributions.
    actions: A possibly batched action tuple.
    action_spec: A nested tuple representing the action spec.

  Returns:
    A Tensor representing the log probability of each action in the batch.
  """
  outer_rank = nest_utils.get_outer_rank(actions, action_spec)

  def _compute_log_prob(single_distribution, single_action):
    # sum log-probs over everything but the batch
    rank = single_action.shape.ndims
    reduce_dims = list(range(outer_rank, rank))
    return tf.reduce_sum(
        input_tensor=single_distribution.log_prob(single_action),
        axis=reduce_dims)

  tf.nest.assert_same_structure(distributions, actions)
  log_probs = [
      _compute_log_prob(dist, action)
      for (dist, action
          ) in zip(tf.nest.flatten(distributions), tf.nest.flatten(actions))
  ]

  # sum log-probs over action tuple
  total_log_probs = tf.add_n(log_probs)

  return total_log_probs


# TODO(ofirnachum): Move to distribution utils.
def entropy(distributions, action_spec):
  """Computes total entropy of distribution.

  Args:
    distributions: A possibly batched tuple of distributions.
    action_spec: A nested tuple representing the action spec.

  Returns:
    A Tensor representing the entropy of each distribution in the batch.
    Assumes actions are independent, so that marginal entropies of each action
    may be summed.
  """
  nested_modes = tf.nest.map_structure(lambda d: d.mode(), distributions)
  outer_rank = nest_utils.get_outer_rank(nested_modes, action_spec)

  def _compute_entropy(single_distribution):
    entropies = single_distribution.entropy()
    # Sum entropies over everything but the batch.
    rank = entropies.shape.ndims
    reduce_dims = list(range(outer_rank, rank))
    return tf.reduce_sum(input_tensor=entropies, axis=reduce_dims)

  entropies = [
      _compute_entropy(dist) for dist in tf.nest.flatten(distributions)
  ]

  # Sum entropies over action tuple.
  total_entropies = tf.add_n(entropies)

  return total_entropies


def discounted_future_sum(values, gamma, num_steps):
  """Discounted future sum of batch-major values.

  Args:
    values: A Tensor of shape [batch_size, total_steps] and dtype float32.
    gamma: A float discount value.
    num_steps: A positive integer number of future steps to sum.

  Returns:
    A Tensor of shape [batch_size, total_steps], where each entry `(i, j)` is
      the result of summing the entries of values starting from
      `gamma^0 * values[i, j]` to
      `gamma^(num_steps - 1) * values[i, j + num_steps - 1]`,
      with zeros padded to values.

      For example, values=[5, 6, 7], gamma=0.9, will result in sequence:
      ```python
      [(5 * 0.9^0 + 6 * 0.9^1 + 7 * 0.9^2), (6 * 0.9^0 + 7 * 0.9^1), 7 * 0.9^0]
      ```

  Raises:
    ValueError: If values is not of rank 2.
  """
  if values.get_shape().ndims != 2:
    raise ValueError('Input must be rank 2 tensor.  Got %d.' %
                     values.get_shape().ndims)

  (batch_size, total_steps) = values.get_shape().as_list()

  num_steps = tf.minimum(num_steps, total_steps)
  discount_filter = tf.reshape(gamma**tf.cast(tf.range(num_steps), tf.float32),
                               [-1, 1, 1])
  padded_values = tf.concat([values, tf.zeros([batch_size, num_steps - 1])], 1)

  convolved_values = tf.squeeze(
      tf.nn.conv1d(
          input=tf.expand_dims(padded_values, -1),
          filters=discount_filter,
          stride=1,
          padding='VALID'), -1)

  return convolved_values


def discounted_future_sum_masked(values, gamma, num_steps, episode_lengths):
  """Discounted future sum of batch-major values.

  Args:
    values: A Tensor of shape [batch_size, total_steps] and dtype float32.
    gamma: A float discount value.
    num_steps: A positive integer number of future steps to sum.
    episode_lengths: A vector shape [batch_size] with num_steps per episode.

  Returns:
    A Tensor of shape [batch_size, total_steps], where each entry is the
      discounted sum as in discounted_future_sum, except with values after
      the end of episode_lengths masked to 0.

  Raises:
    ValueError: If values is not of rank 2, or if total_steps is not defined.
  """
  if values.shape.ndims != 2:
    raise ValueError('Input must be a rank 2 tensor.  Got %d.' % values.shape)

  total_steps = tf.compat.dimension_value(values.shape[1])
  if total_steps is None:
    raise ValueError('total_steps dimension in input '
                     'values[batch_size, total_steps] must be fully defined.')

  episode_mask = tf.cast(
      tf.sequence_mask(episode_lengths, total_steps), tf.float32)
  values *= episode_mask
  return discounted_future_sum(values, gamma, num_steps)


def shift_values(values, gamma, num_steps, final_values=None):
  """Shifts batch-major values in time by some amount.

  Args:
    values: A Tensor of shape [batch_size, total_steps] and dtype float32.
    gamma: A float discount value.
    num_steps: A nonnegative integer amount to shift values by.
    final_values: A float32 Tensor of shape [batch_size] corresponding to the
      values at step num_steps + 1.  Defaults to None (all zeros).

  Returns:
    A Tensor of shape [batch_size, total_steps], where each entry (i, j) is
    gamma^num_steps * values[i, j + num_steps] if j + num_steps < total_steps;
    gamma^(total_steps - j) * final_values[i] otherwise.

  Raises:
    ValueError: If values is not of rank 2.
  """
  if values.get_shape().ndims != 2:
    raise ValueError('Input must be rank 2 tensor.  Got %d.' %
                     values.get_shape().ndims)

  (batch_size, total_steps) = values.get_shape().as_list()
  num_steps = tf.minimum(num_steps, total_steps)

  if final_values is None:
    final_values = tf.zeros([batch_size])

  padding_exponent = tf.expand_dims(
      tf.cast(tf.range(num_steps, 0, -1), tf.float32), 0)
  final_pad = tf.expand_dims(final_values, 1) * gamma**padding_exponent
  return tf.concat([
      gamma**tf.cast(num_steps, tf.float32) * values[:, num_steps:], final_pad
  ], 1)


def get_episode_mask(time_steps):
  """Create a mask that is 0.0 for all final steps, 1.0 elsewhere.

  Args:
    time_steps: A TimeStep namedtuple representing a batch of steps.

  Returns:
    A float32 Tensor with 0s where step_type == LAST and 1s otherwise.
  """
  episode_mask = tf.cast(
      tf.not_equal(time_steps.step_type, ts.StepType.LAST), tf.float32)
  return episode_mask


def get_contiguous_sub_episodes(next_time_steps_discount):
  """Computes mask on sub-episodes which includes only contiguous components.

  Args:
    next_time_steps_discount: Tensor of shape [batch_size, total_steps]
      corresponding to environment discounts on next time steps (i.e.
      next_time_steps.discount).

  Returns:
    A float Tensor of shape [batch_size, total_steps] specifying mask including
      only contiguous components. Each row will be of the form
      [1.0] * a + [0.0] * b, where a >= 1 and b >= 0, and in which the initial
      sequence of ones corresponds to a contiguous sub-episode.
  """
  episode_end = tf.equal(next_time_steps_discount,
                         tf.constant(0, dtype=next_time_steps_discount.dtype))
  mask = tf.math.cumprod(
      1.0 - tf.cast(episode_end, tf.float32), axis=1, exclusive=True)
  return mask


def convert_q_logits_to_values(logits, support):
  """Converts a set of Q-value logits into Q-values using the provided support.

  Args:
    logits: A Tensor representing the Q-value logits.
    support: The support of the underlying distribution.

  Returns:
    A Tensor containing the expected Q-values.
  """
  probabilities = tf.nn.softmax(logits)
  return tf.reduce_sum(input_tensor=support * probabilities, axis=-1)


def generate_tensor_summaries(tag, tensor, step):
  """Generates various summaries of `tensor` such as histogram, max, min, etc.

  Args:
    tag: A namescope tag for the summaries.
    tensor: The tensor to generate summaries of.
    step: Variable to use for summaries.
  """
  with tf.name_scope(tag):
    tf.compat.v2.summary.histogram(name='histogram', data=tensor, step=step)
    tf.compat.v2.summary.scalar(
        name='mean', data=tf.reduce_mean(input_tensor=tensor), step=step)
    tf.compat.v2.summary.scalar(
        name='mean_abs',
        data=tf.reduce_mean(input_tensor=tf.abs(tensor)),
        step=step)
    tf.compat.v2.summary.scalar(
        name='max', data=tf.reduce_max(input_tensor=tensor), step=step)
    tf.compat.v2.summary.scalar(
        name='min', data=tf.reduce_min(input_tensor=tensor), step=step)


# TODO(kbanoop): Support batch mode
def compute_returns(rewards, discounts):
  """Compute the return from each index in an episode.

  Args:
    rewards: Tensor of per-timestep reward in the episode.
    discounts: Tensor of per-timestep discount factor. Should be 0 for final
      step of each episode.

  Returns:
    Tensor of per-timestep cumulative returns.
  """
  rewards.shape.assert_is_compatible_with(discounts.shape)
  if (not rewards.shape.is_fully_defined() or
      not discounts.shape.is_fully_defined()):
    check_shape = tf.compat.v1.assert_equal(
        tf.shape(input=rewards), tf.shape(input=discounts))
  else:
    check_shape = tf.no_op()
  with tf.control_dependencies([check_shape]):
    # Reverse the rewards and discounting for accumulation.
    rewards, discounts = tf.reverse(rewards, [0]), tf.reverse(discounts, [0])

  def discounted_accumulate_rewards(next_step_return, reward_and_discount):
    reward, discount = reward_and_discount
    return next_step_return * discount + reward

  # Cumulatively sum discounted reward R_t.
  #   R_t = r_t + discount * (r_t+1 + discount * (r_t+2 * discount( ...
  # As discount is 0 for terminal states, ends of episode will not include
  #   reward from subsequent timesteps.
  returns = tf.scan(
      discounted_accumulate_rewards, [rewards, discounts],
      initializer=tf.constant(0, dtype=discounts.dtype))
  returns = tf.reverse(returns, [0])
  return returns


def initialize_uninitialized_variables(session, var_list=None):
  """Initialize any pending variables that are uninitialized."""
  if var_list is None:
    var_list = tf.compat.v1.global_variables() + tf.compat.v1.local_variables()
  is_initialized = session.run(
      [tf.compat.v1.is_variable_initialized(v) for v in var_list])
  uninitialized_vars = []
  for flag, v in zip(is_initialized, var_list):
    if not flag:
      uninitialized_vars.append(v)
  if uninitialized_vars:
    logging.info('uninitialized_vars:')
    for v in uninitialized_vars:
      logging.info(v)
    session.run(tf.compat.v1.variables_initializer(uninitialized_vars))


class Checkpointer(object):
  """Checkpoints training state, policy state, and replay_buffer state."""

  def __init__(self, ckpt_dir, max_to_keep=20, **kwargs):
    """A class for making checkpoints.

    If ckpt_dir doesn't exists it creates it.

    Args:
      ckpt_dir: The directory to save checkpoints.
      max_to_keep: Maximum number of checkpoints to keep (if greater than the
        max are saved, the oldest checkpoints are deleted).
      **kwargs: Items to include in the checkpoint.
    """
    self._checkpoint = tf.train.Checkpoint(**kwargs)

    if not tf.io.gfile.exists(ckpt_dir):
      tf.io.gfile.makedirs(ckpt_dir)

    self._manager = tf.train.CheckpointManager(
        self._checkpoint, directory=ckpt_dir, max_to_keep=max_to_keep)

    if self._manager.latest_checkpoint is not None:
      logging.info(
          '%s',
          'Checkpoint available: {}'.format(self._manager.latest_checkpoint))
    else:
      logging.info('%s', 'No checkpoint available at {}'.format(ckpt_dir))
    self._load_status = self._checkpoint.restore(
        self._manager.latest_checkpoint)

  def initialize_or_restore(self, session=None):
    """Initialize or restore graph (based on checkpoint if exists)."""
    self._load_status.initialize_or_restore(session)
    return self._load_status

  def save(self, global_step):
    """Save state to checkpoint."""
    saved_checkpoint = self._manager.save(checkpoint_number=global_step)
    logging.info('%s', 'Saved checkpoint: {}'.format(saved_checkpoint))


def replicate(tensor, outer_shape):
  """Replicates a tensor so as to match the given outer shape.

  Example:
  - t = [[1, 2, 3], [4, 5, 6]] (shape = [2, 3])
  - outer_shape = [2, 1]
  The shape of the resulting tensor is: [2, 1, 2, 3]
  and its content is: [[t], [t]]

  Args:
    tensor: A tf.Tensor.
    outer_shape: Outer shape given as a 1D tensor of type list, numpy or
      tf.Tensor.

  Returns:
    The replicated tensor.

  Raises:
    ValueError: when the outer shape is incorrect.
  """
  outer_shape = tf.convert_to_tensor(value=outer_shape)
  if len(outer_shape.shape) != 1:
    raise ValueError('The outer shape must be a 1D tensor')
  outer_ndims = int(outer_shape.shape[0])
  tensor_ndims = len(tensor.shape)

  # No need to replicate anything if there is no outer dim to add.
  if outer_ndims == 0:
    return tensor

  # Replicate tensor "t" along the 1st dimension.
  tiled_tensor = tf.tile(tensor, [tf.reduce_prod(input_tensor=outer_shape)] +
                         [1] * (tensor_ndims - 1))

  # Reshape to match outer_shape.
  target_shape = tf.concat([outer_shape, tf.shape(input=tensor)], axis=0)
  return tf.reshape(tiled_tensor, target_shape)


def assert_members_are_not_overridden(base_cls,
                                      instance,
                                      white_list=(),
                                      black_list=()):
  """Asserts public members of `base_cls` are not overridden in `instance`.

  If both `white_list` and `black_list` are empty, no public member of
  `base_cls` can be overridden. If a `white_list` is provided, only public
  members in `white_list` can be overridden. If a `black_list` is provided,
  all public members except those in `black_list` can be overridden. Both
  `white_list` and `black_list` cannot be provided at the same, if so a
  ValueError will be raised.

  Args:
    base_cls: A Base class.
    instance: An instance of a subclass of `base_cls`.
    white_list: Optional list of `base_cls` members that can be overridden.
    black_list: Optional list of `base_cls` members that cannot be overridden.

  Raises:
    ValueError if both white_list and black_list are provided.
  """

  if black_list and white_list:
    raise ValueError('Both `black_list` and `white_list` cannot be provided.')

  instance_type = type(instance)
  subclass_members = set(instance_type.__dict__.keys())
  public_members = set(
      [m for m in base_cls.__dict__.keys() if not m.startswith('_')])
  common_members = public_members & subclass_members

  if white_list:
    common_members = common_members - set(white_list)
  elif black_list:
    common_members = common_members & set(black_list)

  overridden_members = [
      m for m in common_members
      if base_cls.__dict__[m] != instance_type.__dict__[m]
  ]
  if overridden_members:
    raise ValueError(
        'Subclasses of {} cannot override most of its base members, but '
        '{} overrides: {}'.format(base_cls, instance_type, overridden_members))


def element_wise_squared_loss(x, y):
  return tf.compat.v1.losses.mean_squared_error(
      x, y, reduction=tf.compat.v1.losses.Reduction.NONE)


def element_wise_huber_loss(x, y):
  return tf.compat.v1.losses.huber_loss(
      x, y, reduction=tf.compat.v1.losses.Reduction.NONE)


def transpose_batch_time(x):
  """Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the first two dimensions.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.rank is not None and x_static_shape.rank < 2:
    return x

  x_rank = tf.rank(x)
  x_t = tf.transpose(a=x, perm=tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape(
          [x_static_shape.dims[1].value,
           x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
  return x_t


def save_spec(spec, file_path):
  """Saves the given spec nest as a StructProto.

  **Note**: Currently this will convert BoundedTensorSpecs into regular
    TensorSpecs.

  Args:
    spec: A nested structure of TensorSpecs.
    file_path: Path to save the encoded spec to.
  """
  signature_encoder = nested_structure_coder.StructureCoder()
  spec = tensor_spec.from_spec(spec)
  spec_proto = signature_encoder.encode_structure(spec)

  dir_path = os.path.dirname(file_path)
  if not tf.io.gfile.exists(dir_path):
    tf.io.gfile.makedirs(dir_path)

  with tf.compat.v2.io.gfile.GFile(file_path, 'wb') as gfile:
    gfile.write(spec_proto.SerializeToString())


def load_spec(file_path):
  """Loads a data spec from a file.

  **Note**: Types for Named tuple classes will not match. Users need to convert
    to these manually:

    # Convert from:
    # 'tensorflow.python.saved_model.nested_structure_coder.Trajectory'
    # to proper TrajectorySpec.
    # trajectory_spec = trajectory.Trajectory(*spec)

  Args:
    file_path: Path to the saved data spec.
  Returns:
    A nested structure of TensorSpecs.
  """
  with tf.compat.v2.io.gfile.GFile(file_path, 'rb') as gfile:
    signature_proto = struct_pb2.StructuredValue.FromString(gfile.read())

  signature_encoder = nested_structure_coder.StructureCoder()
  return signature_encoder.decode_proto(signature_proto)
