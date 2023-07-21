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

"""Methods for computing advantages and target values.
"""

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import


def discounted_return(rewards,
                      discounts,
                      final_value=None,
                      time_major=True,
                      provide_all_returns=True):
  """Computes discounted return.

  ```
  Q_n = sum_{n'=n}^N gamma^(n'-n) * r_{n'} + gamma^(N-n+1)*final_value.
  ```

  For details, see
  "Reinforcement Learning: An Introduction" Second Edition
  by Richard S. Sutton and Andrew G. Barto

  Define abbreviations:
  `B`: batch size representing number of trajectories.
  `T`: number of steps per trajectory.  This is equal to `N - n` in the equation
       above.

  **Note** To replicate the calculation `Q_n` exactly, use
  `discounts = gamma * tf.ones_like(rewards)` and `provide_all_returns=False`.

  Args:
    rewards: Tensor with shape `[T, B]` (or `[T]`) representing rewards.
    discounts: Tensor with shape `[T, B]` (or `[T]`) representing discounts.
    final_value: (Optional.).  Default: An all zeros tensor.  Tensor with shape
      `[B]` (or `[1]`) representing value estimate at `T`. This is optional;
      when set, it allows final value to bootstrap the reward computation.
    time_major: A boolean indicating whether input tensors are time major. False
      means input tensors have shape `[B, T]`.
    provide_all_returns: A boolean; if True, this will provide all of the
      returns by time dimension; if False, this will only give the single
      complete discounted return.

  Returns:
    If `provide_all_returns`:
      A tensor with shape `[T, B]` (or `[T]`) representing the discounted
      returns. The shape is `[B, T]` when `not time_major`.
    If `not provide_all_returns`:
      A tensor with shape `[B]` (or []) representing the discounted returns.
  """
  if not time_major:
    with tf.name_scope("to_time_major_tensors"):
      discounts = tf.transpose(discounts)
      rewards = tf.transpose(rewards)

  if final_value is None:
    final_value = tf.zeros_like(rewards[-1])

  def discounted_return_fn(accumulated_discounted_reward, reward_discount):
    reward, discount = reward_discount
    return accumulated_discounted_reward * discount + reward

  if provide_all_returns:
    returns = tf.nest.map_structure(
        tf.stop_gradient,
        tf.scan(
            fn=discounted_return_fn,
            elems=(rewards, discounts),
            reverse=True,
            initializer=final_value))

    if not time_major:
      with tf.name_scope("to_batch_major_tensors"):
        returns = tf.transpose(returns)
  else:
    returns = tf.foldr(
        fn=discounted_return_fn,
        elems=(rewards, discounts),
        initializer=final_value,
        back_prop=False)

  return tf.stop_gradient(returns)


def generalized_advantage_estimation(values,
                                     final_value,
                                     discounts,
                                     rewards,
                                     td_lambda=1.0,
                                     time_major=True):
  """Computes generalized advantage estimation (GAE).

  For theory, see
  "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  by John Schulman, Philipp Moritz et al.
  See https://arxiv.org/abs/1506.02438 for full paper.

  Define abbreviations:
    (B) batch size representing number of trajectories
    (T) number of steps per trajectory

  Args:
    values: Tensor with shape `[T, B]` representing value estimates.
    final_value: Tensor with shape `[B]` representing value estimate at t=T.
    discounts: Tensor with shape `[T, B]` representing discounts received by
      following the behavior policy.
    rewards: Tensor with shape `[T, B]` representing rewards received by
      following the behavior policy.
    td_lambda: A float32 scalar between [0, 1]. It's used for variance reduction
      in temporal difference.
    time_major: A boolean indicating whether input tensors are time major.
      False means input tensors have shape `[B, T]`.

  Returns:
    A tensor with shape `[T, B]` representing advantages. Shape is `[B, T]` when
    `not time_major`.
  """

  if not time_major:
    with tf.name_scope("to_time_major_tensors"):
      discounts = tf.transpose(discounts)
      rewards = tf.transpose(rewards)
      values = tf.transpose(values)

  with tf.name_scope("gae"):

    next_values = tf.concat(
        [values[1:], tf.expand_dims(final_value, 0)], axis=0)
    delta = rewards + discounts * next_values - values
    weighted_discounts = discounts * td_lambda

    def weighted_cumulative_td_fn(accumulated_td, reversed_weights_td_tuple):
      weighted_discount, td = reversed_weights_td_tuple
      return td + weighted_discount * accumulated_td

    advantages = tf.nest.map_structure(
        tf.stop_gradient,
        tf.scan(
            fn=weighted_cumulative_td_fn,
            elems=(weighted_discounts, delta),
            initializer=tf.zeros_like(final_value),
            reverse=True))

  if not time_major:
    with tf.name_scope("to_batch_major_tensors"):
      advantages = tf.transpose(advantages)

  return tf.stop_gradient(advantages)
