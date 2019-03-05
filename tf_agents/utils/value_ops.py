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

"""Methods for computing advantages and target values.
"""

import tensorflow as tf


def discounted_return(rewards, discounts, final_value=None, time_major=True):
  """Computes discounted return.

  ```
  Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'} + gamma^(T-t+1)*final_value.
  ```

  For details, see
  "Reinforcement Learning: An Introduction" Second Edition
  by Richard S. Sutton and Andrew G. Barto

  Define abbreviations:
  (B) batch size representing number of trajectories
  (T) number of steps per trajectory

  Args:
    rewards: Tensor with shape [T, B] (or [T]) representing rewards.
    discounts: Tensor with shape [T, B] (or [T]) representing discounts.
    final_value: Tensor with shape [B] (or [1]) representing value estimate at
      t=T. This is optional, when set, it allows final value to bootstrap the
      reward to go computation. Otherwise it's zero.
    time_major: A boolean indicating whether input tensors are time major. False
      means input tensors have shape [B, T].

  Returns:
      A tensor with shape [T, B] (or [T]) representing the discounted returns.
      Shape is [B, T] when time_major is false.
  """
  if not time_major:
    with tf.name_scope("to_time_major_tensors"):
      discounts = tf.transpose(a=discounts)
      rewards = tf.transpose(a=rewards)

  if final_value is None:
    final_value = tf.zeros_like(rewards[-1])

  def discounted_return_fn(accumulated_discounted_reward, reward_discount):
    reward, discount = reward_discount
    return accumulated_discounted_reward * discount + reward

  returns = tf.scan(
      fn=discounted_return_fn,
      elems=(rewards, discounts),
      reverse=True,
      initializer=final_value,
      back_prop=False)

  if not time_major:
    with tf.name_scope("to_batch_major_tensors"):
      returns = tf.transpose(a=returns)

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
    values: Tensor with shape [T, B] representing value estimates.
    final_value: Tensor with shape [B] representing value estimate at t=T.
    discounts: Tensor with shape [T, B] representing discounts received by
      following the behavior policy.
    rewards: Tensor with shape [T, B] representing rewards received by following
      the behavior policy.
    td_lambda: A float32 scalar between [0, 1]. It's used for variance reduction
      in temporal difference.
    time_major: A boolean indicating whether input tensors are time major.
      False means input tensors have shape [B, T].

  Returns:
    A tensor with shape [T, B] representing advantages. Shape is [B, T] when
    time_major is false.
  """

  if not time_major:
    with tf.name_scope("to_time_major_tensors"):
      discounts = tf.transpose(a=discounts)
      rewards = tf.transpose(a=rewards)
      values = tf.transpose(a=values)

  with tf.name_scope("gae"):

    next_values = tf.concat(
        [values[1:], tf.expand_dims(final_value, 0)], axis=0)
    delta = rewards + discounts * next_values - values
    weighted_discounts = discounts * td_lambda

    def weighted_cumulative_td_fn(accumulated_td, reversed_weights_td_tuple):
      weighted_discount, td = reversed_weights_td_tuple
      return td + weighted_discount * accumulated_td

    advantages = tf.scan(
        fn=weighted_cumulative_td_fn,
        elems=(weighted_discounts, delta),
        initializer=tf.zeros_like(final_value),
        reverse=True,
        back_prop=False)

  if not time_major:
    with tf.name_scope("to_batch_major_tensors"):
      advantages = tf.transpose(a=advantages)

  return tf.stop_gradient(advantages)
