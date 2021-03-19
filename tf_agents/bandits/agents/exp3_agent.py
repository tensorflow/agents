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

"""Implements the EXP3 bandit algorithm.

Implementation based on

"Bandit Algorithms"
  Lattimore and Szepesvari, 2019
  https://tor-lattimore.com/downloads/book/book.pdf
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.bandits.policies import categorical_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import common


def selective_sum(values: types.Tensor, partitions: types.Int,
                  num_partitions: int) -> types.Tensor:
  """Sums entries in `values`, partitioned using `partitions`.

  For example,

  ```python
     # Returns `[0 + 4 + 5, 2 + 3 + 4]` i.e. `[9, 6]`.
     selective_sum(values=[0, 1, 2, 3, 4, 5],
                   partitions=[0, 1, 1, 1, 0, 0]),
                   num_partitions=2)
  ```

  Args:
    values: a `Tensor` with numerical type.
    partitions: an integer `Tensor` with the same shape as `values`. Entry
      `partitions[i]` indicates the partition to which `values[i]` belongs.
    num_partitions: the number of partitions. All values in `partitions` must
      lie in `[0, num_partitions)`.
  Returns:
    A vector of size `num_partitions` with the same dtype as `values`. Entry `i`
    is the sum of all entries in `values` belonging to partition `i`.
  """
  partitioned_values = tf.dynamic_partition(values, partitions, num_partitions)
  return tf.stack([tf.reduce_sum(partition)
                   for partition in partitioned_values])


def exp3_update_value(reward: types.Float,
                      log_prob: types.Float) -> types.Float:
  return 1. - (1. - reward) / tf.exp(log_prob)


@gin.configurable
class Exp3Agent(tf_agent.TFAgent):
  """An agent implementing the EXP3 bandit algorithm.

  Implementation based on

  "Bandit Algorithms"
    Lattimore and Szepesvari, 2019
    https://tor-lattimore.com/downloads/book/book.pdf
  """

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.BoundedTensorSpec,
               learning_rate: float,
               name: Optional[Text] = None):
    """Initialize an instance of `Exp3Agent`.

    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
      learning_rate: A float valued scalar. A higher value will force the agent
        to converge on a single action more quickly. A lower value will
        encourage more exploration. This value corresponds to the
        `inverse_temperature` argument passed to `CategoricalPolicy`.
      name: a name for this instance of `Exp3Agent`.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)
    self._weights = tf.compat.v2.Variable(
        tf.zeros(self._num_actions), name='weights')
    self._learning_rate = tf.compat.v2.Variable(
        learning_rate, name='learning_rate')
    policy = categorical_policy.CategoricalPolicy(
        weights=self._weights,
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        inverse_temperature=self._learning_rate)
    # TODO(b/127462472): consider policy=GreedyPolicy(collect_policy).
    super(Exp3Agent, self).__init__(time_step_spec=time_step_spec,
                                    action_spec=policy.action_spec,
                                    policy=policy,
                                    collect_policy=policy,
                                    train_sequence_length=None)
    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def weights(self):
    return tf.identity(self._weights)

  @property
  def learning_rate(self):
    return tf.identity(self._learning_rate)

  @learning_rate.setter
  def learning_rate(self, learning_rate):
    return tf.compat.v1.assign(self._learning_rate, learning_rate)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def _train(self, experience, weights=None):
    """Updates the policy based on the data in `experience`.

    Note that `experience` should only contain data points that this agent has
    not previously seen. If `experience` comes from a replay buffer, this buffer
    should be cleared between each call to `train`.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`.
      weights: Unused.

    Returns:
      A `LossInfo` containing the loss *before* the training step is taken.
        Note that the loss does not depend on policy state and comes directly
        from the experience (and is therefore not differentiable).

        In most cases, if `weights` is provided, the entries of this tuple will
        have been calculated with the weights.  Note that each Agent chooses
        its own method of applying weights.
    """
    del weights  # unused
    experience = self._as_trajectory(experience)
    reward = experience.reward
    log_prob = policy_step.get_log_probability(experience.policy_info)
    action = experience.action
    update_value = exp3_update_value(reward, log_prob)
    weight_update = selective_sum(values=update_value,
                                  partitions=action,
                                  num_partitions=self.num_actions)
    tf.compat.v1.assign_add(self._weights, weight_update)

    batch_size = tf.cast(tf.size(reward), dtype=tf.int64)
    self._train_step_counter.assign_add(batch_size)

    return tf_agent.LossInfo(loss=-tf.reduce_sum(experience.reward), extra=())
