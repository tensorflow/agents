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

"""TF metrics for Bandits algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.metrics import tf_metric
from tf_agents.utils import common


@gin.configurable
class RegretMetric(tf_metric.TFStepMetric):
  """Computes the regret with respect to a baseline."""

  def __init__(self, baseline_reward_fn, name='RegretMetric', dtype=tf.float32):
    """Computes the regret with respect to a baseline.

    The regret is computed by computing the difference of the current reward
    from the baseline action reward. The latter is computed by calling the input
    `baseline_reward_fn` function that given a (batched) observation computes
    the baseline action reward.

    Args:
      baseline_reward_fn: function that computes the reward used as a baseline
        for computing the regret.
      name: (str) name of the metric
      dtype: dtype of the metric value.
    """
    self._baseline_reward_fn = baseline_reward_fn
    self.dtype = dtype
    self.regret = common.create_variable(
        initial_value=0, dtype=self.dtype, shape=(), name='regret')
    super(RegretMetric, self).__init__(name=name)

  def call(self, trajectory):
    """Update the regret value.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    baseline_reward = self._baseline_reward_fn(trajectory.observation)
    trajectory_reward = trajectory.reward
    if isinstance(trajectory.reward, dict):
      trajectory_reward = trajectory.reward[bandit_spec_utils.REWARD_SPEC_KEY]
    trajectory_regret = baseline_reward - trajectory_reward
    self.regret.assign(tf.reduce_mean(trajectory_regret))
    return trajectory

  def result(self):
    return tf.identity(
        self.regret, name=self.name)


@gin.configurable
class SuboptimalArmsMetric(tf_metric.TFStepMetric):
  """Computes the number of suboptimal arms with respect to a baseline."""

  def __init__(self, baseline_action_fn, name='SuboptimalArmsMetric',
               dtype=tf.float32):
    """Computes the number of suboptimal arms with respect to a baseline.

    Args:
      baseline_action_fn: function that computes the action used as a baseline
        for computing the metric.
      name: (str) name of the metric
      dtype: dtype of the metric value.
    """
    self._baseline_action_fn = baseline_action_fn
    self.dtype = dtype
    self.suboptimal_arms = common.create_variable(
        initial_value=0, dtype=self.dtype, shape=(), name='suboptimal_arms')
    super(SuboptimalArmsMetric, self).__init__(name=name)

  def call(self, trajectory):
    """Update the metric value.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    baseline_action = self._baseline_action_fn(trajectory.observation)
    disagreement = tf.cast(
        tf.not_equal(baseline_action, trajectory.action), tf.float32)
    self.suboptimal_arms.assign(tf.reduce_mean(disagreement))
    return trajectory

  def result(self):
    return tf.identity(
        self.suboptimal_arms, name=self.name)


@gin.configurable
class ConstraintViolationsMetric(tf_metric.TFStepMetric):
  """Computes the violations of a certain constraint."""

  def __init__(self,
               constraint,
               name='ConstraintViolationMetric',
               dtype=tf.float32):
    """Computes the constraint violations given an input constraint.

    Given a certain constraint, this metric computes how often the selected
    actions in the trajectory violate the constraint.

    Args:
      constraint: an instance of `tf_agents.bandits.agents.BaseConstraint`.
      name: (str) name of the metric
      dtype: dtype of the metric value.
    """
    self._constraint = constraint
    self.dtype = dtype
    self.constraint_violations = common.create_variable(
        initial_value=0.0,
        dtype=self.dtype,
        shape=(),
        name='constraint_violations')
    super(ConstraintViolationsMetric, self).__init__(name=name)

  def call(self, trajectory):
    """Update the constraint violations metric.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    feasibility_prob_all_actions = self._constraint(trajectory.observation)
    feasibility_prob_selected_actions = common.index_with_actions(
        feasibility_prob_all_actions,
        tf.cast(trajectory.action, dtype=tf.int32))
    self.constraint_violations.assign(tf.reduce_mean(
        1.0 - feasibility_prob_selected_actions))
    return trajectory

  def result(self):
    return tf.identity(self.constraint_violations, name=self.name)
