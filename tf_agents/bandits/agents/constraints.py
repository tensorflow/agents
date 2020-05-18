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

"""An API for representing constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import functools
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.agents import loss_utils
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class BaseConstraint(tf.Module):
  """Abstract base class for representing constraints.

  The constraint class provides feasibility computation functionality for
  computing the probability of actions being feasible.
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      name=None):
    """Initialization of the BaseConstraint class.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      name: Python str name of this constraint.
    """
    super(BaseConstraint, self).__init__(name=name)
    if not isinstance(time_step_spec, ts.TimeStep):
      raise ValueError(
          'The `time_step_spec` must be an instance of `TimeStep`, but is `{}`.'
          .format(type(time_step_spec)))

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec

  # Subclasses must implement these methods.
  @abc.abstractmethod
  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""


class NeuralConstraint(BaseConstraint):
  """Class for representing a trainable constraint using a neural network.

  This constraint class uses a neural network to compute the action feasibility.
  In this case, the loss function needs to be exposed for training the neural
  network weights, typically done by the agent that uses this constraint.
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      constraint_network,
      error_loss_fn=tf.compat.v1.losses.mean_squared_error,
      name='NeuralConstraint'):
    """Creates a trainable constraint using a neural network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      constraint_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility. The input structure should be
        consistent with the `observation_spec`.
      error_loss_fn: A function for computing the loss used to train the
        constraint network. The default is `tf.losses.mean_squared_error`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    super(NeuralConstraint, self).__init__(
        time_step_spec,
        action_spec,
        name)

    self._num_actions = bandit_utils.get_num_actions_from_tensor_spec(
        action_spec)

    with self.name_scope:
      constraint_network.create_variables()
    self._constraint_network = constraint_network
    self._error_loss_fn = error_loss_fn

  def initialize(self):
    """Returns an op to initialize the constraint."""
    tf.compat.v1.variables_initializer(self.variables)

  def compute_loss(
      self, observations, actions, rewards, weights=None, training=False):
    """Computes loss for training the constraint network.

    Args:
      observations: A batch of observations.
      actions: A batch of actions.
      rewards: A batch of rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `Tensor` containing the loss for the training step.
    """
    with tf.name_scope('constraint_loss'):
      sample_weights = weights if weights else 1
      predicted_values, _ = self._constraint_network(
          observations, training=training)
      action_predicted_values = common.index_with_actions(
          predicted_values,
          tf.cast(actions, dtype=tf.int32))
      loss = self._error_loss_fn(
          rewards,
          action_predicted_values,
          sample_weights,
          reduction=tf.compat.v1.losses.Reduction.MEAN)
      return loss

  # Subclasses can override this function.
  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    batch_dims = nest_utils.get_outer_shape(
        observation, self._time_step_spec.observation)
    shape = tf.concat([batch_dims, tf.constant(
        self._num_actions, shape=[1], dtype=batch_dims.dtype)], axis=-1)
    return tf.ones(shape)


class QuantileConstraint(NeuralConstraint):
  """Class for representing a trainable quantile constraint.

  This constraint class implements a quantile constraint such as
  ```
  Q_tau(x) >= v
  ```
  or
  ```
  Q_tau(x) <= v
  ```
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      constraint_network,
      quantile=0.5,
      comparator_fn=tf.greater,
      quantile_value=0.0,
      name='QuantileConstraint'):
    """Creates a trainable quantile constraint using a neural network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      constraint_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility.  The input structure should be
        consistent with the `observation_spec`.
      quantile: A float between 0. and 1., the quantile we want to regress.
      comparator_fn: a comparator function, such as tf.greater or tf.less.
      quantile_value: the desired bound (float) we want to enforce on the
        quantile.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    self._quantile_value = quantile_value
    self._comparator_fn = comparator_fn
    self._error_loss_fn = functools.partial(
        loss_utils.pinball_loss,
        quantile=quantile)

    super(QuantileConstraint, self).__init__(
        time_step_spec,
        action_spec,
        constraint_network,
        error_loss_fn=self._error_loss_fn,
        name=name)

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    predicted_quantiles, _ = self._constraint_network(
        observation, training=False)
    is_satisfied = self._comparator_fn(
        predicted_quantiles, self._quantile_value)
    return tf.cast(is_satisfied, tf.float32)
