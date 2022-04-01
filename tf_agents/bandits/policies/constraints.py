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

"""An API for representing constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
from typing import Callable, Iterable, Optional, Text
import six

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.policies import loss_utils
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import utils as policy_utilities
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@six.add_metaclass(abc.ABCMeta)
class BaseConstraint(tf.Module):
  """Abstract base class for representing constraints.

  The constraint class provides feasibility computation functionality for
  computing the probability of actions being feasible.
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      name: Optional[Text] = None):
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
  def __call__(self,
               observation: types.NestedTensor,
               actions: Optional[types.Tensor] = None) -> types.Tensor:
    """Returns the probability of input actions being feasible."""


class NeuralConstraint(BaseConstraint):
  """Class for representing a trainable constraint using a neural network.

  This constraint class uses a neural network to compute the action feasibility.
  In this case, the loss function needs to be exposed for training the neural
  network weights, typically done by the agent that uses this constraint.
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      constraint_network: Optional[types.Network],
      error_loss_fn: types.LossFn = tf.compat.v1.losses.mean_squared_error,
      name: Optional[Text] = 'NeuralConstraint'):
    """Creates a trainable constraint using a neural network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      constraint_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility. The input structure should be
        consistent with the `observation_spec`. If the constraint network is
        not available at construction time, it can be set later on using the
        constraint_network setter.
      error_loss_fn: A function for computing the loss used to train the
        constraint network. The default is `tf.losses.mean_squared_error`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    super(NeuralConstraint, self).__init__(
        time_step_spec,
        action_spec,
        name)

    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)
    if constraint_network is not None:
      with self.name_scope:
        constraint_network.create_variables()
    self._constraint_network = constraint_network
    self._error_loss_fn = error_loss_fn

  @property
  def constraint_network(self):
    return self._constraint_network

  @constraint_network.setter
  def constraint_network(self, constraint_network):
    if constraint_network is not None:
      with self.name_scope:
        constraint_network.create_variables()
    self._constraint_network = constraint_network

  def initialize(self):
    """Returns an op to initialize the constraint."""
    tf.compat.v1.variables_initializer(self.variables)

  def compute_loss(self,
                   observations: types.NestedTensor,
                   actions: types.NestedTensor,
                   rewards: types.Tensor,
                   weights: Optional[types.Float] = None,
                   training: bool = False) -> types.Tensor:
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
      sample_weights = weights if weights is not None else 1
      predicted_values, _ = self._constraint_network(
          observations, training=training)
      action_predicted_values = common.index_with_actions(
          predicted_values,
          tf.cast(actions, dtype=tf.int32))
      # Reduction is done outside of the loss function because non-scalar
      # weights with unknown shapes may trigger shape validation that fails
      # XLA compilation.
      return tf.reduce_mean(
          tf.multiply(
              self._error_loss_fn(
                  rewards,
                  action_predicted_values,
                  reduction=tf.compat.v1.losses.Reduction.NONE),
              sample_weights))

  def _reshape_and_broadcast(self, input_tensor: types.Tensor,
                             to_shape: types.Tensor) -> types.Tensor:
    input_tensor = tf.reshape(input_tensor, [-1, 1])
    return tf.broadcast_to(input_tensor, to_shape)

  # Subclasses can override this function.
  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    batch_dims = nest_utils.get_outer_shape(
        observation, self._time_step_spec.observation)
    shape = tf.concat([batch_dims, tf.constant(
        self._num_actions, shape=[1], dtype=batch_dims.dtype)], axis=-1)
    return tf.ones(shape)


class RelativeConstraint(NeuralConstraint):
  """Class for representing a trainable relative constraint.

  This constraint class implements a relative constraint such as
  ```
  expected_value(action) >= (1 - margin) * expected_value(baseline_action)
  ```
  or
  ```
  expected_value(action) <= (1 - margin) * expected_value(baseline_action)
  ```
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      constraint_network: types.Network,
      error_loss_fn: types.LossFn = tf.compat.v1.losses.mean_squared_error,
      comparator_fn: types.ComparatorFn = tf.greater,
      margin: float = 0.0,
      baseline_action_fn: Optional[Callable[[types.NestedTensor],
                                            types.Tensor]] = None,
      name: Text = 'RelativeConstraint'):
    """Creates a trainable relative constraint using a neural network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      constraint_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility. The input structure should be
        consistent with the `observation_spec`.
      error_loss_fn: A function for computing the loss used to train the
        constraint network. The default is `tf.losses.mean_squared_error`.
      comparator_fn: A comparator function, such as tf.greater or tf.less.
      margin: A float in (0,1] that determines how strongly we want to enforce
        the constraint.
      baseline_action_fn: a callable that given the observation returns the
         baseline action. If None, the baseline action is set to 0.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    self._baseline_action_fn = baseline_action_fn
    self._comparator_fn = comparator_fn
    self._error_loss_fn = error_loss_fn
    self._margin = margin

    super(RelativeConstraint, self).__init__(
        time_step_spec,
        action_spec,
        constraint_network,
        error_loss_fn=self._error_loss_fn,
        name=name)

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    predicted_values, _ = self._constraint_network(
        observation, training=False)

    batch_dims = nest_utils.get_outer_shape(
        observation, self._time_step_spec.observation)
    if self._baseline_action_fn is not None:
      baseline_action = self._baseline_action_fn(observation)
      baseline_action.shape.assert_is_compatible_with(batch_dims)
    else:
      baseline_action = tf.zeros(batch_dims, dtype=tf.int32)

    predicted_values_for_baseline_actions = common.index_with_actions(
        predicted_values,
        tf.cast(baseline_action, dtype=tf.int32))
    predicted_values_for_baseline_actions = self._reshape_and_broadcast(
        predicted_values_for_baseline_actions, tf.shape(predicted_values))
    is_satisfied = self._comparator_fn(
        predicted_values,
        (1 - self._margin) * predicted_values_for_baseline_actions)
    return tf.cast(is_satisfied, tf.float32)


class AbsoluteConstraint(NeuralConstraint):
  """Class for representing a trainable absolute value constraint.

  This constraint class implements an absolute value constraint such as
  ```
  expected_value(action) >= absolute_value
  ```
  or
  ```
  expected_value(action) <= absolute_value
  ```
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      constraint_network: types.Network,
      error_loss_fn: types.LossFn = tf.compat.v1.losses.mean_squared_error,
      comparator_fn: types.ComparatorFn = tf.greater,
      absolute_value: float = 0.0,
      name: Text = 'AbsoluteConstraint'):
    """Creates a trainable absolute constraint using a neural network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      constraint_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility.  The input structure should be
        consistent with the `observation_spec`.
      error_loss_fn: A function for computing the loss used to train the
        constraint network. The default is `tf.losses.mean_squared_error`.
      comparator_fn: a comparator function, such as tf.greater or tf.less.
      absolute_value: the threshold value we want to use in the constraint.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    self._absolute_value = absolute_value
    self._comparator_fn = comparator_fn
    self._error_loss_fn = error_loss_fn

    super(AbsoluteConstraint, self).__init__(
        time_step_spec,
        action_spec,
        constraint_network,
        error_loss_fn=self._error_loss_fn,
        name=name)

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    predicted_values, _ = self._constraint_network(
        observation, training=False)
    is_satisfied = self._comparator_fn(
        predicted_values, self._absolute_value)
    return tf.cast(is_satisfied, tf.float32)


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
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      constraint_network: types.Network,
      quantile: float = 0.5,
      comparator_fn: types.ComparatorFn = tf.greater,
      quantile_value: float = 0.0,
      name: Text = 'QuantileConstraint'):
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


class RelativeQuantileConstraint(NeuralConstraint):
  """Class for representing a trainable relative quantile constraint.

  This constraint class implements a relative quantile constraint such as
  ```
  Q_tau(action) >= Q_tau(baseline_action)
  ```
  or
  ```
  Q_tau(action) <= Q_tau(baseline_action)
  ```
  """

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.BoundedTensorSpec,
               constraint_network: types.Network,
               quantile: float = 0.5,
               comparator_fn: types.ComparatorFn = tf.greater,
               baseline_action_fn: Optional[Callable[[types.Tensor],
                                                     types.Tensor]] = None,
               name: Text = 'RelativeQuantileConstraint'):
    """Creates a trainable relative quantile constraint using a neural network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      constraint_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility.  The input structure should be
        consistent with the `observation_spec`.
      quantile: A float between 0. and 1., the quantile we want to regress.
      comparator_fn: a comparator function, such as tf.greater or tf.less.
      baseline_action_fn: a callable that given the observation returns the
         baseline action. If None, the baseline action is set to 0.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    self._baseline_action_fn = baseline_action_fn
    self._comparator_fn = comparator_fn
    self._error_loss_fn = functools.partial(
        loss_utils.pinball_loss,
        quantile=quantile)

    super(RelativeQuantileConstraint, self).__init__(
        time_step_spec,
        action_spec,
        constraint_network,
        error_loss_fn=self._error_loss_fn,
        name=name)

  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""
    predicted_quantiles, _ = self._constraint_network(
        observation, training=False)
    batch_dims = nest_utils.get_outer_shape(
        observation, self._time_step_spec.observation)

    if self._baseline_action_fn is not None:
      baseline_action = self._baseline_action_fn(observation)
      baseline_action.shape.assert_is_compatible_with(batch_dims)
    else:
      baseline_action = tf.zeros(batch_dims, dtype=tf.int32)

    predicted_quantiles_for_baseline_actions = common.index_with_actions(
        predicted_quantiles,
        tf.cast(baseline_action, dtype=tf.int32))
    predicted_quantiles_for_baseline_actions = self._reshape_and_broadcast(
        predicted_quantiles_for_baseline_actions, tf.shape(predicted_quantiles))
    is_satisfied = self._comparator_fn(
        predicted_quantiles, predicted_quantiles_for_baseline_actions)
    return tf.cast(is_satisfied, tf.float32)


class InputNetworkConstraint(BaseConstraint):
  """Class for representing a constraint using an input network.

  This constraint class uses an input network to compute the action feasibility.
  It assumes that the input network is already trained and it can be provided
  at construction time or later using the set_network() function.
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      input_network: Optional[types.Network] = None,
      name: Optional[Text] = 'InputNetworkConstraint'):
    """Creates a constraint using an input network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      input_network: An instance of `tf_agents.network.Network` used to
        provide estimates of action feasibility.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.
    """
    super(InputNetworkConstraint, self).__init__(
        time_step_spec,
        action_spec,
        name)
    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)
    self._network = input_network

  @property
  def network(self):
    return self._network

  @network.setter
  def network(self, input_network):
    self._network = input_network

  def compute_loss(self,
                   observations: types.NestedTensor,
                   actions: types.NestedTensor,
                   rewards: types.Tensor,
                   weights: Optional[types.TensorOrArray] = None,
                   training: bool = False) -> types.Tensor:
    with tf.name_scope('constraint_loss'):
      return tf.constant(0.0)

  # Subclasses must implement these methods.
  @abc.abstractmethod
  def __call__(self, observation, actions=None):
    """Returns the probability of input actions being feasible."""


def compute_feasibility_probability(
    observation: types.NestedTensor,
    constraints: Iterable[BaseConstraint],
    batch_size: types.Int,
    num_actions: int,
    action_mask: Optional[types.Tensor] = None) -> types.Float:
  """Helper function to compute the action feasibility probability."""
  feasibility_prob = tf.ones([batch_size, num_actions])
  if action_mask is not None:
    feasibility_prob = tf.cast(action_mask, tf.float32)
  for c in constraints:
    # We assume the constraints are independent.
    action_feasibility = c(observation)
    feasibility_prob *= action_feasibility
  return feasibility_prob


def construct_mask_from_multiple_sources(
    observation: types.NestedTensor,
    observation_and_action_constraint_splitter: types.Splitter,
    constraints: Iterable[BaseConstraint],
    max_num_actions: int) -> Optional[types.Tensor]:
  """Constructs an action mask from multiple sources.

  The sources include:
  -- The action mask encoded in the observation,
  -- the `num_actions` feature restricting the number of actions per sample,
  -- the feasibility mask implied by constraints.

  The resulting mask disables all actions that are masked out in any of the
  three sources.

  Args:
    observation: A nest of Tensors containing the observation.
    observation_and_action_constraint_splitter: The observation action mask
      splitter function if the observation has action mask.
    constraints: Iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.NeuralConstraint`.
    max_num_actions: The maximum number of actions per sample.

  Returns:
    An action mask in the form of a `[batch_size, max_num_actions]` 0-1 tensor.
  """
  mask = None
  if observation_and_action_constraint_splitter is not None:
    observation, mask = observation_and_action_constraint_splitter(observation)
  elif (isinstance(observation, dict) and
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY in observation):
    number_of_actions = observation[bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY]
    mask = tf.sequence_mask(
        lengths=number_of_actions, maxlen=max_num_actions, dtype=tf.int32)

  first_observation = tf.nest.flatten(observation)[0]
  batch_size = tf.shape(first_observation)[0]
  if constraints:
    feasibility_prob = compute_feasibility_probability(
        observation, constraints, batch_size,
        max_num_actions, mask)
    # Probabilistic masking.
    mask = tfp.distributions.Bernoulli(probs=feasibility_prob).sample()
  return mask
