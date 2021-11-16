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

"""Bandit environment that returns random observations and rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import gin

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import bandit_tf_environment as bte
from tf_agents.trajectories import time_step
from tf_agents.typing import types
from tf_agents.utils import common


@gin.configurable
class EnvironmentDynamics(tf.Module, metaclass=abc.ABCMeta):
  """Abstract class to represent a non-stationary environment dynamics.

  This class is used with the NonStationaryStochasticEnvironment class below to
  obtain a non-stationary environment.
  To define a dynamics, derive from this class and define the abstract methods
  and properties below.
  To work correctly with graph and eager mode, Tensorflow variables must be
  defined in the constructor of this class. When used within a
  `BanditTFEnvironment` autodeps in reset and step functions will handle
  automatically the operation order.

  """

  @abc.abstractproperty
  def batch_size(self) -> types.Int:
    """Returns the batch size used for observations and rewards."""
    pass

  @abc.abstractproperty
  def observation_spec(self) -> types.TensorSpec:
    """Specification of the observations."""
    pass

  @abc.abstractproperty
  def action_spec(self) -> types.TensorSpec:
    """Specification of the actions."""
    pass

  @abc.abstractmethod
  def observation(self, env_time: types.Int) -> types.NestedTensor:
    """Returns an observation batch for the given time.

    Args:
      env_time: The scalar int64 tensor of the environment time step. This is
        incremented by the environment after the reward is computed.

    Returns:
      The observation batch with spec according to `observation_spec.`
    """
    pass

  @abc.abstractmethod
  def reward(self,
             observation: types.NestedTensor,
             env_time: types.Int) -> types.NestedTensor:
    """Reward for the given observation and time step.

    Args:
      observation: A batch of observations with spec according to
        `observation_spec.`
      env_time: The scalar int64 tensor of the environment time step. This is
        incremented by the environment after the reward is computed.

    Returns:
      A batch of rewards with spec shape [batch_size, num_actions] containing
      rewards for all arms.
    """
    pass


def _create_variable_from_spec_nest(specs, batch_size):
  def create_variable(spec):
    return common.create_variable(
        name=spec.name,
        dtype=spec.dtype,
        shape=[batch_size] + spec.shape.as_list())
  return tf.nest.map_structure(create_variable, specs)


def _assign_variable_nest(variables, values):
  return tf.nest.map_structure(lambda variable, value: variable.assign(value),
                               variables,
                               values)


def _read_value_nest(variables):
  return tf.nest.map_structure(lambda variable: variable.read_value(),
                               variables)


@gin.configurable
class NonStationaryStochasticEnvironment(bte.BanditTFEnvironment):
  """Implements a general non-stationary environment.

  This environment keeps a Tensorflow variable (`env_time`) to keep track of the
  current timme. This is incremented after every update of the reward tensor.
  The `EnvironmentDynamics` object passed to the constructor determines how
  observations and rewards are computed for the current time.
  """

  def __init__(self, environment_dynamics: EnvironmentDynamics):
    """Initializes a non-stationary environment with the given dynamics.

    Args:
      environment_dynamics: An instance of `EnvironmentDynamics` defining how
        the environment evolves over time.
    """
    self._env_time = tf.compat.v2.Variable(
        0, trainable=False, name='env_time', dtype=tf.int64)
    self._environment_dynamics = environment_dynamics
    observation_spec = environment_dynamics.observation_spec
    self._observation = _create_variable_from_spec_nest(
        observation_spec, environment_dynamics.batch_size)
    time_step_spec = time_step.time_step_spec(observation_spec)
    super(NonStationaryStochasticEnvironment, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=environment_dynamics.action_spec,
        batch_size=environment_dynamics.batch_size)

  @property
  def environment_dynamics(self) -> EnvironmentDynamics:
    return self._environment_dynamics

  def _apply_action(self, action: types.NestedTensor) -> types.NestedTensor:
    self._reward = self._environment_dynamics.reward(self._observation,
                                                     self._env_time)
    tf.compat.v1.assign_add(self._env_time,
                            self._environment_dynamics.batch_size)
    return common.index_with_actions(
        self._reward, tf.cast(action, dtype=tf.int32))

  def _observe(self) -> types.NestedTensor:
    _assign_variable_nest(
        self._observation,
        self._environment_dynamics.observation(self._env_time))
    return _read_value_nest(self._observation)
