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

"""Common mock env and policy for testing drivers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from typing import Any

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.policies import py_policy
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


def make_replay_buffer(policy):
  """Default replay buffer factory."""
  return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      policy.trajectory_spec, batch_size=1)


class PyEnvironmentMock(py_environment.PyEnvironment):
  """Dummy Blackjack-like environment that increments `state` by `action`.

  The environment resets when state becomes greater or equal than final_state.
  Actions are 1 or 2.
  A reward of 1 for all non restart states.
  """

  def __init__(self, final_state=3):
    self._state = np.int32(0)
    self._action_spec = specs.BoundedArraySpec([],
                                               np.int32,
                                               minimum=1,
                                               maximum=2,
                                               name='action')
    self._observation_spec = specs.ArraySpec([], np.int32, name='observation')
    self._final_state = final_state
    super(PyEnvironmentMock, self).__init__()

  @property
  def batched(self):
    return False

  def _reset(self):
    self._state = np.int32(0)
    return ts.restart(self._state)

  def _step(self, action):
    if action < self._action_spec.minimum or action > self._action_spec.maximum:
      raise ValueError('Action should be in [{0}, {1}], but saw: {2}'.format(
          self._action_spec.minimum, self._action_spec.maximum, action))
    if action.shape != ():  # pylint: disable=g-explicit-bool-comparison
      raise ValueError('Action should be a scalar.')

    if self._state >= self._final_state:
      # Start a new episode. Ignore action
      return self.reset()

    self._state += action
    self._state = np.int32(self._state)
    if self._state < self._final_state:
      return ts.transition(self._state, 1.)
    else:
      return ts.termination(self._state, 1.)

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def get_info(self) -> Any:
    return {}


class TFPolicyMock(tf_policy.TFPolicy):
  """Mock policy takes actions 1 and 2, alternating."""

  def __init__(self,
               time_step_spec,
               action_spec,
               batch_size=1,
               policy_state_spec_name='policy_state_spec',
               policy_state_name='policy_state',
               initial_policy_state=None):
    batch_shape = (batch_size,)
    self._batch_shape = batch_shape
    minimum = np.asarray(1, dtype=np.int32)
    maximum = np.asarray(2, dtype=np.int32)
    self._maximum = maximum
    policy_state_spec = specs.BoundedTensorSpec((),
                                                tf.int32,
                                                minimum=minimum,
                                                maximum=maximum,
                                                name=policy_state_spec_name)
    info_spec = action_spec
    self._policy_state = common.create_variable(
        name=policy_state_name,
        initial_value=maximum,
        shape=batch_shape,
        dtype=tf.int32)
    if initial_policy_state is None:
      self._initial_policy_state = tf.fill([batch_size],
                                           tf.constant(0, tf.int32))
    else:
      self._initial_policy_state = initial_policy_state

    super(TFPolicyMock, self).__init__(time_step_spec, action_spec,
                                       policy_state_spec, info_spec)

  def _get_initial_state(self, batch_size):
    return self._initial_policy_state

  def _action(self, time_step, policy_state, seed):
    del seed

    # Reset the policy for batch indices that have restarted episode.
    policy_state = tf.compat.v1.where(time_step.is_first(),
                                      self._initial_policy_state, policy_state)

    # Take actions 1 and 2 alternating.
    action = tf.cast(tf.math.floormod(policy_state, 2) + 1, tf.int32)
    new_policy_state = tf.cast(policy_state + tf.constant(
        1, shape=self._batch_shape, dtype=tf.int32), tf.int32)
    policy_info = tf.cast(action * 2, tf.int32)
    return policy_step.PolicyStep(action, new_policy_state, policy_info)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError('Not implemented.')

  def _variables(self):
    return ()


class PyPolicyMock(py_policy.PyPolicy):
  """Mock policy takes actions 1 and 2, alternating."""

  # For batched environments, use a initial policy state of size [batch_size].
  def __init__(self,
               time_step_spec,
               action_spec,
               initial_policy_state=np.int32(2)):
    policy_state_spec = specs.BoundedArraySpec((),
                                               np.int32,
                                               minimum=1,
                                               maximum=2,
                                               name='policy_state_spec')
    policy_info_spec = specs.BoundedArraySpec((),
                                              np.int32,
                                              minimum=1,
                                              maximum=2,
                                              name='policy_info_spec')
    self._initial_policy_state = initial_policy_state
    super(PyPolicyMock, self).__init__(time_step_spec, action_spec,
                                       policy_state_spec, policy_info_spec)

  def _get_initial_state(self, batch_size=None):
    return self._initial_policy_state

  def _action(self, time_step, policy_state):
    # Reset the policy when starting a new episode.
    is_time_step_first = time_step.is_first()
    if np.isscalar(is_time_step_first):
      if is_time_step_first:
        policy_state = self._initial_policy_state
    else:
      policy_state[is_time_step_first] = self._initial_policy_state[
          is_time_step_first]

    # Take actions 1 and 2 alternating.
    action = (policy_state % 2) + 1
    policy_info = np.int32(action * 2)
    action = np.int32(action)
    policy_state = np.int32(policy_state + 1)
    return policy_step.PolicyStep(action, policy_state, policy_info)


class NumStepsObserver(object):
  """Class to count number of steps run by an observer."""

  def __init__(self, variable_scope='num_steps_step_observer'):
    with tf.compat.v1.variable_scope(variable_scope):
      self._num_steps = common.create_variable(
          'num_steps', 0, shape=[], dtype=tf.int32)

  @property
  def num_steps(self):
    return self._num_steps

  @num_steps.setter
  def num_steps(self, num_steps):
    self._num_steps.assign(num_steps)

  def __call__(self, traj):
    num_steps = tf.reduce_sum(
        input_tensor=tf.cast(~traj.is_boundary(), dtype=tf.int32))
    with tf.control_dependencies([self._num_steps.assign_add(num_steps)]):
      return tf.nest.map_structure(tf.identity, traj)


class NumStepsTransitionObserver(object):
  """Class to count number of steps run by an observer."""

  def __init__(self, variable_scope='num_steps_step_observer'):
    with tf.compat.v1.variable_scope(variable_scope):
      self._num_steps = common.create_variable(
          'num_steps', 0, shape=[], dtype=tf.int32)

  @property
  def num_steps(self):
    return self._num_steps

  @num_steps.setter
  def num_steps(self, num_steps):
    self._num_steps.assign(num_steps)

  def __call__(self, transition):
    _, _, next_time_step = transition
    num_steps = tf.reduce_sum(
        input_tensor=tf.cast(~next_time_step.is_first(), dtype=tf.int32))
    with tf.control_dependencies([self._num_steps.assign_add(num_steps)]):
      return tf.nest.map_structure(tf.identity, transition)


class NumEpisodesObserver(object):
  """Class to count number of episodes run by an observer."""

  def __init__(self, variable_scope='num_episodes_step_observer'):
    with tf.compat.v1.variable_scope(variable_scope):
      self._num_episodes = common.create_variable(
          'num_episodes', 0, shape=[], dtype=tf.int32)

  @property
  def num_episodes(self):
    return self._num_episodes

  @num_episodes.setter
  def num_episodes(self, num_episodes):
    self._num_episodes.assign(num_episodes)

  def __call__(self, traj):
    num_episodes = tf.reduce_sum(
        input_tensor=tf.cast(traj.is_boundary(), dtype=tf.int32))
    with tf.control_dependencies([
        self._num_episodes.assign_add(num_episodes)
    ]):
      return tf.nest.map_structure(tf.identity, traj)


def make_random_trajectory():
  """Creates a random trajectory.

  This trajectory contains Tensors shaped `[1, 6, ...]` where `1` is the batch
  and `6` is the number of time steps.

  Observations are unbounded but actions are bounded to take values within
  `[1, 2]`.

  Policy info is also provided, and is equal to the actions.  It can be removed
  via:

  ```python
  traj = make_random_trajectory().clone(policy_info=())
  ```

  Returns:
    A `Trajectory`.
  """
  time_step_spec = ts.time_step_spec(
      tensor_spec.TensorSpec([], tf.int32, name='observation'))
  action_spec = tensor_spec.BoundedTensorSpec([],
                                              tf.int32,
                                              minimum=1,
                                              maximum=2,
                                              name='action')
  # info and policy state specs match that of TFPolicyMock.
  outer_dims = [1, 6]  # (batch_size, time)
  traj = trajectory.Trajectory(
      observation=tensor_spec.sample_spec_nest(
          time_step_spec.observation, outer_dims=outer_dims),
      action=tensor_spec.sample_bounded_spec(
          action_spec, outer_dims=outer_dims),
      policy_info=tensor_spec.sample_bounded_spec(
          action_spec, outer_dims=outer_dims),
      reward=tf.fill(outer_dims, tf.constant(0, dtype=tf.float32)),
      # step_type is F M L F M L.
      step_type=tf.reshape(tf.range(0, 6) % 3, outer_dims),
      # next_step_type is M L F M L F.
      next_step_type=tf.reshape(tf.range(1, 7) % 3, outer_dims),
      discount=tf.fill(outer_dims, tf.constant(1, dtype=tf.float32)),
  )
  return traj, time_step_spec, action_spec
