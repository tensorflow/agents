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

"""Common mock env and policy for testing drivers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.environments import time_step as ts
from tf_agents.policies import policy_step
from tf_agents.policies import py_policy
from tf_agents.policies import tf_policy


class PyEnvironmentMock(py_environment.Base):
  """Dummy Blackjack-like environment that increments `state` by `action`.

  The environment resets when state becomes greater or equal than final_state.
  Actions are 1 or 2.
  A reward of 1 for all non restart states.
  """

  def __init__(self, final_state=3):
    self._state = 0
    self._action_spec = specs.BoundedArraySpec([],
                                               np.int32,
                                               minimum=1,
                                               maximum=2,
                                               name='action')
    self._observation_spec = specs.ArraySpec([], np.int64, name='observation')
    self._final_state = final_state

  @property
  def batched(self):
    return False

  def reset(self):
    self._state = 0
    return ts.restart(self._state)

  def step(self, action):
    if action < self._action_spec.minimum or action > self._action_spec.maximum:
      raise ValueError('Action should be in [{0}, {1}], but saw: {2}'.format(
          self._action_spec.minimum, self._action_spec.maximum,
          action))

    if self._state >= self._final_state:
      # Start a new episode. Ignore action
      self._state = 0
      return ts.restart(self._state)

    self._state += action
    if self._state < self._final_state:
      return ts.transition(self._state, 1.)
    else:
      return ts.termination(self._state, 1.)

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec


class TFPolicyMock(tf_policy.Base):
  """Mock policy takes actions 1 and 2, alternating."""

  def __init__(self, time_step_spec, action_spec, batch_size=1):
    batch_shape = (batch_size,)
    self._batch_shape = batch_shape
    minimum = np.asarray(1, dtype=np.int32)
    maximum = np.asarray(2, dtype=np.int32)
    self._maximum = maximum
    policy_state_spec = specs.BoundedTensorSpec(
        (), tf.int32, minimum=minimum, maximum=maximum,
        name='policy_state_spec')
    info_spec = action_spec
    self._policy_state = tf.get_variable(
        name='policy_state',
        shape=batch_shape,
        dtype=tf.int32,
        initializer=tf.constant_initializer(maximum))
    self._initial_policy_state = tf.constant(
        0, shape=batch_shape, dtype=tf.int32)

    super(TFPolicyMock, self).__init__(time_step_spec, action_spec,
                                       policy_state_spec, info_spec)

  def _get_initial_state(self, batch_size):
    return tf.fill([batch_size], self._maximum)

  def _action(self, time_step, policy_state, seed):
    del seed

    # Reset the policy for batch indices that have restarted episode.
    policy_state = tf.where(time_step.is_first(),
                            self._initial_policy_state,
                            policy_state)

    # Take actions 1 and 2 alternating.
    action = tf.floormod(policy_state, 2) + 1
    new_policy_state = policy_state + tf.constant(
        1, shape=self._batch_shape, dtype=tf.int32)
    policy_info = action * 2
    return policy_step.PolicyStep(action, new_policy_state, policy_info)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError('Not implemented.')

  def _variables(self):
    return ()


class PyPolicyMock(py_policy.Base):
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
    self._initial_policy_state = initial_policy_state
    super(PyPolicyMock, self).__init__(time_step_spec, action_spec,
                                       policy_state_spec)

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
    policy_info = action * 2
    return policy_step.PolicyStep(action, policy_state + 1, policy_info)
