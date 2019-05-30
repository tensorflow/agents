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

"""Treat multiple non-batch environments as a single batch environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=line-too-long
# multiprocessing.dummy provides a pure *multithreaded* threadpool that works
# in both python2 and python3 (concurrent.futures isn't available in python2).
#   https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.dummy
from multiprocessing import dummy as mp_threads
# pylint: enable=line-too-long

import gin
import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.utils import nest_utils


@gin.configurable
class BatchedPyEnvironment(py_environment.PyEnvironment):
  """Batch together multiple py environments and act as a single batch.

  The environments should only access shared python variables using
  shared mutex locks (from the threading module).
  """

  def __init__(self, envs):
    """Batch together multiple (non-batched) py environments.

    The environments can be different but must use the same action and
    observation specs.

    Args:
      envs: List python environments (must be non-batched).

    Raises:
      ValueError: If envs is not a list or tuple, or is zero length, or if
        one of the envs is already batched.
      ValueError: If the action or observation specs don't match.
    """
    if not isinstance(envs, (list, tuple)):
      raise ValueError("envs must be a list or tuple.  Got: %s" % envs)
    batched_envs = [(i, env) for i, env in enumerate(envs) if env.batched]
    if batched_envs:
      raise ValueError(
          "Some of the envs are already batched: %s" % batched_envs)
    self._envs = envs
    self._num_envs = len(envs)
    self._action_spec = self._envs[0].action_spec()
    self._observation_spec = self._envs[0].observation_spec()
    self._time_step_spec = self._envs[0].time_step_spec()
    if any(env.action_spec() != self._action_spec for env in self._envs):
      raise ValueError(
          "All environments must have the same action spec.  Saw: %s" %
          [env.action_spec() for env in self._envs])
    if any(env.time_step_spec() != self._time_step_spec for env in self._envs):
      raise ValueError(
          "All environments must have the same time_step_spec.  Saw: %s" %
          [env.time_step_spec() for env in self._envs])
    # Create a multiprocessing threadpool for execution.
    self._pool = mp_threads.Pool(self._num_envs)
    super(BatchedPyEnvironment, self).__init__()

  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    return len(self._envs)

  @property
  def envs(self):
    return self._envs

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def time_step_spec(self):
    return self._time_step_spec

  def _reset(self):
    """Reset all environments and combine the resulting observation.

    Returns:
      Time step with batch dimension.
    """
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(self._envs[0].reset())
    else:
      time_steps = self._pool.map(lambda env: env.reset(), self._envs)
      return stack_time_steps(time_steps)

  def _step(self, actions):
    """Forward a batch of actions to the wrapped environments.

    Args:
      actions: Batched action, possibly nested, to apply to the environment.

    Raises:
      ValueError: Invalid actions.

    Returns:
      Batch of observations, rewards, and done flags.
    """

    if self._num_envs == 1:
      actions = nest_utils.unbatch_nested_array(actions)
      time_steps = self._envs[0].step(actions)
      return nest_utils.batch_nested_array(time_steps)
    else:
      unstacked_actions = unstack_actions(actions)
      if len(unstacked_actions) != self.batch_size:
        raise ValueError(
            "Primary dimension of action items does not match "
            "batch size: %d vs. %d" % (len(unstacked_actions), self.batch_size))
      time_steps = self._pool.map(
          lambda env_action: env_action[0].step(env_action[1]),
          zip(self._envs, unstacked_actions))
      return stack_time_steps(time_steps)

  def close(self):
    """Send close messages to the external process and join them."""
    self._pool.map(lambda env: env.close(), self._envs)
    self._pool.close()
    self._pool.join()


# TODO(b/124447001): Factor these helper functions out into common utils.
def stack_time_steps(time_steps):
  """Given a list of TimeStep, combine to one with a batch dimension."""
  return fast_map_structure(lambda *arrays: np.stack(arrays), *time_steps)


def unstack_actions(batched_actions):
  """Returns a list of actions from potentially nested batch of actions."""
  flattened_actions = tf.nest.flatten(batched_actions)
  unstacked_actions = [
      tf.nest.pack_sequence_as(batched_actions, actions)
      for actions in zip(*flattened_actions)
  ]
  return unstacked_actions


def fast_map_structure(func, *structure):
  """List tf.nest.map_structure, but skipping the slow assert_same_structure."""
  flat_structure = [tf.nest.flatten(s) for s in structure]
  entries = zip(*flat_structure)
  return tf.nest.pack_sequence_as(structure[0], [func(*x) for x in entries])
