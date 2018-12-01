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

"""Base class for drivers that takes steps in an environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


from tf_agents.environments import tf_environment
from tf_agents.policies import tf_policy


@six.add_metaclass(abc.ABCMeta)
class Driver(object):
  """A driver that takes steps in an environment using a TF policy."""

  def __init__(self, env, policy, observers=None):
    """Creates a Driver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after the driver is run.
        Each observer is a callable(TimeStepAction) that returns the input.
        TimeStepAction.time_step is a stacked batch [N+1, batch_size, ...] of
        timesteps and TimeStepAction.action is a stacked batch
        [N, batch_size, ...] of actions in time major form.

    Raises:
      ValueError:
        If env is not a tf_environment.Base or policy is not an instance of
        tf_policy.Base.
    """

    if not isinstance(env, tf_environment.Base):
      raise ValueError('`env` must be an instance of tf_environment.Base.')

    if not isinstance(policy, tf_policy.Base):
      raise ValueError('`policy` must be an instance of tf_policy.Base.')

    self._env = env
    self._policy = policy
    self._observers = observers or []

  @property
  def observers(self):
    return self._observers

  @abc.abstractmethod
  def run(self):
    """Takes steps in the environment and updates observers."""
