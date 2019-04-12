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

"""Python Policies API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six
import tensorflow as tf
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


@six.add_metaclass(abc.ABCMeta)
class Base(object):
  """Abstract base class for Python Policies.

  The `action(time_step, policy_state)` method returns a PolicyStep named tuple
  containing the following nested arrays:
    `action`: The action to be applied on the environment.
    `state`: The state of the policy (E.g. RNN state) to be fed into the next
      call to action.
    `info`: Optional side information such as action log probabilities.

  For stateful policies, e.g. those containing RNNs, an initial policy state can
  be obtained through a call to `get_initial_state()`.

  Example of simple use in Python:

    py_env = PyEnvironment()
    policy = PyPolicy()

    time_step = py_env.reset()
    policy_state = policy.get_initial_state()

    acc_reward = 0
    while not time_step.is_last():
      action_step = policy.action(time_step, policy_state)
      policy_state = action_step.state
      time_step = py_env.step(action_step.action)
      acc_reward += time_step.reward
  """

  # TODO(kbanoop): Expose a batched/batch_size property.
  def __init__(self, time_step_spec, action_spec, policy_state_spec=(),
               info_spec=()):
    """Initialization of Base class.

    Args:
      time_step_spec: A `TimeStep` ArraySpec of the expected time_steps.
        Usually provided by the user to the subclass.
      action_spec: A nest of BoundedArraySpec representing the actions.
        Usually provided by the user to the subclass.
      policy_state_spec: A nest of ArraySpec representing the policy state.
        Provided by the subclass, not directly by the user.
      info_spec: A nest of ArraySpec representing the policy info.
        Provided by the subclass, not directly by the user.
    """
    common.assert_members_are_not_overridden(base_cls=Base, instance=self)
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    # TODO(kbanoop): rename policy_state to state.
    self._policy_state_spec = policy_state_spec
    self._info_spec = info_spec
    self._setup_specs()

  def _setup_specs(self):
    self._policy_step_spec = policy_step.PolicyStep(
        action=self._action_spec, state=self._policy_state_spec,
        info=self._info_spec)
    self._trajectory_spec = trajectory.from_transition(
        self._time_step_spec, self._policy_step_spec, self._time_step_spec)

  def get_initial_state(self, batch_size=None):
    """Returns an initial state usable by the policy.

    Args:
      batch_size: An optional batch size.

    Returns:
      An initial policy state.
    """
    return self._get_initial_state(batch_size)

  def action(self, time_step, policy_state=()):
    """Generates next action given the time_step and policy_state.


    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: An optional previous policy_state.

    Returns:
      A PolicyStep named tuple containing:
        `action`: A nest of action Arrays matching the `action_spec()`.
        `state`: A nest of policy states to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.
    """
    return self._action(time_step, policy_state)

  @property
  def time_step_spec(self):
    """Describes the `TimeStep` np.Arrays expected by `action(time_step)`.

    Returns:
      A `TimeStep` namedtuple with `ArraySpec` objects instead of np.Array,
      which describe the shape, dtype and name of each array expected by
      `action()`.
    """
    return self._time_step_spec

  @property
  def action_spec(self):
    """Describes the ArraySpecs of the np.Array returned by `action()`.

    `action` can be a single np.Array, or a nested dict, list or tuple of
    np.Array.

    Returns:
      A single BoundedArraySpec, or a nested dict, list or tuple of
      `BoundedArraySpec` objects, which describe the shape and
      dtype of each np.Array returned by `action()`.
    """
    return self._action_spec

  @property
  def policy_state_spec(self):
    """Describes the arrays expected by functions with `policy_state` as input.

    Returns:
      A single BoundedArraySpec, or a nested dict, list or tuple of
      `BoundedArraySpec` objects, which describe the shape and
      dtype of each np.Array returned by `action()`.
    """
    return self._policy_state_spec

  @property
  def info_spec(self):
    """Describes the Arrays emitted as info by `action()`.

    Returns:
      A nest of ArraySpec which describe the shape and dtype of each Array
      emitted as `info` by `action()`.
    """
    return self._info_spec

  @property
  def policy_step_spec(self):
    """Describes the output of `action()`.

    Returns:
      A nest of ArraySpec which describe the shape and dtype of each Array
      emitted by `action()`.
    """
    return self._policy_step_spec

  @property
  def trajectory_spec(self):
    """Describes the data collected when using this policy with an environment.

    Returns:
      A `Trajectory` containing all array specs associated with the
      time_step_spec and policy_step_spec of this policy.
    """
    return self._trajectory_spec

  @abc.abstractmethod
  def _action(self, time_step, policy_state):
    """Implementation of `action`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: An Array, or a nested dict, list or tuple of
        Arrays representing the previous policy_state.

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: A nest of action Arrays matching the `action_spec()`.
        `state`: A nest of policy states to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.
    """

  def _get_initial_state(self, batch_size):
    """Default implementation of `get_initial_state`.

    This implementation returns arrays of all zeros matching `batch_size` and
    spec `self.policy_state_spec`.

    Args:
      batch_size: The batch shape.

    Returns:
      A nested object of type `policy_state` containing properly
      initialized Arrays.
    """
    def _zero_array(spec):
      if batch_size is None:
        shape = spec.shape
      else:
        shape = (batch_size,) + spec.shape
      return np.zeros(shape, spec.dtype)

    return tf.nest.map_structure(_zero_array, self._policy_state_spec)
