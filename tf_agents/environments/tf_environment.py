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

"""TensorFlow RL Environment API.

Represents a task to be solved, an environment has to define three methods:
`reset()`, `current_time_step()` and `step()`.

The reset() method returns current timestep after resetting the environment.
The current_time_step() method returns current timestep initializing the
environmet if needed. Only needed in Graph-Mode.
The step(action) method applies the action and returns the new timestep.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Base(object):
  """Abstract base class for TF RL environments.

  The `current_time_step()` method returns current `time_step`, resetting the
  environment if necessary.

  The `step(action)` method applies the action and returns the new `time_step`.
  This method will also reset the environment if needed and ignore the action in
  that case.

  The `reset()` method returns `time_step` that results from an environment
  reset and is guaranteed to have step_type=ts.FIRST

  The `reset()` is only needed for explicit resets, in general the Environment
  will reset automatically when needed, for example, when no episode was started
  or when stepped after the end of the episode was reached
  (i.e. step_type=ts.LAST).

  Example for collecting an episode in Eager mode:

    tf_env = TFEnvironment()

    # reset() creates the initial time_step and resets the Environment.
    time_step = tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = tf_env.step(action_step.action)

  Example of simple use in Graph Mode:

    tf_env = TFEnvironment()

    # current_time_step() creates the initial TimeStep.
    time_step = tf_env.current_time_step()
    action_step = policy.action(time_step)
    # It applies the action and returns the new TimeStep.
    next_time_step = tf_env.step(action_step.action)

    sess.run([time_step, action_step, next_time_step])

  Example with explicit resets in Graph-Mode:

    reset_op = tf_env.reset()
    time_step = tf_env.current_time_step()
    action_step = policy.action(time_step)
    # It applies the action and returns the new TimeStep.
    next_time_step = tf_env.step(action_step.action)

    # The Environment will initialize before starting.
    sess.run([time_step, action_step, next_time_step])
    # This will force reset the Environment.
    sess.run(reset_op)
    # This will apply a new action in the Environment.
    sess.run([time_step, action_step, next_time_step])


  Example of random actions in Graph mode:

    tf_env = TFEnvironment()

    # The action needs to depend on time_step using control_dependencies.
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tensor_spec.sample_bounded_spec(tf_env.action_spec())
    next_time_step = tf_env.step(action)

    sess.run([timestep, action, next_timestep])

  Example of collecting full episodes with a while_loop:

    tf_env = TFEnvironment()

    # reset() creates the initial time_step
    time_step = tf_env.reset()
    c = lambda t: tf.logical_not(t.is_last())
    body = lambda t: [tf_env.step(t.observation)]

    final_time_step = tf.while_loop(c, body, [time_step])

    sess.run(final_time_step)

  """

  def __init__(self, time_step_spec=None, action_spec=None, batch_size=1):
    """Meant to be called by subclass constructors.

    Args:
      time_step_spec: A `TimeStep` namedtuple containing `TensorSpec`s
        defining the tensors returned by
        `step()` (step_type, reward, discount, and observation).
      action_spec: A nest of BoundedTensorSpec representing the actions of the
        environment.
      batch_size: The batch size expected for the actions and observations.
    """

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._batch_size = batch_size

  @abc.abstractmethod
  def current_time_step(self):
    """Returns the current `TimeStep`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

  @abc.abstractmethod
  def reset(self):
    """Resets the environment and returns the current_time_step.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

  @abc.abstractmethod
  def step(self, action):
    """Steps the environment according to the action.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `action`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. In this case
    `action` will be ignored.

    Expected sequences look like:

      time_step -> action -> next_time_step

    The action should depend on the previous time_step for correctness.

    Args:
      action: A Tensor, or a nested dict, list or tuple of Tensors
        corresponding to `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """

  def render(self):
    """Renders a frame from the environment.

    Raises:
      NotImplementedError: If the environment does not support rendering.
    """
    raise NotImplementedError('No rendering support.')

  def time_step_spec(self):
    """Describes the `TimeStep` tensors returned by `step()`.

    Returns:
      A `TimeStep` namedtuple containing `TensorSpec`s defining the tensors
      returned by `step()` (step_type, reward, discount, and observation).
    """
    return self._time_step_spec

  def action_spec(self):
    """Describes the TensorSpecs of the Tensors expected by `step(action)`.

    `action` can be a single Tensor, or a nested dict, list or tuple of
    Tensors.

    Returns:
      An single TensorSpec, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the shape and
      dtype of each Tensor expected by `step()`.
    """
    return self._action_spec

  def observation_spec(self):
    """Defines the TensorSpec of observations provided by the environment.

    Returns:
      Same structure as returned by `time_step_spec().observation`.
    """
    return self.time_step_spec().observation

  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    return self._batch_size
