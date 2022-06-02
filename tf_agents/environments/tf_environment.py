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

"""TensorFlow RL Environment API.

Represents a task to be solved, an environment has to define three methods:
`reset`, `current_time_step` and `step`.

- The reset() method returns current time_step after resetting the environment.
- The current_time_step() method returns current time_step initializing the
environmet if needed. Only needed in graph mode.
- The step(action) method applies the action and returns the new time_step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class TFEnvironment(object):
  """Abstract base class for TF RL environments.

  The `current_time_step()` method returns current `time_step`, resetting the
  environment if necessary.

  The `step(action)` method applies the action and returns the new `time_step`.
  This method will also reset the environment if needed and ignore the action in
  that case.

  The `reset()` method returns `time_step` that results from an environment
  reset and is guaranteed to have step_type=ts.FIRST

  The `reset()` method is only needed for explicit resets. In general, the
  environment will reset automatically when needed, for example, when no
  episode was started or when it reaches a step after the end of the episode
  (i.e. step_type=ts.LAST).

  Example for collecting an episode in eager mode:

    tf_env = TFEnvironment()

    # reset() creates the initial time_step and resets the environment.
    time_step = tf_env.reset()
    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = tf_env.step(action_step.action)

  Example of simple use in graph mode:

    tf_env = TFEnvironment()

    # current_time_step() creates the initial TimeStep.
    time_step = tf_env.current_time_step()
    action_step = policy.action(time_step)
    # Apply the action and return the new TimeStep.
    next_time_step = tf_env.step(action_step.action)

    sess.run([time_step, action_step, next_time_step])

  Example with explicit resets in graph mode:

    reset_op = tf_env.reset()
    time_step = tf_env.current_time_step()
    action_step = policy.action(time_step)
    # Apply the action and return the new TimeStep.
    next_time_step = tf_env.step(action_step.action)

    # The environment will initialize before starting.
    sess.run([time_step, action_step, next_time_step])
    # This will force reset the Environment.
    sess.run(reset_op)
    # This will apply a new action in the environment.
    sess.run([time_step, action_step, next_time_step])

  Example of random actions in graph mode:

    tf_env = TFEnvironment()

    # Action needs to depend on the time_step using control_dependencies.
    time_step = tf_env.current_time_step()
    with tf.control_dependencies([time_step.step_type]):
      action = tensor_spec.sample_bounded_spec(tf_env.action_spec())
    next_time_step = tf_env.step(action)

    sess.run([time_step, action, next_time_step])

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
    """Initializes the environment.

    Meant to be called by subclass constructors.

    Args:
      time_step_spec: A `TimeStep` namedtuple containing `TensorSpec`s
        defining the Tensors returned by
        `step()` (step_type, reward, discount, and observation).
      action_spec: A nest of BoundedTensorSpec representing the actions of the
        environment.
      batch_size: The batch size expected for the actions and observations.
    """

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._batch_size = batch_size

  def time_step_spec(self):
    """Describes the `TimeStep` specs of Tensors returned by `step()`.

    Returns:
      A `TimeStep` namedtuple containing `TensorSpec` objects defining the
      Tensors returned by `step()`, i.e.
      (step_type, reward, discount, observation).
    """
    return self._time_step_spec

  def action_spec(self):
    """Describes the specs of the Tensors expected by `step(action)`.

    `action` can be a single Tensor, or a nested dict, list or tuple of
    Tensors.

    Returns:
      An single `TensorSpec`, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the shape and
      dtype of each Tensor expected by `step()`.
    """
    return self._action_spec

  def observation_spec(self):
    """Defines the `TensorSpec` of observations provided by the environment.

    Returns:
      A `TensorSpec`, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the observation.
    """
    return self.time_step_spec().observation

  def reward_spec(self):
    """Defines the `TensorSpec` of rewards provided by the environment.

    Returns:
      A `TensorSpec`, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the reward.
    """
    return self.time_step_spec().reward

  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    return self._batch_size

  def current_time_step(self):
    """Returns the current `TimeStep`.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this time_step.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """
    return self._current_time_step()

  def reset(self):
    """Resets the environment and returns the current time_step.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this time_step.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """
    return self._reset()

  def step(self, action):
    """Steps the environment according to the action.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` should reset the environment (note that
    it is expected that whoever defines this method, calls reset in this case),
    start a new sequence and `action` will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset()` has not been called. In this case
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
        reward: Reward at this time_step.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
    """
    return self._step(action)

  def render(self):
    """Renders a frame from the environment.

    Raises:
      NotImplementedError: If the environment does not support rendering.
    """
    raise NotImplementedError('No rendering support.')

  @abc.abstractmethod
  def _current_time_step(self):
    """Returns the current `TimeStep`."""

  @abc.abstractmethod
  def _reset(self):
    """Resets the environment and returns the current time_step."""

  @abc.abstractmethod
  def _step(self, action):
    """Steps the environment according to the action."""
