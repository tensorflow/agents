# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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

# TODO(78251218): Should we reference deepmind's dmsuite package?
Follows the same API as Deepmind's Environment API as seen in:
  https://github.com/deepmind/dm_control

The reset() method returns timestep, step_state and reset_op
The step(action, step_state) method returns timestep and step_state
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Base(object):
  """Abstract base class for TF RL environments.

  The `reset()` method returns `timestep, step_state, reset_op`
  The `step(action, step_state)` method returns the new `timestep, step_state`

  The `reset_op` is only needed for explicit resets, in general the Environment
  will reset automatically.

  Example of simple use:

    tf_env = TFEnvironment()

    # reset() creates the initial timestep and step_state, plus a reset_op
    timestep, step_state, _ = tf_env.reset()
    # TODO(kbanoop): Fix docs using policy to be explicit about policy_action.
    action = policy.action(timestep)
    # It applies the action and returns the new TimeStep.
    next_timestep, _ = tf_env.step(action, step_state)

    sess.run([timestep, action, next_timestep])

  Example with multiple steps:

    tf_env = TFEnvironment()

    # reset() creates the initial timestep and step_state, plus a reset_op
    timestep, step_state, reset_op = tf_env.reset()
    n_step = [timestep]
    for i in range(n):
      action = policy.action(timestep)
      n_step.append(action)
      timestep, step_state = tf_env.step(action, step_state)
      n_step.append(timestep)

    # n_step contains [timestep, action, timestep, action, ...]
    sess.run(n_step)

  Example with explicit resets:

    timestep, step_state, reset_op = tf_env.reset()
    action = policy.action(timestep)
    # It applies the action and returns the new TimeStep.
    next_timestep, _ = tf_env.step(action, step_state)

    # The Environment would be reset before starting.
    sess.run([timestep, action, next_timestep])
    # Will force reset the Environment.
    sess.run(reset_op)
    sess.run([timestep, action, next_timestep])


  Example of random actions:

    tf_env = TFEnvironment()

    # reset() creates the initial timestep and step_state, plus a reset_op
    timestep, step_state, reset_op = tf_env.reset()
    # The action doesn't need to depend on timestep because the tf_env add the
    # needed control_dependencies.
    action = tf.random_normal()
    next_timestep, _ = tf_env.step(action, step_state)

    sess.run(reset_op)
    sess.run([timestep, action, next_timestep])

  Example of collecting an episode with while_loop:

    tf_env = TFEnvironment()

    # reset() creates the initial timestep and step_state, plus a reset_op
    timestep, step_state, reset_op = tf_env.reset()
    c = lambda t, s: t.is_last()
    body = lambda t, s: tf_env.step(policy.action(t), s)

    episode = tf.while_loop(c, body, (timestep, step_state))

    sess.run(episode)
  """

  @abc.abstractmethod
  def current_time_step(self, step_state=None):
    """Returns a `TimeStep` and a step_state.

    Args:
      step_state: An optional initial Tensor, or a nested dict, list or tuple of
        Tensors representing the initial step_state. It is used to chain
        dependencies across steps, but can be used to also pass specific step
        states across steps. If it is None a `tf.constant(0)` Tensor would be
        used.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
      A step_state initial Tensor, or a nested dict, list or tuple of Tensors,
        representing the step state.
    """

  @abc.abstractmethod
  def reset(self, step_state=None):
    """Returns a `TimeStep`, a step_state and a reset_op of the environment.

    Args:
      step_state: An optional initial Tensor, or a nested dict, list or tuple of
        Tensors representing the initial step_state. It is used to chain
        dependencies across steps, but can be used to also pass specific step
        states across steps. If it is None a `tf.constant(0)` Tensor would be
        used.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
      A step_state initial Tensor, or a nested dict, list or tuple of Tensors,
        representing the step state.
      A reset_op.
    """

  @abc.abstractmethod
  def step(self, action, step_state):
    """Updates the environment according to the action.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `action`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. Again, in this case
    `action` will be ignored.

    This method should include a control_dependency on step_state before
    applying the action, and then create a new step_state with a
    control_dependency on the timestep produced by the action.

      step_state -> action -> time_step -> new_step_state

    This chain of control dependencies make sure that previous steps are
    executed before this one, whether the action depends on the previous
    time_step or not.

    Args:
      action: A Tensor, or a nested dict, list or tuple of Tensors
        corresponding to `action_spec()`.
      step_state: A Tensor, or a nested dict, list or tuple of
        Tensors representing the previous step_state. It is used to chain
        dependencies across steps, but can be used to also pass specific step
        states across steps.

    Returns:
      A `TimeStep` namedtuple containing:
        step_type: A `StepType` value.
        reward: Reward at this timestep.
        discount: A discount in the range [0, 1].
        observation: A Tensor, or a nested dict, list or tuple of Tensors
          corresponding to `observation_spec()`.
      A step_state Tensor, or a nested dict, list or tuple of Tensors,
        representing the new step state.

    """

  def render(self):
    """Renders a frame from the environment.

    Raises:
      NotImplementedError: If the environment does not support rendering.
    """
    raise NotImplementedError('No rendering support.')

  @abc.abstractmethod
  def time_step_spec(self):
    """Describes the `TimeStep` tensors returned by `step()`.

    Returns:
      A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
      which describe the shape, dtype and name of each tensor returned by
      `step()`.
    """

  @abc.abstractmethod
  def action_spec(self):
    """Describes the TensorSpecs of the Tensors expected by `step(action)`.

    `action` can be a single Tensor, or a nested dict, list or tuple of
    Tensors.

    Returns:
      An single TensorSpec, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the shape and
      dtype of each Tensor expected by `step()`.
    """

  def observation_spec(self):
    """Defines the TensorSpec of observations provided by the environment.

    Returns:
      Same structure as returned by `time_step_spec().observation` but with
      TensorSpec objects.
    """
    return self.time_step_spec().observation

  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    raise NotImplementedError('batch_size is not implemented')
