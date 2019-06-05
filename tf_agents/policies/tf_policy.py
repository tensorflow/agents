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

"""TensorFlow Policies API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils

tfd = tfp.distributions


class Base(tf.Module):
  """Abstract base class for TF Policies.

  Example of simple use in TF:

    tf_env = SomeTFEnvironment()
    policy = SomeTFPolicy()

    time_step, step_state, reset_env = tf_env.reset()
    policy_state = policy.get_initial_state(batch_size=tf_env.batch_size)
    action_step = policy.action(time_step, policy_state)
    next_time_step, _ = env.step(action_step.action, step_state)

    sess.run([time_step, action, next_time_step])


  Example of using the same policy for several steps:

    tf_env = SomeTFEnvironment()
    policy = SomeTFPolicy()

    exp_policy = SomeTFPolicy()
    update_policy = exp_policy.update(policy)
    policy_state = exp_policy.get_initial_state(tf_env.batch_size)

    time_step, step_state, _ = tf_env.reset()
    action_step, policy_state, _ = exp_policy.action(time_step, policy_state)
    next_time_step, step_state = env.step(action_step.action, step_state)

    for j in range(num_episodes):
      sess.run(update_policy)
      for i in range(num_steps):
        sess.run([time_step, action_step, next_time_step])


  Example with multiple steps:

    tf_env = SomeTFEnvironment()
    policy = SomeTFPolicy()

    # reset() creates the initial time_step and step_state, plus a reset_op
    time_step, step_state, reset_op = tf_env.reset()
    policy_state = policy.get_initial_state(tf_env.batch_size)
    n_step = [time_step]
    for i in range(n):
      action_step = policy.action(time_step, policy_state)
      policy_state = action_step.state
      n_step.append(action_step)
      time_step, step_state = tf_env.step(action_step.action, step_state)
      n_step.append(time_step)

    # n_step contains [time_step, action, time_step, action, ...]
    sess.run(n_step)

  Example with explicit resets:

    tf_env = SomeTFEnvironment()
    policy = SomeTFPolicy()
    policy_state = policy.get_initial_state(tf_env.batch_size)

    time_step, step_state, reset_env = tf_env.reset()
    action_step = policy.action(time_step, policy_state)
    # It applies the action and returns the new TimeStep.
    next_time_step, _ = tf_env.step(action_step.action, step_state)
    next_action_step = policy.action(next_time_step, policy_state)

    # The Environment and the Policy would be reset before starting.
    sess.run([time_step, action_step, next_time_step, next_action_step])
    # Will force reset the Environment and the Policy.
    sess.run([reset_env])
    sess.run([time_step, action_step, next_time_step, next_action_step])
  """

  # TODO(b/127327645) Remove this attribute.
  # This attribute allows subclasses to back out of automatic tf.function
  # attribute inside TF1 (for autodeps).
  _enable_functions = True

  def __init__(self,
               time_step_spec,
               action_spec,
               policy_state_spec=(),
               info_spec=(),
               clip=True,
               emit_log_probability=False,
               automatic_state_reset=True,
               name=None):
    """Initialization of Base class.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps. Usually
        provided by the user to the subclass.
      action_spec: A nest of BoundedTensorSpec representing the actions. Usually
        provided by the user to the subclass.
      policy_state_spec: A nest of TensorSpec representing the policy_state.
        Provided by the subclass, not directly by the user.
      info_spec: A nest of TensorSpec representing the policy info. Provided by
        the subclass, not directly by the user.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      emit_log_probability: Emit log-probabilities of actions, if supported. If
        True, policy_step.info will have CommonFields.LOG_PROBABILITY set.
        Please consult utility methods provided in policy_step for setting and
        retrieving these. When working with custom policies, either provide a
        dictionary info_spec or a namedtuple with the field 'log_probability'.
      automatic_state_reset:  If `True`, then `get_initial_policy_state` is used
        to clear state in `action()` and `distribution()` for for time steps
        where `time_step.is_first()`.
      name: A name for this module. Defaults to the class name.
    """
    super(Base, self).__init__(name=name)
    common.assert_members_are_not_overridden(base_cls=Base, instance=self)
    if not isinstance(time_step_spec, ts.TimeStep):
      raise ValueError(
          'The `time_step_spec` must be an instance of `TimeStep`, but is `{}`.'
          .format(type(time_step_spec)))

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._policy_state_spec = policy_state_spec
    self._emit_log_probability = emit_log_probability
    if emit_log_probability:
      log_probability_spec = tensor_spec.BoundedTensorSpec(
          shape=(), dtype=tf.float32, maximum=0, minimum=-float('inf'),
          name='log_probability')
      log_probability_spec = tf.nest.map_structure(
          lambda _: log_probability_spec, action_spec)
      info_spec = policy_step.set_log_probability(info_spec,
                                                  log_probability_spec)

    self._info_spec = info_spec
    self._setup_specs()
    self._clip = clip
    self._action_fn = common.function_in_tf1()(self._action)
    self._automatic_state_reset = automatic_state_reset

  def _setup_specs(self):
    self._policy_step_spec = policy_step.PolicyStep(
        action=self._action_spec,
        state=self._policy_state_spec,
        info=self._info_spec)
    self._trajectory_spec = trajectory.from_transition(self._time_step_spec,
                                                       self._policy_step_spec,
                                                       self._time_step_spec)

  def variables(self):
    """Returns the list of Variables that belong to the policy."""
    return self._variables()

  # TODO(kbanoop): Consider get_initial_state(inputs=None, batch_size=None).
  def get_initial_state(self, batch_size):
    """Returns an initial state usable by the policy.

    Args:
      batch_size: The batch shape.

    Returns:
      A nested object of type `policy_state` containing properly
      initialized Tensors.
    """
    return self._get_initial_state(batch_size=batch_size)

  def _maybe_reset_state(self, time_step, policy_state):
    if policy_state is ():  # pylint: disable=literal-comparison
      return policy_state

    batch_size = tf.compat.dimension_value(time_step.discount.shape[0])
    if batch_size is None:
      batch_size = tf.shape(time_step.discount)[0]

    # Make sure we call this with a kwarg as it may be wrapped in tf.function
    # which would expect a tensor if it was not a kwarg.
    zero_state = self.get_initial_state(batch_size=batch_size)
    condition = time_step.is_first()
    # When experience is a sequence we only reset automatically for the first
    # time_step in the sequence as we can't easily generalize how the policy is
    # unrolled over the sequence.
    if nest_utils.get_outer_rank(time_step, self._time_step_spec) > 1:
      condition = time_step.is_first()[:, 0, ...]
    return nest_utils.where(condition, zero_state, policy_state)

  def action(self, time_step, policy_state=(), seed=None):
    """Generates next action given the time_step and policy_state.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seed: Seed to use if action performs sampling (optional).

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: An action Tensor matching the `action_spec()`.
        `state`: A policy state tensor to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.

    Raises:
      RuntimeError: If subclass __init__ didn't call super().__init__.
    """
    if self._enable_functions and getattr(self, '_action_fn', None) is None:
      raise RuntimeError(
          'Cannot find _action_fn.  Did %s.__init__ call super?' %
          type(self).__name__)
    if self._enable_functions:
      action_fn = self._action_fn
    else:
      action_fn = self._action

    tf.nest.assert_same_structure(time_step, self._time_step_spec)
    tf.nest.assert_same_structure(policy_state, self._policy_state_spec)

    if self._automatic_state_reset:
      policy_state = self._maybe_reset_state(time_step, policy_state)
    step = action_fn(time_step=time_step, policy_state=policy_state, seed=seed)

    def clip_action(action, action_spec):
      if isinstance(action_spec, tensor_spec.BoundedTensorSpec):
        return common.clip_to_spec(action, action_spec)
      return action

    if self._clip:
      clipped_actions = tf.nest.map_structure(clip_action, step.action,
                                              self._action_spec)
      step = step._replace(action=clipped_actions)

    tf.nest.assert_same_structure(step, self._policy_step_spec)

    def compare_to_spec(value, spec):
      return value.dtype.is_compatible_with(spec.dtype)

    compatibility = tf.nest.flatten(
        tf.nest.map_structure(compare_to_spec, step.action, self.action_spec))

    if not all(compatibility):
      get_dtype = lambda x: x.dtype
      action_dtypes = tf.nest.map_structure(get_dtype, step.action)
      spec_dtypes = tf.nest.map_structure(get_dtype, self.action_spec)

      raise TypeError('Policy produced an action with a dtype that doesn\'t '
                      'match its action_spec. Got action: %s with '
                      'action_spec: %s' % (action_dtypes, spec_dtypes))

    return step

  def distribution(self, time_step, policy_state=()):
    """Generates the distribution over next actions given the time_step.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.

    Returns:
      A `PolicyStep` named tuple containing:

        `action`: A tf.distribution capturing the distribution of next actions.
        `state`: A policy state tensor for the next call to distribution.
        `info`: Optional side information such as action log probabilities.
    """
    tf.nest.assert_same_structure(time_step, self._time_step_spec)
    tf.nest.assert_same_structure(policy_state, self._policy_state_spec)
    if self._automatic_state_reset:
      policy_state = self._maybe_reset_state(time_step, policy_state)
    step = self._distribution(time_step=time_step, policy_state=policy_state)
    if self.emit_log_probability:
      # This here is set only for compatibility with info_spec in constructor.
      info = policy_step.set_log_probability(
          step.info,
          tf.nest.map_structure(
              lambda _: tf.constant(0., dtype=tf.float32),
              policy_step.get_log_probability(self._info_spec)))
      step = step._replace(info=info)
    tf.nest.assert_same_structure(step, self._policy_step_spec)
    return step

  def update(self, policy, tau=1.0, sort_variables_by_name=False):
    """Update the current policy with another policy.

    This would include copying the variables from the other policy.

    Args:
      policy: Another policy it can update from.
      tau: A float scalar in [0, 1]. When tau is 1.0 (default), we do a hard
        update.
      sort_variables_by_name: A bool, when True would sort the variables by name
        before doing the update.

    Returns:
      An TF op to do the update.
    """
    if self.variables():
      return common.soft_variables_update(
          policy.variables(),
          self.variables(),
          tau=tau,
          sort_variables_by_name=sort_variables_by_name)
    else:
      return tf.no_op()

  @property
  def emit_log_probability(self):
    """Whether this policy instance emits log probabilities or not."""
    return self._emit_log_probability

  @property
  def time_step_spec(self):
    """Describes the `TimeStep` tensors returned by `step()`.

    Returns:
      A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
      which describe the shape, dtype and name of each tensor returned by
      `step()`.
    """
    return self._time_step_spec

  @property
  def action_spec(self):
    """Describes the TensorSpecs of the Tensors expected by `step(action)`.

    `action` can be a single Tensor, or a nested dict, list or tuple of
    Tensors.

    Returns:
      An single BoundedTensorSpec, or a nested dict, list or tuple of
      `BoundedTensorSpec` objects, which describe the shape and
      dtype of each Tensor expected by `step()`.
    """
    return self._action_spec

  @property
  def policy_state_spec(self):
    """Describes the Tensors expected by `step(_, policy_state)`.

    `policy_state` can be an empty tuple, a single Tensor, or a nested dict,
    list or tuple of Tensors.

    Returns:
      An single TensorSpec, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the shape and
      dtype of each Tensor expected by `step(_, policy_state)`.
    """
    return self._policy_state_spec

  @property
  def info_spec(self):
    """Describes the Tensors emitted as info by `action` and `distribution`.

    `info` can be an empty tuple, a single Tensor, or a nested dict,
    list or tuple of Tensors.

    Returns:
      An single TensorSpec, or a nested dict, list or tuple of
      `TensorSpec` objects, which describe the shape and
      dtype of each Tensor expected by `step(_, policy_state)`.
    """
    return self._info_spec

  @property
  def policy_step_spec(self):
    """Describes the output of `action()`.

    Returns:
      A nest of TensorSpec which describe the shape and dtype of each Tensor
      emitted by `action()`.
    """
    return self._policy_step_spec

  # TODO(kbanoop, ebrevdo): Should this be collect_data_spec to mirror agents?
  @property
  def trajectory_spec(self):
    """Describes the Tensors written when using this policy with an environment.

    Returns:
      A `Trajectory` containing all tensor specs associated with the
      observation_spec, action_spec, policy_state_spec, and info_spec of
      this policy.
    """
    return self._trajectory_spec

  # Subclasses MAY optionally override _action.
  def _action(self, time_step, policy_state, seed):
    """Implementation of `action`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seed: Seed to use if action performs sampling (optional).

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: An action Tensor matching the `action_spec()`.
        `state`: A policy state tensor to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.
    """
    seed_stream = tfd.SeedStream(seed=seed, salt='ppo_policy')
    distribution_step = self._distribution(time_step, policy_state)
    actions = tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                    distribution_step.action)
    info = distribution_step.info
    if self.emit_log_probability:
      try:
        log_probability = tf.nest.map_structure(lambda a, d: d.log_prob(a),
                                                actions,
                                                distribution_step.action)
        info = policy_step.set_log_probability(info, log_probability)
      except:
        raise TypeError('%s does not support emitting log-probabilities.' %
                        type(self).__name__)

    return distribution_step._replace(action=actions, info=info)

  ## Subclasses MUST implement these.

  @abc.abstractmethod
  def _distribution(self, time_step, policy_state):
    """Implementation of `distribution`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: A (optionally nested) of tfp.distribution.Distribution
          capturing the distribution of next actions.
        `state`: A policy state tensor for the next call to distribution.
        `info`: Optional side information such as action log probabilities.
    """
    pass

  @abc.abstractmethod
  def _variables(self):
    """Returns an iterable of `tf.Variable` objects used by this policy."""
    pass

  # Subclasses MAY optionally overwrite _get_initial_state.
  def _get_initial_state(self, batch_size):
    """Default implementation of `get_initial_state`.

    This implementation returns tensors of all zeros matching `batch_size` and
    spec `self.policy_state_spec`.

    Args:
      batch_size: The batch shape.

    Returns:
      A nested object of type `policy_state` containing properly
      initialized Tensors.
    """

    def _zero_tensor(spec):
      if batch_size is None:
        shape = spec.shape
      else:
        spec_shape = tf.convert_to_tensor(value=spec.shape, dtype=tf.int32)
        shape = tf.concat(([batch_size], spec_shape), axis=0)
      dtype = spec.dtype
      return tf.zeros(shape, dtype)

    return tf.nest.map_structure(_zero_tensor, self._policy_state_spec)
