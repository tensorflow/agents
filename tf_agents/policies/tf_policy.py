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

"""TensorFlow Policies API."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
from typing import Optional, Text, Sequence

import six
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.distributions import reparameterized_sampling
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


tfd = tfp.distributions


@six.add_metaclass(abc.ABCMeta)
class TFPolicy(tf.Module):
  """Abstract base class for TF Policies.

  The Policy represents a mapping from `time_steps` recieved from the
  environment to `actions` that can be applied to the environment.

  Agents expose two policies. A `policy` meant for deployment and evaluation,
  and a `collect_policy` for collecting data from the environment. The
  `collect_policy` is usually stochastic for exploring the environment better
  and may log auxilliary information such as log probabilities required for
  training as well. `Policy` objects can also be created directly by the users
  without using an `Agent`.

  The main methods of TFPolicy are:

  * `action`: Maps a `time_step` from the environment to an action.
  * `distribution`: Maps a `time_step` to a distribution over actions.
  * `get_initial_state`: Generates the initial state for stateful policies, e.g.
      RNN/LSTM policies.

  Example usage:

  ```
  env = SomeTFEnvironment()
  policy = TFRandomPolicy(env.time_step_spec(), env.action_spec())
  # Or policy = agent.policy or agent.collect_policy

  policy_state = policy.get_initial_state(env.batch_size)
  time_step = env.reset()

  while not time_step.is_last():
    policy_step = policy.action(time_step, policy_state)
    time_step = env.step(policy_step.action)

    policy_state = policy_step.state
    # policy_step.info may contain side info for logging, such as action log
    # probabilities.
  ```

  Policies can be saved to disk as SavedModels (see policy_saver.py and
  policy_loader.py) or as TF Checkpoints.

  A `PyTFEagerPolicy` can be used to wrap a `TFPolicy` so that it works with
  `PyEnvironment`s.


  **NOTE**: For API consistency, subclasses are not allowed to override public
  methods of `TFPolicy` class. Instead, they may implement the protected methods
  including `_get_initial_state`, `_action`, and `_distribution`. This
  public-calls-private convention allowed this base class to do things like
  properly add `spec` and shape checks, which provide users an easier experience
  when debugging their environments and networks.

  For researchers, and those developing new Policies, the `TFPolicy` base class
  constructor also accept a `validate_args` parameter.  If `False`, this
  disables all spec structure, dtype, and shape checks in the public methods of
  these classes.  It allows algorithm developers to iterate and try different
  input and output structures without worrying about overly restrictive
  requirements, or input and output states being in a certain format.  However,
  *disabling argument validation* can make it very hard to identify structural
  input or algorithmic errors; and should not be done for final, or
  production-ready, Policies.  In addition to having implementations that may
  disagree with specs, this mean that the resulting Policy may no longer
  interact well with other parts of TF-Agents.  Examples include impedance
  mismatches with Actor/Learner APIs, replay buffers, and the model export
  functionality in `PolicySaver.
  """

  # TODO(b/127327645) Remove this attribute.
  # This attribute allows subclasses to back out of automatic tf.function
  # attribute inside TF1 (for autodeps).
  _enable_functions = True

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      policy_state_spec: types.NestedTensorSpec = (),
      info_spec: types.NestedTensorSpec = (),
      clip: bool = True,
      emit_log_probability: bool = False,
      automatic_state_reset: bool = True,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      validate_args: bool = True,
      name: Optional[Text] = None):
    """Initialization of TFPolicy class.

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
      observation_and_action_constraint_splitter: A function used to process
        observations with action constraints. These constraints can indicate,
        for example, a mask of valid/invalid actions for a given state of the
        environment. The function takes in a full observation and returns a
        tuple consisting of 1) the part of the observation intended as input to
        the network and 2) the constraint. An example
        `observation_and_action_constraint_splitter` could be as simple as: ```
        def observation_and_action_constraint_splitter(observation): return
          observation['network_input'], observation['constraint'] ```
        *Note*: when using `observation_and_action_constraint_splitter`, make
          sure the provided `q_network` is compatible with the network-specific
          half of the output of the
          `observation_and_action_constraint_splitter`. In particular,
          `observation_and_action_constraint_splitter` will be called on the
          observation before passing to the network. If
          `observation_and_action_constraint_splitter` is None, action
          constraints are not applied.
      validate_args: Python bool.  Whether to verify inputs to, and outputs of,
        functions like `action` and `distribution` against spec structures,
        dtypes, and shapes.

        Research code may prefer to set this value to `False` to allow iterating
        on input and output structures without being hamstrung by overly
        rigid checking (at the cost of harder-to-debug errors).

        See also `TFAgent.validate_args`.
      name: A name for this module. Defaults to the class name.
    """
    super(TFPolicy, self).__init__(name=name)
    common.check_tf1_allowed()
    common.tf_agents_gauge.get_cell('TFAPolicy').set(True)
    common.assert_members_are_not_overridden(base_cls=TFPolicy, instance=self)
    if not isinstance(time_step_spec, ts.TimeStep):
      raise ValueError(
          'The `time_step_spec` must be an instance of `TimeStep`, but is `{}`.'
          .format(type(time_step_spec)))

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._policy_state_spec = policy_state_spec
    self._emit_log_probability = emit_log_probability
    self._validate_args = validate_args

    if emit_log_probability:
      log_probability_spec = tensor_spec.BoundedTensorSpec(
          shape=(),
          dtype=tf.float32,
          maximum=0,
          minimum=-float('inf'),
          name='log_probability')
      log_probability_spec = tf.nest.map_structure(
          lambda _: log_probability_spec, action_spec)
      info_spec = policy_step.set_log_probability(info_spec,
                                                  log_probability_spec)  # pytype: disable=wrong-arg-types

    self._info_spec = info_spec
    self._setup_specs()
    self._clip = clip
    self._action_fn = common.function_in_tf1()(self._action)
    self._automatic_state_reset = automatic_state_reset
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)

  def _setup_specs(self):
    self._policy_step_spec = policy_step.PolicyStep(
        action=self._action_spec,
        state=self._policy_state_spec,
        info=self._info_spec)
    self._trajectory_spec = trajectory.from_transition(self._time_step_spec,
                                                       self._policy_step_spec,
                                                       self._time_step_spec)

  def variables(self) -> Sequence[tf.Variable]:
    """Returns the list of Variables that belong to the policy."""
    # Ignore self._variables() in favor of using tf.Module's tracking.
    return super(TFPolicy, self).variables

  @property
  def observation_and_action_constraint_splitter(self) -> types.Splitter:
    return self._observation_and_action_constraint_splitter

  @property
  def validate_args(self) -> bool:
    """Whether `action` & `distribution` validate input and output args."""
    return self._validate_args

  def get_initial_state(self,
                        batch_size: Optional[types.Int]) -> types.NestedTensor:
    """Returns an initial state usable by the policy.

    Args:
      batch_size: Tensor or constant: size of the batch dimension. Can be None
        in which case no dimensions gets added.

    Returns:
      A nested object of type `policy_state` containing properly
      initialized Tensors.
    """
    return self._get_initial_state(batch_size)

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

  def action(self,
             time_step: ts.TimeStep,
             policy_state: types.NestedTensor = (),
             seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
    """Generates next action given the time_step and policy_state.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seed: Seed to use if action performs sampling (optional).

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: An action Tensor matching the `action_spec`.
        `state`: A policy state tensor to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.

    Raises:
      RuntimeError: If subclass __init__ didn't call super().__init__.
      ValueError or TypeError: If `validate_args is True` and inputs or
        outputs do not match `time_step_spec`, `policy_state_spec`,
        or `policy_step_spec`.
    """
    if self._enable_functions and getattr(self, '_action_fn', None) is None:
      raise RuntimeError(
          'Cannot find _action_fn.  Did %s.__init__ call super?' %
          type(self).__name__)
    if self._enable_functions:
      action_fn = self._action_fn
    else:
      action_fn = self._action

    if self._validate_args:
      time_step = nest_utils.prune_extra_keys(self._time_step_spec, time_step)
      policy_state = nest_utils.prune_extra_keys(
          self._policy_state_spec, policy_state)
      nest_utils.assert_same_structure(
          time_step,
          self._time_step_spec,
          message='time_step and time_step_spec structures do not match')
      # TODO(b/158804957): Use literal comparison because in some strange cases
      # (tf.function? autograph?) the expression "x not in (None, (), [])" gets
      # converted to a tensor.
      if not (policy_state is None or policy_state is () or policy_state is []):  # pylint: disable=literal-comparison
        nest_utils.assert_same_structure(
            policy_state,
            self._policy_state_spec,
            message=('policy_state and policy_state_spec '
                     'structures do not match'))

    if self._automatic_state_reset:
      policy_state = self._maybe_reset_state(time_step, policy_state)
    step = action_fn(time_step=time_step, policy_state=policy_state, seed=seed)

    def clip_action(action, action_spec):
      if isinstance(action_spec, tensor_spec.BoundedTensorSpec):
        return common.clip_to_spec(action, action_spec)
      return action

    if self._validate_args:
      nest_utils.assert_same_structure(
          step.action, self._action_spec,
          message='action and action_spec structures do not match')

    if self._clip:
      clipped_actions = tf.nest.map_structure(clip_action,
                                              step.action,
                                              self._action_spec)
      step = step._replace(action=clipped_actions)

    if self._validate_args:
      nest_utils.assert_same_structure(
          step,
          self._policy_step_spec,
          message='action output and policy_step_spec structures do not match')

      def compare_to_spec(value, spec):
        return value.dtype.is_compatible_with(spec.dtype)

      compatibility = [
          compare_to_spec(v, s) for (v, s)
          in zip(tf.nest.flatten(step.action),
                 tf.nest.flatten(self.action_spec))]

      if not all(compatibility):
        get_dtype = lambda x: x.dtype
        action_dtypes = tf.nest.map_structure(get_dtype, step.action)
        spec_dtypes = tf.nest.map_structure(get_dtype, self.action_spec)

        raise TypeError('Policy produced an action with a dtype that doesn\'t '
                        'match its action_spec. Got action:\n  %s\n with '
                        'action_spec:\n  %s' % (action_dtypes, spec_dtypes))

    return step

  def distribution(
      self, time_step: ts.TimeStep, policy_state: types.NestedTensor = ()
  ) -> policy_step.PolicyStep:
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

    Raises:
      ValueError or TypeError: If `validate_args is True` and inputs or
        outputs do not match `time_step_spec`, `policy_state_spec`,
        or `policy_step_spec`.
    """
    if self._validate_args:
      time_step = nest_utils.prune_extra_keys(self._time_step_spec, time_step)
      policy_state = nest_utils.prune_extra_keys(
          self._policy_state_spec, policy_state)
      nest_utils.assert_same_structure(
          time_step,
          self._time_step_spec,
          message='time_step and time_step_spec structures do not match')
      nest_utils.assert_same_structure(
          policy_state,
          self._policy_state_spec,
          message='policy_state and policy_state_spec structures do not match')
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
    if self._validate_args:
      nest_utils.assert_same_structure(
          step,
          self._policy_step_spec,
          message=('distribution output and policy_step_spec structures '
                   'do not match'))
    return step

  def update(self,
             policy,
             tau: float = 1.0,
             tau_non_trainable: Optional[float] = None,
             sort_variables_by_name: bool = False) -> tf.Operation:
    """Update the current policy with another policy.

    This would include copying the variables from the other policy.

    Args:
      policy: Another policy it can update from.
      tau: A float scalar in [0, 1]. When tau is 1.0 (the default), we do a hard
        update. This is used for trainable variables.
      tau_non_trainable: A float scalar in [0, 1] for non_trainable variables.
        If None, will copy from tau.
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
          tau_non_trainable=tau_non_trainable,
          sort_variables_by_name=sort_variables_by_name)
    else:
      return tf.no_op()

  @property
  def emit_log_probability(self) -> bool:
    """Whether this policy instance emits log probabilities or not."""
    return self._emit_log_probability

  @property
  def time_step_spec(self) -> ts.TimeStep:
    """Describes the `TimeStep` tensors returned by `step()`.

    Returns:
      A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
      which describe the shape, dtype and name of each tensor returned by
      `step()`.
    """
    return self._time_step_spec

  @property
  def action_spec(self) -> types.NestedTensorSpec:
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
  def policy_state_spec(self) -> types.NestedTensorSpec:
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
  def info_spec(self) -> types.NestedTensorSpec:
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
  def policy_step_spec(self) -> policy_step.PolicyStep:
    """Describes the output of `action()`.

    Returns:
      A nest of TensorSpec which describe the shape and dtype of each Tensor
      emitted by `action()`.
    """
    return self._policy_step_spec

  # TODO(kbanoop, ebrevdo): Should this be collect_data_spec to mirror agents?
  @property
  def trajectory_spec(self) -> trajectory.Trajectory:
    """Describes the Tensors written when using this policy with an environment.

    Returns:
      A `Trajectory` containing all tensor specs associated with the
      observation_spec, action_spec, policy_state_spec, and info_spec of
      this policy.
    """
    return self._trajectory_spec

  @property
  def collect_data_spec(self) -> trajectory.Trajectory:
    """Describes the Tensors written when using this policy with an environment.

    Returns:
      A nest of TensorSpec which describe the shape and dtype of each Tensor
      required to train the agent which generated this policy.
    """
    return self._trajectory_spec

  # Subclasses MAY optionally override _action.
  def _action(self, time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
    """Implementation of `action`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seed: Seed to use if action performs sampling (optional).

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: An action Tensor matching the `action_spec`.
        `state`: A policy state tensor to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.
    """
    seed_stream = tfp.util.SeedStream(seed=seed, salt='tf_agents_tf_policy')
    distribution_step = self._distribution(time_step, policy_state)  # pytype: disable=wrong-arg-types
    actions = tf.nest.map_structure(
        lambda d: reparameterized_sampling.sample(d, seed=seed_stream()),
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
  def _distribution(
      self, time_step: ts.TimeStep,
      policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
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
    raise NotImplementedError()

  # Subclasses MAY optionally overwrite _get_initial_state.
  def _get_initial_state(self, batch_size: int) -> types.NestedTensor:
    """Returns the initial state of the policy network.

    Args:
      batch_size: A constant or Tensor holding the batch size. Can be None, in
        which case the state will not have a batch dimension added.

    Returns:
      A nest of zero tensors matching the spec of the policy network state.
    """
    return tensor_spec.zero_spec_nest(
        self._policy_state_spec,
        outer_dims=None if batch_size is None else [batch_size])
