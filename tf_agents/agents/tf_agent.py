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

"""TensorFlow RL Agent API."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import abc
import collections
from typing import Optional

import six
import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils


LossInfo = collections.namedtuple("LossInfo", ("loss", "extra"))


@six.add_metaclass(abc.ABCMeta)
class TFAgent(tf.Module):
  """Abstract base class for TF-based RL and Bandits agents.

  The agent serves the following purposes:

  * Training by reading minibatches of `experience`, and updating some set
    of network weights (using the `train` method).

  * Exposing `policy` objects which can be used to interact with an environment:
    either to explore and collect new training data, or to maximize reward
    in the given task.

  The agents' main training methods and properties are:

  * `initialize`: Perform any self-initialization before training.

  * `train`: This method reads minibatch experience from a replay buffer or
    logs on disk, and updates some internal networks.

  * `preprocess_sequence`: Some algorithms need to perform sequence
    preprocessing on logs containing "full episode" or "long subset" sequences,
    to create intermediate items that can then be used by `train`, even if
    `train` does not see the full sequences.  In many cases this is just
    the identity: it passes experience through untouched.  This function
    is typically passed to the argument

    `ReplayBuffer.as_dataset(..., sequence_preprocess_fn=...)`

  * `training_data_spec`: Property that describes the structure expected of
    the `experience` argument passed to `train`.

  * `train_sequence_length`: Property that describes the **second** dimension
    of all tensors in the `experience` argument passed `train`.  All tensors
    passed to train must have the shape `[batch_size, sequence_length, ...]`,
    and some Agents require this to be a fixed value.  For example, in regular
    `DQN`, this second `sequence_length` dimension must be equal to `2` in all
    `experience`.  In contrast, `n-step DQN` will have this equal to `n + 1` and
    `DQN` agents constructed with `RNN` networks will have this equal to `None`,
    meaning any length sequences are allowed.

    This value may be `None`, to mean minibatches containing subsequences of any
    length are allowed (so long as they're all the same length).  This is
    typically the case with agents constructed with `RNN` networks.

    This value is typically passed as a ReplayBuffer's
    `as_dataset(..., num_steps=...)` argument.

  * `collect_data_spec`: Property that describes the structure expected of
    experience collected by `agent.collect_policy`.  This is typically
    identical to `training_data_spec`, but may be different if
    `preprocess_sequence` method is not the identity.  In this case,
    `preprocess_sequence` is expected to read sequences matching
    `collect_data_spec` and emit sequences matching `training_data_spec`.

  The agent exposes `TFPolicy` objects for interacting with environments:

  * `policy`: Property that returns a policy meant for "exploiting" the
    environment to its best ability.  This tends to mean the "production" policy
    that doesn't collect additional info for training.  Works best when
    the agent is fully trained.

    TODO(b/154870654): Not all agents are properly exporting properly greedy
    "production" policies yet.  We have to clean this up.  In particular,
    we have to update PPO and SAC's `policy` objects.

  * `collect_policy`: Property that returns a policy meant for "exploring"
    the environment to collect more data for training.  This tends to mean
    a policy involves some level of randomized behavior and additional info
    logging.

  * `time_step_spec`: Property describing the observation and reward signatures
    of the environment this agent's policies operate in.

  * `action_spec`: Property describing the action signatures of the environment
    this agent's policies operate in.


  **NOTE**: For API consistency, subclasses are not allowed to override public
  methods of `TFAgent` class. Instead, they may implement the protected methods
  including `_initialize`, `_train`, and `_preprocess_sequence`. This
  public-calls-private convention allowed this base class to do things like
  properly add `spec` and shape checks, which provide users an easier experience
  when debugging their environments and networks.
  """

  # TODO(b/127327645) Remove this attribute.
  # This attribute allows subclasses to back out of automatic tf.function
  # attribute inside TF1 (for autodeps).
  _enable_functions = True

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      policy: tf_policy.TFPolicy,
      collect_policy: tf_policy.TFPolicy,
      train_sequence_length: Optional[int],
      num_outer_dims: int = 2,
      training_data_spec: Optional[types.NestedTensorSpec] = None,
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      enable_summaries: bool = True,
      train_step_counter: Optional[tf.Variable] = None):
    """Meant to be called by subclass constructors.

    Args:
      time_step_spec: A nest of tf.TypeSpec representing the time_steps.
        Provided by the user.
      action_spec: A nest of BoundedTensorSpec representing the actions.
        Provided by the user.
      policy: An instance of `tf_policy.TFPolicy` representing the
        Agent's current policy.
      collect_policy: An instance of `tf_policy.TFPolicy` representing the
        Agent's current data collection policy (used to set `self.step_spec`).
      train_sequence_length: A python integer or `None`, signifying the number
        of time steps required from tensors in `experience` as passed to
        `train()`.  All tensors in `experience` will be shaped `[B, T, ...]` but
        for certain agents, `T` should be fixed.  For example, DQN requires
        transitions in the form of 2 time steps, so for a non-RNN DQN Agent, set
        this value to 2.  For agents that don't care, or which can handle `T`
        unknown at graph build time (i.e. most RNN-based agents), set this
        argument to `None`.
      num_outer_dims: The number of outer dimensions for the agent. Must be
        either 1 or 2. If 2, training will require both a batch_size and time
        dimension on every Tensor; if 1, training will require only a batch_size
        outer dimension.
      training_data_spec: A nest of TensorSpec specifying the structure of data
        the train() function expects. If None, defaults to the trajectory_spec
        of the collect_policy.
      debug_summaries: A bool; if true, subclasses should gather debug
        summaries.
      summarize_grads_and_vars: A bool; if true, subclasses should additionally
        collect gradient and variable summaries.
      enable_summaries: A bool; if false, subclasses should not gather any
        summaries (debug or otherwise); subclasses should gate *all* summaries
        using either `summaries_enabled`, `debug_summaries`, or
        `summarize_grads_and_vars` properties.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.

    Raises:
      ValueError: If `num_outer_dims` is not in `[1, 2]`.
    """
    common.check_tf1_allowed()
    common.tf_agents_gauge.get_cell("TFAgent").set(True)
    common.tf_agents_gauge.get_cell(str(type(self))).set(True)
    if not isinstance(time_step_spec, ts.TimeStep):
      raise TypeError(
          "The `time_step_spec` must be an instance of `TimeStep`, but is `{}`."
          .format(type(time_step_spec)))

    if num_outer_dims not in [1, 2]:
      raise ValueError("num_outer_dims must be in [1, 2].")

    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self._policy = policy
    self._collect_policy = collect_policy
    self._train_sequence_length = train_sequence_length
    self._num_outer_dims = num_outer_dims
    self._debug_summaries = debug_summaries
    self._summarize_grads_and_vars = summarize_grads_and_vars
    self._enable_summaries = enable_summaries
    self._training_data_spec = training_data_spec
    # Data context for data collected directly from the collect policy.
    self._collect_data_context = data_converter.DataContext(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=collect_policy.info_spec)
    # Data context for data passed to train().  May be different if
    # training_data_spec is provided.
    if training_data_spec is not None:
      # training_data_spec can be anything; so build a data_context
      # via best-effort with fall-backs to the collect data spec.
      training_discount_spec = getattr(
          training_data_spec, "discount", time_step_spec.discount)
      training_observation_spec = getattr(
          training_data_spec, "observation", time_step_spec.observation)
      training_reward_spec = getattr(
          training_data_spec, "reward", time_step_spec.reward)
      training_step_type_spec = getattr(
          training_data_spec, "step_type", time_step_spec.step_type)
      training_policy_info_spec = getattr(
          training_data_spec, "policy_info", collect_policy.info_spec)
      training_action_spec = getattr(
          training_data_spec, "action", action_spec)
      self._data_context = data_converter.DataContext(
          time_step_spec=ts.TimeStep(
              discount=training_discount_spec,
              observation=training_observation_spec,
              reward=training_reward_spec,
              step_type=training_step_type_spec),
          action_spec=training_action_spec,
          info_spec=training_policy_info_spec)
    else:
      self._data_context = data_converter.DataContext(
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          info_spec=collect_policy.info_spec)
    if train_step_counter is None:
      train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    self._train_step_counter = train_step_counter
    self._train_fn = common.function_in_tf1()(self._train)
    self._initialize_fn = common.function_in_tf1()(self._initialize)
    self._preprocess_sequence_fn = common.function_in_tf1()(
        self._preprocess_sequence)
    self._loss_fn = common.function_in_tf1()(self._loss)

  def initialize(self) -> Optional[tf.Operation]:
    """Initializes the agent.

    Returns:
      An operation that can be used to initialize the agent.

    Raises:
      RuntimeError: If the class was not initialized properly (`super.__init__`
        was not called).
    """
    if self._enable_functions and getattr(self, "_initialize_fn", None) is None:
      raise RuntimeError(
          "Cannot find _initialize_fn.  Did %s.__init__ call super?"
          % type(self).__name__)
    if self._enable_functions:
      return self._initialize_fn()
    else:
      return self._initialize()

  def preprocess_sequence(self,
                          experience: types.NestedTensor) -> types.NestedTensor:
    """Defines preprocess_sequence function to be fed into replay buffers.

    This defines how we preprocess the collected data before training.
    Defaults to pass through for most agents.
    Structure of `experience` must match that of `self.collect_data_spec`.

    Args:
      experience: a `Trajectory` shaped [batch, time, ...] or [time, ...] which
        represents the collected experience data.

    Returns:
      A post processed `Trajectory` with the same shape as the input.
    """
    if self._enable_functions:
      preprocessed_sequence = self._preprocess_sequence_fn(experience)
    else:
      preprocessed_sequence = self._preprocess_sequence(experience)

    return preprocessed_sequence

  def train(self,
            experience: types.NestedTensor,
            weights: Optional[types.Tensor] = None,
            **kwargs) -> LossInfo:
    """Trains the agent.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.training_data_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.train_step_length` if that
        property is not `None`.
      weights: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
        containing weights to be used when calculating the total train loss.
        Weights are typically multiplied elementwise against the per-batch loss,
        but the implementation is up to the Agent.
      **kwargs: Any additional data to pass to the subclass.

    Returns:
        A `LossInfo` loss tuple containing loss and info tensors.
        - In eager mode, the loss values are first calculated, then a train step
          is performed before they are returned.
        - In graph mode, executing any or all of the loss tensors
          will first calculate the loss value(s), then perform a train step,
          and return the pre-train-step `LossInfo`.

    Raises:
      RuntimeError: If the class was not initialized properly (`super.__init__`
        was not called).
    """
    if self._enable_functions and getattr(self, "_train_fn", None) is None:
      raise RuntimeError(
          "Cannot find _train_fn.  Did %s.__init__ call super?"
          % type(self).__name__)

    if self._enable_functions:
      loss_info = self._train_fn(
          experience=experience, weights=weights, **kwargs)
    else:
      loss_info = self._train(experience=experience, weights=weights, **kwargs)

    if not isinstance(loss_info, LossInfo):
      raise TypeError(
          "loss_info is not a subclass of LossInfo: {}".format(loss_info))
    return loss_info

  def loss(self,
           experience: types.NestedTensor,
           weights: Optional[types.Tensor] = None,
           **kwargs) -> LossInfo:
    """Gets loss from the agent.

    If the user calls this from _train, it must be in a `tf.GradientTape` scope
    in order to apply gradients to trainable variables.
    If intermediate gradient steps are needed, _loss and _train will return
    different values since _loss only supports updating all gradients at once
    after all losses have been calculated.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.training_data_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.train_step_length` if that
        property is not `None`.
      weights: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
        containing weights to be used when calculating the total train loss.
        Weights are typically multiplied elementwise against the per-batch loss,
        but the implementation is up to the Agent.
      **kwargs: Any additional data as args to `loss`.

    Returns:
        A `LossInfo` loss tuple containing loss and info tensors.

    Raises:
      RuntimeError: If the class was not initialized properly (`super.__init__`
        was not called).
    """
    if self._enable_functions and getattr(self, "_loss_fn", None) is None:
      raise RuntimeError(
          "Cannot find _loss_fn.  Did %s.__init__ call super?"
          % type(self).__name__)

    if self._enable_functions:
      loss_info = self._loss_fn(
          experience=experience, weights=weights, **kwargs)
    else:
      loss_info = self._loss(experience=experience, weights=weights, **kwargs)

    if not isinstance(loss_info, LossInfo):
      raise TypeError(
          "loss_info is not a subclass of LossInfo: {}".format(loss_info))
    return loss_info

  def _apply_loss(self, aggregated_losses, variables_to_train, tape, optimizer):
    total_loss = aggregated_losses.total_loss
    tf.debugging.check_numerics(total_loss, "Loss is inf or nan")
    assert list(variables_to_train), "No variables in the agent's network."

    grads = tape.gradient(total_loss, variables_to_train)
    grads_and_vars = list(zip(grads, variables_to_train))

    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self.summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

    if self.summaries_enabled:
      dict_losses = {
          "loss": aggregated_losses.weighted,
          "reg_loss": aggregated_losses.regularization,
          "total_loss": total_loss
      }
      common.summarize_scalar_dict(
          dict_losses, step=self.train_step_counter, name_scope="Losses/")

  def post_process_policy(self) -> tf_policy.TFPolicy:
    """Post process policies after training.

    The policies of some agents require expensive post processing after training
    before they can be used. e.g. A Recommender agent might require rebuilding
    an index of actions. For such agents, this method will return a post
    processed version of the policy. The post processing may either update the
    existing policies in place or create a new policy, depnding on the agent.
    The default implementation for agents that do not want to override this
    method is to return agent.policy.

    Returns:
      The post processed policy.
    """
    return self.policy

  @property
  def time_step_spec(self) -> ts.TimeStep:
    """Describes the `TimeStep` tensors expected by the agent.

    Returns:
      A `TimeStep` namedtuple with `TensorSpec` objects instead of Tensors,
      which describe the shape, dtype and name of each tensor.
    """
    return self._time_step_spec

  @property
  def action_spec(self) -> types.NestedTensorSpec:
    """TensorSpec describing the action produced by the agent.

    Returns:
      An single BoundedTensorSpec, or a nested dict, list or tuple of
      `BoundedTensorSpec` objects, which describe the shape and
      dtype of each action Tensor.
    """
    return self._action_spec

  @property
  def data_context(self) -> data_converter.DataContext:
    return self._data_context

  @property
  def collect_data_context(self) -> data_converter.DataContext:
    return self._collect_data_context

  @property
  def policy(self) -> tf_policy.TFPolicy:
    """Return the current policy held by the agent.

    Returns:
      A `tf_policy.TFPolicy` object.
    """
    return self._policy

  @property
  def collect_policy(self) -> tf_policy.TFPolicy:
    """Return a policy that can be used to collect data from the environment.

    Returns:
      A `tf_policy.TFPolicy` object.
    """
    return self._collect_policy

  @property
  def collect_data_spec(self) -> types.NestedTensorSpec:
    """Returns a `Trajectory` spec, as expected by the `collect_policy`.

    Returns:
      A `Trajectory` spec.
    """
    return self.collect_data_context.trajectory_spec

  @property
  def training_data_spec(self) -> types.NestedTensorSpec:
    """Returns a trajectory spec, as expected by the train() function."""
    if self._training_data_spec is not None:
      return self._training_data_spec
    else:
      return self.collect_data_spec

  @property
  def train_sequence_length(self) -> int:
    """The number of time steps needed in experience tensors passed to `train`.

    Train requires experience to be a `Trajectory` containing tensors shaped
    `[B, T, ...]`.  This argument describes the value of `T` required.

    For example, for non-RNN DQN training, `T=2` because DQN requires single
    transitions.

    If this value is `None`, then `train` can handle an unknown `T` (it can be
    determined at runtime from the data).  Most RNN-based agents fall into
    this category.

    Returns:
      The number of time steps needed in experience tensors passed to `train`.
      May be `None` to mean no constraint.
    """
    return self._train_sequence_length

  @property
  def summaries_enabled(self) -> bool:
    return self._enable_summaries

  @property
  def debug_summaries(self) -> bool:
    return self._debug_summaries and self.summaries_enabled

  @property
  def summarize_grads_and_vars(self) -> bool:
    return self._summarize_grads_and_vars and self.summaries_enabled

  @property
  def train_step_counter(self) -> tf.Variable:
    return self._train_step_counter

  def _initialize(self) -> Optional[tf.Operation]:
    """Returns an op to initialize the agent."""
    pass

  def _preprocess_sequence(
      self, experience: types.NestedTensor) -> types.NestedTensor:
    """Defines preprocess_sequence function to be fed into replay buffers.

    This defines how we preprocess the collected data before training.
    Defaults to pass through for most agents. Subclasses may override this.

    Args:
      experience: a `Trajectory` shaped [batch, time, ...] or [time, ...] which
        represents the collected experience data.

    Returns:
      A post processed `Trajectory` with the same shape as the input.
    """
    return experience

  def _loss(self, experience: types.NestedTensor,
            weights: types.Tensor) -> Optional[LossInfo]:
    """Computes loss.

    This method does not increment self.train_step_counter or upgrade gradients.
    By default, any networks are called with `training=False`.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.training_data_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.train_step_length` if that property is not
        `None`.
      weights: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
        containing weights to be used when calculating the total train loss.
        Weights are typically multiplied elementwise against the per-batch loss,
        but the implementation is up to the Agent.

    Returns:
        A `LossInfo` containing the loss *before* the training step is taken.
        In most cases, if `weights` is provided, the entries of this tuple will
        have been calculated with the weights.  Note that each Agent chooses
        its own method of applying weights.

    Raises:
      NotImplementedError: If this method has not been overridden.
    """
    raise NotImplementedError("Loss not implemented.")

  # Subclasses must implement these methods.
  @abc.abstractmethod
  def _train(self, experience: types.NestedTensor,
             weights: types.Tensor) -> LossInfo:
    """Returns an op to train the agent.

    This method *must* increment self.train_step_counter exactly once.
    TODO(b/126271669): Consider automatically incrementing this

    Args:
      experience: A batch of experience data in the form of a `Trajectory`. The
        structure of `experience` must match that of `self.training_data_spec`.
        All tensors in `experience` must be shaped `[batch, time, ...]` where
        `time` must be equal to `self.train_step_length` if that property is
        not `None`.
      weights: (optional).  A `Tensor`, either `0-D` or shaped `[batch]`,
        containing weights to be used when calculating the total train loss.
        Weights are typically multiplied elementwise against the per-batch loss,
        but the implementation is up to the Agent.

    Returns:
        A `LossInfo` containing the loss *before* the training step is taken.
        In most cases, if `weights` is provided, the entries of this tuple will
        have been calculated with the weights.  Note that each Agent chooses
        its own method of applying weights.
    """
