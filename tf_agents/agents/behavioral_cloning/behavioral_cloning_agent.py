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

"""Behavioral Cloning Agents.

Implements generic form of behavioral cloning.

Users must provide their own loss functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils


class BehavioralCloningLossInfo(collections.namedtuple(
    'BehavioralCloningLossInfo', ('loss',))):
  """Stores a per-batch-entry loss value."""
  pass


@gin.configurable
class BehavioralCloningAgent(tf_agent.TFAgent):
  """An behavioral cloning Agent.

  Implements behavioral cloning, wherein the network learns to clone
  given experience.  Users must provide their own loss functions. Note this
  implementation will use a QPolicy. To use with other policies subclass this
  agent and override the `_get_policies` method. Note the cloning_network must
  match the requirements of the generated policies.

  Behavioral cloning was proposed in the following articles:

  Pomerleau, D.A., 1991. Efficient training of artificial neural networks for
  autonomous navigation. Neural Computation, 3(1), pp.88-97.

  Russell, S., 1998, July. Learning agents for uncertain environments.
  In Proceedings of the eleventh annual conference on Computational learning
  theory (pp. 101-103). ACM.
  """

  # TODO(b/127327645): This causes a loop failure when RNNs are enabled.
  _enable_functions = False

  def __init__(
      self,
      time_step_spec,
      action_spec,
      cloning_network,
      optimizer,
      epsilon_greedy=0.1,
      # Params for training.
      loss_fn=None,
      gradient_clipping=None,
      # Params for debugging
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=None,
      name=None):
    """Creates an behavioral cloning Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      cloning_network: A tf_agents.network.Network to be used by the agent.
        The network will be called as

          ```
          network(observation, step_type, network_state=None)
          ```
        (with `network_state` optional) and must return a 2-tuple with elements
        `(output, next_network_state)` where `output` will be passed as the
        first argument to `loss_fn`, and used by a `Policy`.  Input tensors will
        be shaped `[batch, time, ...]` when training, and they will be shaped
        `[batch, ...]` when the network is called within a `Policy`.  If
        `cloning_network` has an empty network state, then for training
        `time` will always be `1` (individual examples).
      optimizer: The optimizer to use for training.
      epsilon_greedy: probability of choosing a random action in the default
        epsilon-greedy collect policy (used only if a wrapper is not provided to
        the collect_policy method).
      loss_fn: A function for computing the error between the output of the
        cloning network and the action that was taken. If None, the loss
        depends on the action dtype.  If the dtype is integer, then `loss_fn`
        is

        ```python
        def loss_fn(logits, action):
          return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=action - action_spec.minimum, logits=logits)
        ```

        If the dtype is floating point, the loss is
        `tf.math.squared_difference`.

        `loss_fn` must return a loss value for each element of the batch.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If the action spec contains more than one action.
    """
    tf.Module.__init__(self, name=name)

    flat_action_spec = tf.nest.flatten(action_spec)
    self._num_actions = [
        spec.maximum - spec.minimum + 1 for spec in flat_action_spec
    ]

    # TODO(oars): Get behavioral cloning working with more than one dim in
    # the actions.
    if len(flat_action_spec) > 1:
      raise NotImplementedError(
          'Multi-arity actions are not currently supported.')

    if loss_fn is None:
      loss_fn = self._get_default_loss_fn(flat_action_spec[0])

    self._cloning_network = cloning_network
    self._loss_fn = loss_fn
    self._epsilon_greedy = epsilon_greedy
    self._optimizer = optimizer
    self._gradient_clipping = gradient_clipping

    policy, collect_policy = self._get_policies(time_step_spec, action_spec,
                                                cloning_network)

    super(BehavioralCloningAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=1 if not cloning_network.state_spec else None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _get_default_loss_fn(self, spec):
    if spec.dtype.is_floating:
      return tf.math.squared_difference
    if spec.shape.ndims > 1:
      raise NotImplementedError(
          'Only scalar and one dimensional integer actions are supported.')
    # TODO(ebrevdo): Maybe move the subtraction of the minimum into a
    # self._label_fn and rewrite this.
    def xent_loss_fn(logits, actions):
      # Subtract the minimum so that we get a proper cross entropy loss on
      # [0, maximum - minimum).
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=actions - spec.minimum)

    return xent_loss_fn

  def _get_policies(self, time_step_spec, action_spec, cloning_network):
    policy = q_policy.QPolicy(
        time_step_spec, action_spec, q_network=self._cloning_network)
    collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
        policy, epsilon=self._epsilon_greedy)
    policy = greedy_policy.GreedyPolicy(policy)
    return policy, collect_policy

  def _initialize(self):
    return tf.no_op()

  def _train(self, experience, weights=None):
    loss_info = self._loss(experience, weights=weights)

    transform_grads_fn = None
    if self._gradient_clipping is not None:
      transform_grads_fn = eager_utils.clip_gradient_norms_fn(
          self._gradient_clipping)

    loss_info = eager_utils.create_train_step(
        loss_info,
        self._optimizer,
        total_loss_fn=lambda loss_info: loss_info.loss,
        global_step=self.train_step_counter,
        transform_grads_fn=transform_grads_fn,
        summarize_gradients=self._summarize_grads_and_vars,
        variables_to_train=lambda: self._cloning_network.trainable_weights,
    )

    return loss_info

  @eager_utils.future_in_eager_mode
  # TODO(b/79688437): Figure out how to enable defun for Eager mode.
  # @tfe.defun
  def _loss(self, experience, weights=None):
    """Computes loss for behavioral cloning.

    Args:
      experience: A `Trajectory` containing experience.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.

    Returns:
      loss: A `LossInfo` struct.

    Raises:
      ValueError:
        If the number of actions is greater than 1.
    """
    with tf.name_scope('loss'):
      actions = tf.nest.flatten(experience.action)[0]
      logits, _ = self._cloning_network(
          experience.observation,
          experience.step_type)

      boundary_weights = tf.cast(~experience.is_boundary(), logits.dtype)
      error = boundary_weights * self._loss_fn(logits, actions)

      if nest_utils.is_batched_nested_tensors(
          experience.action, self.action_spec, num_outer_dims=2):
        # Do a sum over the time dimension.
        error = tf.reduce_sum(input_tensor=error, axis=1)

      # Average across the elements of the batch.
      # Note: We use an element wise loss above to ensure each element is always
      #   weighted by 1/N where N is the batch size, even when some of the
      #   weights are zero due to boundary transitions. Weighting by 1/K where K
      #   is the actual number of non-zero weight would artificially increase
      #   their contribution in the loss. Think about what would happen as
      #   the number of boundary samples increases.
      if weights is not None:
        error *= weights
      loss = tf.reduce_mean(input_tensor=error)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='loss', data=loss, step=self.train_step_counter)

      if self._summarize_grads_and_vars:
        with tf.name_scope('Variables/'):
          for var in self._cloning_network.trainable_weights:
            tf.compat.v2.summary.histogram(
                name=var.name.replace(':', '_'),
                data=var,
                step=self.train_step_counter)

      if self._debug_summaries:
        common.generate_tensor_summaries('errors', error,
                                         self.train_step_counter)

      return tf_agent.LossInfo(loss, BehavioralCloningLossInfo(loss=error))
