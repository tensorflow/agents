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

"""CEM Policy for QTOPT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Optional, Sequence, Tuple, Union

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import tf_policy
from tf_agents.policies.samplers import qtopt_cem_actions_sampler
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils

try:
  # Python 3.3 and above.
  collections_abc = collections.abc
except AttributeError:
  collections_abc = collections


def convert_nest_lists_to_np_array(nested_list: types.NestedTensorOrArray
                                   ) -> Union[Tuple[Any], np.ndarray]:
  """Convert nest lists to numpy array.

  Args:
    nested_list: A nested strucutre of lists.

  Raises:
    ValueError: If the input is not
      collections_abc.Mapping/tuple/list/np.ndarray.

  Returns:
    A nested structure of numpy arrays.
  """
  if isinstance(nested_list, collections_abc.Mapping):
    ordered_items = [
        (k, convert_nest_lists_to_np_array(v))
        for k, v in nested_list.items()]
    if isinstance(nested_list, collections.defaultdict):
      subset = type(nested_list)(nested_list.default_factory, ordered_items)
    else:
      subset = type(nested_list)(ordered_items)
    return subset
  elif isinstance(nested_list, tuple):
    return tuple(convert_nest_lists_to_np_array(v) for v in nested_list)
  elif isinstance(nested_list, list):
    return np.array(nested_list, np.float32)
  elif isinstance(nested_list, np.ndarray):
    # Converting all dtype to be float32 makes it easier to do CEM loop.
    # The dtype of the final sampled action will be converted back according
    # to the spec in the sampler.
    if nested_list.dtype != np.float32:
      nested_list = nested_list.astype(np.float32)
    return nested_list
  else:
    raise ValueError('The nested type is not supported.')


@gin.configurable
class CEMPolicy(tf_policy.TFPolicy):
  """Class to build CEM Policy.

  Some notations used in the comments below are:

  B: batch_size
  A: action_size
  N: num_samples
  M: num_elites

  Supports the following three cases.

  1. Nested action_spec containing 1d continuous action.
  2. Nested action_spec containing 1d continuous or discrete action.
  3. Nested action_spec containing 1d continuous action and 1 one_hot action.

  Note: all actions need to be 1-d array.
  """

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               q_network: network.Network,
               sampler: qtopt_cem_actions_sampler.ActionsSampler,
               init_mean: types.NestedArray,
               init_var: types.NestedArray,
               actor_policy: Optional[tf_policy.TFPolicy] = None,
               minimal_var: float = 0.0001,
               info_spec: types.NestedSpecTensorOrArray = (),
               num_samples: int = 32,
               num_elites: int = 4,
               num_iterations: int = 32,
               emit_log_probability: bool = False,
               preprocess_state_action: bool = True,
               training: bool = False,
               weights: types.NestedTensorOrArray = None,
               name: Optional[str] = None):
    """Builds a CEM-Policy given a network and a sampler.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      q_network: An instance of a `tf_agents.network.Network`, callable via
        `network(observation, step_type) -> (output, final_state)`.
      sampler: Samples the actions needed for the CEM.
      init_mean: A list or tuple or scalar, reprenting initial mean for actions.
      init_var: A list or tuple or scalar, reprenting initial var for actions.
      actor_policy: Optional actor policy.
      minimal_var: Minimal variance to prevent CEM distributon collapsing.
      info_spec: A policy info spec.
      num_samples: Number of samples to sample each round.
      num_elites: Number of best actions each round to refit the distribution
        with.
      num_iterations: Number of iterations to run the CEM loop.
      emit_log_probability: Whether to emit log-probs in info of `PolicyStep`.
      preprocess_state_action: The shape of state is (B, ...) and the shape of
        action is (B, N, A). When preprocess_state_action is enabled, the state
        will be tile_batched to be (BxN, ...) and the action will be reshaped
        to be (BxN, A). When preprocess_state_action is not enabled, the same
        operation needs to be done inside the network. This is helpful when the
        input have large memory requirements and the replication of state could
        happen after a few layers inside the network.
      training: Whether it is in training mode or inference mode.
      weights: A nested structure of weights w/ the same structure as action.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      ValueError: If `q_network.action_spec` exists and is not compatible with
        `action_spec`.
    """
    network_action_spec = getattr(q_network, 'action_spec', None)

    if network_action_spec is not None:
      if not action_spec.is_compatible_with(network_action_spec):
        raise ValueError(
            'action_spec must be compatible with q_network.action_spec; '
            'instead got action_spec=%s, q_network.action_spec=%s' %
            (action_spec, network_action_spec))

    if q_network:
      network_utils.check_single_floating_network_output(
          q_network.create_variables(),
          expected_output_shape=(), label=str(q_network))
      policy_state_spec = q_network.state_spec
    else:
      policy_state_spec = ()

    self._actor_policy = actor_policy
    self._q_network = q_network
    self._init_mean = init_mean
    self._init_var = init_var
    self._minimal_var = minimal_var
    self._num_samples = num_samples  # N
    self._num_elites = num_elites  # M
    self._num_iterations = num_iterations
    self._actions_sampler = sampler
    self._observation_spec = time_step_spec.observation
    self._training = training
    self._preprocess_state_action = preprocess_state_action
    self._weights = weights

    super(CEMPolicy, self).__init__(
        time_step_spec,
        action_spec,
        info_spec=info_spec,
        policy_state_spec=policy_state_spec,
        clip=False,
        emit_log_probability=emit_log_probability,
        name=name)

  def _initial_params(self, batch_size: tf.Tensor
                      ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns initial mean and variance tensors.

    Broadcasts the initial mean and variance to the requested batch_size.

    Args:
      batch_size: The requested batch_size.

    Returns:
      mean: A [B, A] sized tensors where each row is the initial_mean.
      var: A [B, A] sized tensors where each row is the initial_var.
    """

    def broadcast_to_batch(array):
      tensor = tf.constant(array)
      action_size = tf.shape(tensor)[-1]
      return tf.broadcast_to(tensor, [batch_size, action_size])

    mean = tf.nest.map_structure(
        broadcast_to_batch,
        convert_nest_lists_to_np_array(self._init_mean))
    var = tf.nest.map_structure(
        broadcast_to_batch,
        convert_nest_lists_to_np_array(self._init_var))

    return mean, var

  def actor_func(
      self, observation: types.NestedTensorOrArray,
      step_type: Optional[tf.Tensor], policy_state: Sequence[tf.Tensor]
  ) -> Tuple[types.NestedTensor, types.NestedTensor, Sequence[tf.Tensor]]:
    """Returns an action to perform using CEM given the q network.

    Args:
      observation: Observation for which we need to find a CEM based action.
      step_type: A `StepType` enum value.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.

    Returns:
      A [B, A] tensor representing the action taken by the actor,
      that is best in the CEM sense
    """
    batch_size = nest_utils.get_outer_shape(
        observation, self._observation_spec)[0]
    outer_rank = nest_utils.get_outer_rank(
        observation, self._observation_spec)
    if outer_rank == 2:
      seq_size = nest_utils.get_outer_shape(
          observation, self._observation_spec)[1]
      batch_size = batch_size * seq_size
    # Run the sampler to sample N actions from a distribution
    # Run the q_func for each of those actions and choose the best M (<N)
    # Fit another distribution to the best M actions and repeat k times.

    def body(mean, var, i, iters, best_actions, best_scores,
             best_next_policy_state):
      """Defines the body of the while loop in graph.

      Args:
        mean: A [B, A] sized tensor, where, a is the size of the action space,
          indicating the mean value of the sample distribution
        var : [B, A]  sized tensor, where, a is the size of the action space,
          indicating the variance value of the sample distribution
        i: tensor indicating the iteration index
        iters: tensor representing the max number of iteration to run
        best_actions: the best action so far as per the CEM iteration [B, A]
        best_scores: best score.
        best_next_policy_state: A Tensor, or a nested dict, list or tuple of
          Tensors representing the best next policy_state.

      Returns:
        list of tensors [new_mean, new_var, i+1, iters, new_best_actions,
          new_best_scores, new_best_next_policy_state]
      """
      del best_actions, best_scores, best_next_policy_state

      # Prevent variance to collapse to 0.0 (stuck in local minimum).
      var = tf.nest.map_structure(
          lambda v: tf.maximum(v, self._minimal_var), var)

      # Sample a batch of actions with the shape of [B, N, A] or [BxT, N, A]
      actions = self._actions_sampler.sample_batch_and_clip(
          self._num_samples, mean, var, observation)  # pytype: disable=wrong-arg-count  # trace-all-classes

      if outer_rank == 2:
        scores, next_policy_state = self._score_with_time(
            observation, actions, step_type, policy_state, seq_size)  # [BxT, N]
      else:
        scores, next_policy_state = self._score(
            observation, actions, step_type, policy_state)  # [B, N]

      best_scores, ind = tf.nn.top_k(scores, self._num_elites)  # ind: [B, M]

      actions_float = tf.nest.map_structure(
          lambda t: tf.cast(t, tf.float32), actions)
      mean, var = self._actions_sampler.refit_distribution_to(
          ind, actions_float)

      best_next_policy_state = next_policy_state

      def select_best_actions(actions):
        best_actions = tf.gather(actions, ind, batch_dims=1)
        return best_actions

      best_actions = tf.nest.map_structure(select_best_actions, actions)
      return [mean, var, tf.add(i, 1), iters, best_actions, best_scores,
              best_next_policy_state]

    def cond(mean, var, i, iters, best_actions, best_scores,
             best_next_policy_state):
      del mean, var, best_actions, best_scores, best_next_policy_state
      return tf.less(i, iters)

    mean, var = self._initial_params(batch_size)
    iters = tf.constant(self._num_iterations)

    def init_best_actions(action_spec):
      best_actions = tf.zeros(
          [batch_size, self._num_elites, action_spec.shape.as_list()[0]],
          dtype=action_spec.dtype)
      return best_actions

    best_actions = tf.nest.map_structure(init_best_actions, self._action_spec)
    best_scores = tf.zeros([batch_size, self._num_elites], dtype=tf.float32)

    # Run the while loop for CEM in-graph.
    _, _, _, _, best_actions, best_scores, best_next_policy_state = (
        tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[mean, var, 0, iters, best_actions, best_scores,
                       policy_state],
            shape_invariants=[
                tf.nest.map_structure(
                    lambda m: [None] + m.get_shape()[1:], mean),
                tf.nest.map_structure(
                    lambda v: [None] + v.get_shape()[1:], var),
                tf.TensorShape(()), iters.get_shape(),
                tf.nest.map_structure(
                    lambda a: [None] + a.get_shape()[1:], best_actions),
                tf.TensorShape([None, self._num_elites]),
                () if policy_state is () else tf.nest.map_structure(  # pylint: disable=literal-comparison
                    lambda state: state.get_shape(), policy_state)]))

    if outer_rank == 2:
      best_actions = tf.nest.map_structure(
          lambda x: tf.reshape(  # pylint: disable=g-long-lambda
              x, [-1, seq_size, self._num_elites, tf.shape(x)[-1]]),
          best_actions)
      best_scores = tf.reshape(best_scores, [-1, seq_size, self._num_elites])

    return best_actions, best_scores, best_next_policy_state

  def compute_target_q(self,
                       observation: types.NestedTensorOrArray,
                       action: types.NestedTensorOrArray,
                       step_type: Optional[tf.Tensor] = None,
                       policy_state: Sequence[tf.Tensor] = ()
                       ) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    scores, next_policy_state = self._q_network((observation, action),
                                                step_type=step_type,
                                                network_state=policy_state,
                                                training=self._training)  # [B]
    return scores, next_policy_state

  def _score(
      self, observation, sample_actions, step_type=None, policy_state=()
      ) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    """Scores the sample actions internally as part of CEM.

    Args:
      observation: A batch of observation tensors or NamedTuples, whatever the
        q_func will handle. CEM is agnostic to it.
      sample_actions: A [B, N, A] sized tensor, where batch is the batch size, N
        is the sample size for the CEM, a is the size of the action space.
      step_type: A `StepType` enum value.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.

    Returns:
      a tensor of shape [B, N] representing the scores for the actions.
    """

    def expand_to_megabatch(feature):
      # Collapse second dimension of megabatch.
      dim = tf.shape(feature)[2]
      return tf.reshape(feature, [-1, dim])

    if self._preprocess_state_action:
      # [B, N, A] -> [BxN, A]
      sample_actions = tf.nest.map_structure(
          expand_to_megabatch, sample_actions)
      # TODO(b/138331671) Move tf.contrib.seq2seq.tile_batch to utils.common
      # [B, ...] -> [BxN, ...]
      observation = nest_utils.tile_batch(observation, self._num_samples)
      step_type = nest_utils.tile_batch(step_type, self._num_samples)
      policy_state = nest_utils.tile_batch(policy_state, self._num_samples)

    scores, next_policy_state = self.compute_target_q(
        observation, sample_actions, step_type, policy_state)  # [BxN]

    if self._preprocess_state_action:
      next_policy_state = tf.nest.map_structure(
          lambda x: tf.reshape(x, [-1, self._num_samples] + x.shape.as_list(  # pylint:disable=g-long-lambda
          )[1:])[:, 0, ...], next_policy_state)

    scores = tf.reshape(scores, [-1, self._num_samples])  # [B, N]

    return scores, next_policy_state

  def _score_with_time(self,
                       observation: types.NestedTensorOrArray,
                       sample_actions: types.NestedTensorOrArray,
                       step_type: Optional[tf.Tensor],
                       policy_state: Sequence[tf.Tensor],
                       seq_size: tf.Tensor
                       ) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    """Scores the sample actions internally as part of CEM.

    Args:
      observation: A batch of state tensors or NamedTuples, whatever the q_func
        will handle. CEM is agnostic to it.
      sample_actions: A [BxT, N, A] sized tensor, where batch is the batch size,
        N is the sample size for the CEM, a is the size of the action space.
      step_type: A `StepType` enum value.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seq_size: Size for the time dimension.

    Returns:
      a tensor of shape [BxT, N] representing the scores for the actions.
    """

    # Reshape input to be compatible with network: [BxN, T, A]

    def expand_to_megabatch(feature):
      # Collapse second dimension of megabatch.
      dim = tf.shape(feature)[2]
      dim_sample = tf.shape(feature)[1]
      feature = tf.reshape(feature, [-1, seq_size, dim_sample, dim])
      feature = tf.transpose(feature, [0, 2, 1, 3])
      return tf.reshape(feature, [-1, seq_size, dim])

    def decouple_batch_time(feature):
      dim = tf.shape(feature)[2]
      dim_sample = tf.shape(feature)[1]
      return tf.reshape(feature, [-1, seq_size, dim_sample, dim])

    if self._preprocess_state_action:
      # [BxT, N, A] -> [BxN, T, A]
      sample_actions = tf.nest.map_structure(
          expand_to_megabatch, sample_actions)
      # TODO(b/138331671) Move tf.contrib.seq2seq.tile_batch to utils.common
      # [B, T, ...] -> [BxN, T, ...]
      observation = nest_utils.tile_batch(observation, self._num_samples)
      step_type = nest_utils.tile_batch(step_type, self._num_samples)
      policy_state = nest_utils.tile_batch(policy_state, self._num_samples)
    else:
      # [BxT, N, A] -> [B, T, N, A]
      sample_actions = tf.nest.map_structure(
          decouple_batch_time, sample_actions)

    scores, next_policy_state = self.compute_target_q(
        observation, sample_actions, step_type, policy_state)  # [BxN, T]

    if self._preprocess_state_action:
      next_policy_state = tf.nest.map_structure(
          lambda x: tf.reshape(x, [-1, self._num_samples] + x.shape.as_list(  # pylint:disable=g-long-lambda
          )[1:])[:, 0, ...], next_policy_state)

    scores = tf.reshape(scores, [-1, self._num_samples, seq_size])  # [B, N, T]
    scores = tf.transpose(scores, [0, 2, 1])  # [B, T, N]
    scores = tf.reshape(scores, [-1, self._num_samples])  # [BxT, N]

    return scores, next_policy_state

  def _variables(self):
    return self._q_network.variables

  def _distribution(self,
                    time_step: ts.TimeStep,
                    policy_state: Sequence[tf.Tensor]
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
    """
    if not policy_state and self._q_network and self._q_network.state_spec:
      batch_size = nest_utils.get_outer_shape(
          time_step.observation, self._observation_spec)[0]
      policy_state = self.get_initial_state(batch_size=batch_size)
    best_actions, best_scores, best_next_policy_state = self.actor_func(
        time_step.observation, time_step.step_type, policy_state)
    def select_best_action(actions):
      best_action = actions[..., 0, :]  # [B, A]
      return best_action

    best_action = tf.nest.map_structure(select_best_action, best_actions)

    best_action_consider_actor = best_action
    best_score_consider_actor = best_scores[..., 0]
    if self._actor_policy is not None:
      potential_best_action = self._actor_policy.action(time_step).action
      potential_best_q, _ = self.compute_target_q(
          time_step.observation, potential_best_action)
      use_cem = tf.cast(
          best_score_consider_actor > potential_best_q, tf.float32)
      best_score_consider_actor = (
          best_score_consider_actor * use_cem +
          (tf.ones_like(use_cem, tf.float32) - use_cem) * potential_best_q)

      def select_best_action_consider_actor(action1, action2):
        use_cem_expanded = tf.expand_dims(
            tf.cast(use_cem, action1.dtype), axis=-1)
        return (action1 * use_cem_expanded +
                action2 * (tf.ones_like(use_cem_expanded, action1.dtype)
                           - use_cem_expanded))
      best_action_consider_actor = tf.nest.map_structure(
          select_best_action_consider_actor,
          best_action_consider_actor, potential_best_action)

    distribution = tf.nest.map_structure(tfp.distributions.Deterministic,
                                         best_action_consider_actor)
    if self._info_spec and 'target_q' in self._info_spec:
      batch_size = nest_utils.get_outer_shape(
          time_step, self._time_step_spec)[0]
      info = tf.nest.map_structure(
          lambda spec: tf.zeros(tf.concat([[batch_size], spec.shape], axis=-1)),
          self._info_spec)
      info['target_q'] = best_score_consider_actor
    else:
      batch_size = nest_utils.get_outer_shape(
          time_step, self._time_step_spec)[0]
      info = tf.nest.map_structure(
          lambda spec: tf.zeros(tf.concat([[batch_size], spec.shape], axis=-1)),
          self._info_spec)
    return policy_step.PolicyStep(distribution, best_next_policy_state, info)
