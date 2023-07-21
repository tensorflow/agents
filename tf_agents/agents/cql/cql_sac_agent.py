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

"""A CQL-SAC Agent.

Implements Conservative Q Learning from

"Conservative Q-Learning for Offline RL"
  Kumar et al., 2020
  https://arxiv.org/abs/2006.04779
"""

from typing import Callable, Dict, NamedTuple, Optional, Text, Tuple, Union

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity

CqlSacLossInfo = NamedTuple('CqlSacLossInfo',
                            [('critic_loss', types.Tensor),
                             ('actor_loss', types.Tensor),
                             ('alpha_loss', types.Tensor),
                             ('cql_loss', types.Tensor),
                             ('cql_alpha', types.Tensor),
                             ('cql_alpha_loss', Optional[types.Tensor])])


@gin.configurable
class CqlSacAgent(sac_agent.SacAgent):
  """A CQL-SAC Agent based on the SAC Agent."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               critic_network: network.Network,
               actor_network: network.Network,
               actor_optimizer: types.Optimizer,
               critic_optimizer: types.Optimizer,
               alpha_optimizer: types.Optimizer,
               cql_alpha: Union[types.Float, tf.Variable],
               num_cql_samples: int,
               include_critic_entropy_term: bool,
               use_lagrange_cql_alpha: bool,
               cql_alpha_learning_rate: Union[types.Float, tf.Variable] = 1e-4,
               cql_tau: Union[types.Float, tf.Variable] = 10.0,
               random_seed: Optional[int] = None,
               reward_noise_variance: Union[types.Float, tf.Variable] = 0.0,
               num_bc_steps: int = 0,
               actor_loss_weight: types.Float = 1.0,
               critic_loss_weight: types.Float = 0.5,
               alpha_loss_weight: types.Float = 1.0,
               actor_policy_ctor: Callable[
                   ..., tf_policy.TFPolicy] = actor_policy.ActorPolicy,
               critic_network_2: Optional[network.Network] = None,
               target_critic_network: Optional[network.Network] = None,
               target_critic_network_2: Optional[network.Network] = None,
               target_update_tau: types.Float = 1.0,
               target_update_period: types.Int = 1,
               td_errors_loss_fn: types.LossFn = tf.math.squared_difference,
               gamma: types.Float = 1.0,
               reward_scale_factor: types.Float = 1.0,
               initial_log_alpha: types.Float = 0.0,
               use_log_alpha_in_alpha_loss: bool = True,
               target_entropy: Optional[types.Float] = None,
               gradient_clipping: Optional[types.Float] = None,
               log_cql_alpha_clipping: Optional[Tuple[types.Float,
                                                      types.Float]] = None,
               softmax_temperature: types.Float = 1.0,
               bc_debug_mode: bool = False,
               debug_summaries: bool = False,
               summarize_grads_and_vars: bool = False,
               train_step_counter: Optional[tf.Variable] = None,
               name: Optional[Text] = None):
    """Creates a CQL-SAC Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      critic_network: A function critic_network((observations, actions)) that
        returns the q_values for each observation and action.
      actor_network: A function actor_network(observation, action_spec) that
        returns action distribution.
      actor_optimizer: The optimizer to use for the actor network.
      critic_optimizer: The default optimizer to use for the critic network.
      alpha_optimizer: The default optimizer to use for the alpha variable.
      cql_alpha: The weight on CQL loss. This can be a tf.Variable.
      num_cql_samples: Number of samples for importance sampling in CQL.
      include_critic_entropy_term: Whether to include the entropy term in the
        target for the critic loss.
      use_lagrange_cql_alpha: Whether to use a Lagrange threshold to
        tune cql_alpha during training.
      cql_alpha_learning_rate: The learning rate to tune cql_alpha.
      cql_tau: The threshold for the expected difference in Q-values which
        determines the tuning of cql_alpha.
      random_seed: Optional seed for tf.random.
      reward_noise_variance: The noise variance to introduce to the rewards.
      num_bc_steps: Number of behavioral cloning steps.
      actor_loss_weight: The weight on actor loss.
      critic_loss_weight: The weight on critic loss.
      alpha_loss_weight: The weight on alpha loss.
      actor_policy_ctor: The policy class to use.
      critic_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
        the second critic network during Q learning.  The weights from
        `critic_network` are copied if this is not provided.
      target_critic_network: (Optional.)  A `tf_agents.network.Network` to be
        used as the target critic network during Q learning. Every
        `target_update_period` train steps, the weights from `critic_network`
        are copied (possibly withsmoothing via `target_update_tau`) to `
        target_critic_network`.  If `target_critic_network` is not provided, it
        is created by making a copy of `critic_network`, which initializes a new
        network with the same structure and its own layers and weights.
        Performing a `Network.copy` does not work when the network instance
        already has trainable parameters (e.g., has already been built, or when
        the network is sharing layers with another).  In these cases, it is up
        to you to build a copy having weights that are not shared with the
        original `critic_network`, so that this can be used as a target network.
        If you provide a `target_critic_network` that shares any weights with
        `critic_network`, a warning will be logged but no exception is thrown.
      target_critic_network_2: (Optional.) Similar network as
        target_critic_network but for the critic_network_2. See documentation
        for target_critic_network. Will only be used if 'critic_network_2' is
        also specified.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      td_errors_loss_fn:  A function for computing the elementwise TD errors
        loss.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      initial_log_alpha: Initial value for log_alpha.
      use_log_alpha_in_alpha_loss: A boolean, whether using log_alpha or alpha
        in alpha loss. Certain implementations of SAC use log_alpha as log
        values are generally nicer to work with.
      target_entropy: The target average policy entropy, for updating alpha. The
        default value is negative of the total number of actions.
      gradient_clipping: Norm length to clip gradients.
      log_cql_alpha_clipping: (Minimum, maximum) values to clip log CQL alpha.
      softmax_temperature: Temperature value which weights Q-values before
        the `cql_loss` logsumexp calculation.
      bc_debug_mode: Whether to run a behavioral cloning mode where the critic
        loss only depends on CQL loss. Useful when debugging and checking that
        CQL loss can be driven down to zero.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run. Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.
    """
    super(CqlSacAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        critic_network=critic_network,
        actor_network=actor_network,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        actor_loss_weight=actor_loss_weight,
        critic_loss_weight=critic_loss_weight,
        alpha_loss_weight=alpha_loss_weight,
        actor_policy_ctor=actor_policy_ctor,
        critic_network_2=critic_network_2,
        target_critic_network=target_critic_network,
        target_critic_network_2=target_critic_network_2,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        initial_log_alpha=initial_log_alpha,
        use_log_alpha_in_alpha_loss=use_log_alpha_in_alpha_loss,
        target_entropy=target_entropy,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        name=name)
    self._use_lagrange_cql_alpha = use_lagrange_cql_alpha
    if self._use_lagrange_cql_alpha:
      self._log_cql_alpha = tf.Variable(tf.math.log(cql_alpha), trainable=True)
      self._cql_tau = cql_tau
      self._cql_alpha_optimizer = tf.keras.optimizers.Adam(
          learning_rate=cql_alpha_learning_rate)
    else:
      self._cql_alpha = cql_alpha

    self._num_cql_samples = num_cql_samples
    self._include_critic_entropy_term = include_critic_entropy_term
    self._action_seed_stream = tfp.util.SeedStream(
        seed=random_seed, salt='random_actions')
    self._reward_seed_stream = tfp.util.SeedStream(
        seed=random_seed, salt='random_reward_noise')
    self._reward_noise_variance = reward_noise_variance
    self._num_bc_steps = num_bc_steps
    self._log_cql_alpha_clipping = log_cql_alpha_clipping
    self._softmax_temperature = softmax_temperature
    self._bc_debug_mode = bc_debug_mode

  def _check_action_spec(self, action_spec):
    super(CqlSacAgent, self)._check_action_spec(action_spec)

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError(
          'Only single action specs are supported now, but action spec is: {}'
          .format(action_spec))

  def _train(self, experience, weights):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    transition = self._as_transition(experience)
    time_steps, policy_steps, next_time_steps = transition
    actions = policy_steps.action

    trainable_critic_variables = list(object_identity.ObjectIdentitySet(
        self._critic_network_1.trainable_variables +
        self._critic_network_2.trainable_variables))

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self._critic_loss_with_optional_entropy_term(
          time_steps,
          actions,
          next_time_steps,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights,
          training=True)
      critic_loss *= self._critic_loss_weight

      cql_alpha = self._get_cql_alpha()
      cql_loss = self._cql_loss(time_steps, actions, training=True)

      if self._bc_debug_mode:
        cql_critic_loss = cql_loss * cql_alpha
      else:
        cql_critic_loss = critic_loss + (cql_loss * cql_alpha)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    tf.debugging.check_numerics(cql_loss, 'CQL loss is inf or nan.')
    critic_grads = tape.gradient(cql_critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self._actor_loss_weight * self.actor_loss(
          time_steps, actions=actions, weights=weights)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    alpha_variable = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert alpha_variable, 'No alpha variable to optimize.'
      tape.watch(alpha_variable)
      alpha_loss = self._alpha_loss_weight * self.alpha_loss(
          time_steps, weights=weights)
    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_grads = tape.gradient(alpha_loss, alpha_variable)
    self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

    # Based on the equation (24), which automates CQL alpha with the "budget"
    # parameter tau. CQL(H) is now CQL-Lagrange(H):
    # ```
    # min_Q max_{alpha >= 0} alpha * (log_sum_exp(Q(s, a')) - Q(s, a) - tau)
    # ```
    # If the expected difference in Q-values is less than tau, alpha
    # will adjust to be closer to 0. If the difference is higher than tau,
    # alpha is likely to take on high values and more aggressively penalize
    # Q-values.
    cql_alpha_loss = tf.constant(0.)
    if self._use_lagrange_cql_alpha:
      cql_alpha_variable = [self._log_cql_alpha]
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(cql_alpha_variable)
        cql_alpha_loss = -self._get_cql_alpha() * (cql_loss - self._cql_tau)
      tf.debugging.check_numerics(cql_alpha_loss,
                                  'CQL alpha loss is inf or nan.')
      cql_alpha_gradients = tape.gradient(cql_alpha_loss, cql_alpha_variable)
      self._apply_gradients(cql_alpha_gradients, cql_alpha_variable,
                            self._cql_alpha_optimizer)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss', data=alpha_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='cql_loss', data=cql_loss, step=self.train_step_counter)
      if self._use_lagrange_cql_alpha:
        tf.compat.v2.summary.scalar(
            name='cql_alpha_loss',
            data=cql_alpha_loss,
            step=self.train_step_counter)
    tf.compat.v2.summary.scalar(
        name='cql_alpha', data=cql_alpha, step=self.train_step_counter)
    tf.compat.v2.summary.scalar(
        name='sac_alpha', data=tf.exp(self._log_alpha),
        step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = cql_critic_loss + actor_loss + alpha_loss

    extra = CqlSacLossInfo(
        critic_loss=critic_loss,
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        cql_loss=cql_loss,
        cql_alpha=cql_alpha,
        cql_alpha_loss=cql_alpha_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)

  def _transpose_tile_and_batch_dims(
      self, original_tensor: types.Tensor) -> types.Tensor:
    """Transposes [tile, batch, ...] to [batch, tile, ...]."""
    return tf.transpose(
        original_tensor, [1, 0] + list(range(2, len(original_tensor.shape))))

  def _actions_and_log_probs(
      self,
      time_steps: ts.TimeStep,
      training: Optional[bool] = False) -> Tuple[types.Tensor, types.Tensor]:
    """Get actions and corresponding log probabilities from policy."""
    # Get raw action distribution from policy, and initialize bijectors list.
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._train_policy.get_initial_state(batch_size)
    if training:
      action_distribution = self._train_policy.distribution(
          time_steps, policy_state=policy_state).action
    else:
      action_distribution = self._policy.distribution(
          time_steps, policy_state=policy_state).action

    # Sample actions and log_pis from transformed distribution.
    actions = tf.nest.map_structure(
        lambda d: d.sample((), seed=self._action_seed_stream()),
        action_distribution)
    log_pi = common.log_probability(action_distribution, actions,
                                    self.action_spec)

    return actions, log_pi

  def _sample_and_transpose_actions_and_log_probs(
      self,
      time_steps: ts.TimeStep,
      num_action_samples: int,
      training: Optional[bool] = False) -> Tuple[types.Tensor, types.Tensor]:
    """Samples actions and corresponding log probabilities from policy."""
    # Get raw action distribution from policy, and initialize bijectors list.
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    policy_state = self._train_policy.get_initial_state(batch_size)
    if training:
      action_distribution = self._train_policy.distribution(
          time_steps, policy_state=policy_state).action
    else:
      action_distribution = self._policy.distribution(
          time_steps, policy_state=policy_state).action

    actions = tf.nest.map_structure(
        lambda d: d.sample(num_action_samples, seed=self._action_seed_stream()),
        action_distribution)
    log_pi = common.log_probability(
        action_distribution, actions, self.action_spec)

    # Swap the first two axes for a [batch, self._num_cql_samples, ...] shape.
    actions = self._transpose_tile_and_batch_dims(actions)
    log_pi = self._transpose_tile_and_batch_dims(log_pi)
    return actions, log_pi

  def _flattened_multibatch_tensor(
      self, original_tensor: types.Tensor) -> types.Tensor:
    """Flattens the batch and tile dimensions into a single dimension.

    Args:
      original_tensor: Input tensor of shape [batch_size, tile, dim].

    Returns:
      Flattened tensor with the outer dimension (batch_size * tile).
    """
    spec = tf.TensorSpec(
        shape=original_tensor.shape[2:], dtype=original_tensor.dtype)
    flattened_tensor, _ = nest_utils.flatten_multi_batched_nested_tensors(
        original_tensor, spec)
    return flattened_tensor

  def _get_q_values(
      self, target_input: Tuple[types.Tensor,
                                types.Tensor], step_type: types.Tensor,
      reshape_batch_size: Optional[int],
      training: Optional[bool] = False) -> Tuple[types.Tensor, types.Tensor]:
    """Gets the Q-values of target_input.

    Uses the smaller of the critic network outputs since learned Q functions
    can overestimate Q-values.

    Args:
      target_input: Tuple of (observation, sampled actions) tensors.
      step_type: `Tensor` of `StepType` enum values.
      reshape_batch_size: Batch size to reshape the Q values to [batch_size,
        self._num_cql_samples, ...]. If None, do not reshape.
      training: Whether training should be applied.

    Returns:
      Tuple[`Tensor`, `Tensor`] of Q-values.
    """
    q_values1, _ = self._critic_network_1(
        target_input, step_type, training=training)
    q_values2, _ = self._critic_network_2(
        target_input, step_type, training=training)

    # Optionally reshape to [batch_size, num_cql_samples, q_value_dim].
    if reshape_batch_size is not None:
      reshaped_dims = [reshape_batch_size, self._num_cql_samples] + (
          q_values1.shape.as_list()[1:])
      q_values1 = tf.reshape(q_values1, reshaped_dims)
      q_values2 = tf.reshape(q_values2, reshaped_dims)

    q_values1 = tf.expand_dims(q_values1, axis=-1)
    q_values2 = tf.expand_dims(q_values2, axis=-1)
    return q_values1, q_values2

  def _cql_loss_debug_summaries(self, debug_summaries_dict: Dict[str,
                                                                 types.Tensor]):
    """Generates summaries for _cql_loss."""
    if self._debug_summaries:
      with tf.name_scope('cql_loss'):
        for key in debug_summaries_dict:
          tf.compat.v2.summary.scalar(
              name=key,
              data=debug_summaries_dict[key],
              step=self.train_step_counter)

  def _cql_loss(self, time_steps: ts.TimeStep, actions: types.Tensor,
                training: Optional[bool] = False) -> types.Tensor:
    """Computes CQL loss for SAC training in continuous action spaces.

    Extends the standard critic loss to minimize Q-values sampled from a policy
    and maximize values of the dataset actions.

    Based on the `CQL(H)` equation (4) in (Kumar et al., 2020):

    ```
    log_sum_exp(Q(s, a')) - Q(s, a)
    ```

    Other variants of CQL-SAC, such as `CQL(R)`, can override this function.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      training: Whether training should be applied.

    Returns:
      cql_loss: A scalar CQL loss.
    """
    # Tile tensors to match CQL action sampling.
    # Shape [batch_size * self._num_cql_samples, ...]
    observation = nest_utils.tile_batch(
        time_steps.observation, self._num_cql_samples)
    step_type = nest_utils.tile_batch(
        time_steps.step_type, self._num_cql_samples)

    # Sample self._num_cql_samples from the policy distribution.
    # We do not update actor during CQL loss.
    sampled_actions, sampled_actions_log_probs = (
        self._sample_and_transpose_actions_and_log_probs(
            time_steps, self._num_cql_samples, training=False))
    # Shape [batch_size * self._num_cql_samples, ...]
    sampled_actions = self._flattened_multibatch_tensor(sampled_actions)
    target_input = (observation, sampled_actions)
    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
    q_estimates1, q_estimates2 = self._get_q_values(
        target_input,
        step_type,
        reshape_batch_size=batch_size,
        training=False)
    debug_summaries_dict = {}
    debug_summaries_dict['q_estimates1'] = tf.reduce_mean(q_estimates1)
    debug_summaries_dict['q_estimates2'] = tf.reduce_mean(q_estimates2)

    # We're supposed to be taking an unweighted sum of Q-values of actions
    # from the policy and uniform distributions. We correct for the fact that
    # we're drawing samples from either distribution by subtracting the
    # corresponding log probability.
    # Based on Appendix F:
    # ```
    # q_value_unif = \sum_{a_i~\Unif(a)}^{N} [exp(Q(s, a_i)) / \Unif(a)]
    # q_value_pi =   \sum_{a_i~\pi(a|s)}^{N} [exp(Q(s, a_i)) / \pi(a|s)]
    # log_sum_exp(Q(s, a')) = log((1/2N * q_value_unif) + (1/2N  * q_value_pi))
    # ```
    policy_log_probs1 = (q_estimates1 * self._softmax_temperature
                        ) - sampled_actions_log_probs[..., None]
    policy_log_probs2 = (q_estimates2 * self._softmax_temperature
                        ) - sampled_actions_log_probs[..., None]

    # Sample self._num_cql_samples from the uniform-at-random distribution.
    flattened_action_spec = tf.nest.flatten(self.action_spec)[0]
    uniform_actions = tf.random.uniform(
        tf.shape(sampled_actions),
        minval=flattened_action_spec.minimum,
        maxval=flattened_action_spec.maximum,
        seed=self._action_seed_stream())
    target_input = (observation, uniform_actions)
    q_uniform1, q_uniform2 = self._get_q_values(target_input, step_type,
                                                batch_size, training=False)
    debug_summaries_dict['q_uniform1'] = tf.reduce_mean(q_uniform1)
    debug_summaries_dict['q_uniform2'] = tf.reduce_mean(q_uniform2)

    # Uniform density is `(1/range)^dimension`, so the log probability of the
    # uniform distribution is `-log(range)*dimension`.
    # Once again, we subtract this from the Q-value to correct for drawing
    # from this distribution and contribute to an unweighted sum of Q-values.
    uniform_actions_log_probs = tf.reduce_sum(-tf.math.log(
        flattened_action_spec.maximum -
        flattened_action_spec.minimum)) * flattened_action_spec.shape[0]
    uniform_log_probs1 = (q_uniform1 *
                          self._softmax_temperature) - uniform_actions_log_probs
    uniform_log_probs2 = (q_uniform2 *
                          self._softmax_temperature) - uniform_actions_log_probs

    # Importance sampled estimate of the Q-value sum. We do this since we
    # can't tractably compute the exact Q-value in a continuous action space.
    # Based on the first part of equation (4):
    # ```log_sum_exp(Q(s, a'))```
    # Concatenates and collapses along the self._num_cql_samples dimension.
    combined_log_probs1 = tf.concat([policy_log_probs1, uniform_log_probs1],
                                    axis=1)
    combined_log_probs2 = tf.concat([policy_log_probs2, uniform_log_probs2],
                                    axis=1)

    logsumexp1 = tf.math.reduce_logsumexp(
        combined_log_probs1, axis=1) * 1.0 / self._softmax_temperature
    logsumexp2 = tf.math.reduce_logsumexp(
        combined_log_probs2, axis=1) * 1.0 / self._softmax_temperature

    target_input = (time_steps.observation, actions)
    q_original1, q_original2 = self._get_q_values(
        target_input,
        time_steps.step_type,
        reshape_batch_size=None,
        training=training)
    debug_summaries_dict['q_original1'] = tf.reduce_mean(q_original1)
    debug_summaries_dict['q_original2'] = tf.reduce_mean(q_original2)

    cql_loss = tf.reduce_mean((logsumexp1 - q_original1) +
                              (logsumexp2 - q_original2)) / 2.0

    self._cql_loss_debug_summaries(debug_summaries_dict)

    return cql_loss

  @common.function(autograph=True)
  def actor_loss(self,
                 time_steps: ts.TimeStep,
                 actions: types.Tensor,
                 weights: Optional[types.Tensor] = None,
                 training: Optional[bool] = True) -> types.Tensor:
    """Computes actor_loss equivalent to the SAC actor_loss.

    Uses behavioral cloning for the first `self._num_bc_steps` of training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      training: Whether training should be applied.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)

      sampled_actions, sampled_log_pi = self._actions_and_log_probs(
          time_steps, training=training)

      # Behavioral cloning: train the policy to reproduce actions from
      # the dataset.
      if self.train_step_counter < self._num_bc_steps:
        distribution, _ = self._actor_network(time_steps.observation,
                                              time_steps.step_type, ())
        actor_log_prob = distribution.log_prob(actions)
        actor_loss = tf.exp(self._log_alpha) * sampled_log_pi - actor_log_prob
        target_q_values = tf.zeros(tf.shape(sampled_log_pi))
      else:
        target_input = (time_steps.observation, sampled_actions)
        target_q_values1, _ = self._critic_network_1(
            target_input, time_steps.step_type, training=False)
        target_q_values2, _ = self._critic_network_2(
            target_input, time_steps.step_type, training=False)
        target_q_values = tf.minimum(target_q_values1, target_q_values2)
        actor_loss = tf.exp(self._log_alpha) * sampled_log_pi - target_q_values

      if actor_loss.shape.rank > 1:
        # Sum over the time dimension.
        actor_loss = tf.reduce_sum(
            actor_loss, axis=range(1, actor_loss.shape.rank))
      reg_loss = self._actor_network.losses if self._actor_network else None
      agg_loss = common.aggregate_losses(
          per_example_loss=actor_loss,
          sample_weight=weights,
          regularization_loss=reg_loss)
      actor_loss = agg_loss.total_loss
      self._actor_loss_debug_summaries(actor_loss, sampled_actions,
                                       sampled_log_pi, target_q_values,
                                       time_steps)

      return actor_loss

  def _get_cql_alpha(self) -> types.Tensor:
    """Returns CQL alpha."""
    if self._use_lagrange_cql_alpha:
      log_cql_alpha = self._log_cql_alpha
      if self._log_cql_alpha_clipping is not None:
        log_cql_alpha = tf.clip_by_value(
            log_cql_alpha,
            clip_value_min=self._log_cql_alpha_clipping[0],
            clip_value_max=self._log_cql_alpha_clipping[1])
      cql_alpha = tf.math.exp(log_cql_alpha)
      return cql_alpha
    else:
      return tf.convert_to_tensor(self._cql_alpha)

  def _critic_loss_with_optional_entropy_term(
      self,
      time_steps: ts.TimeStep,
      actions: types.Tensor,
      next_time_steps: ts.TimeStep,
      td_errors_loss_fn: types.LossFn,
      gamma: types.Float = 1.0,
      reward_scale_factor: types.Float = 1.0,
      weights: Optional[types.Tensor] = None,
      training: bool = False) -> types.Tensor:
    r"""Computes the critic loss for CQL-SAC training.

    The original SAC critic loss is:
    ```
    (q(s, a) - (r(s, a) + \gamma q(s', a') - \gamma \alpha \log \pi(a'|s')))^2
    ```

    The CQL-SAC critic loss makes the entropy term optional.
    CQL may value unseen actions higher since it lower-bounds the value of
    seen actions. This makes the policy entropy potentially redundant in the
    target term, since it will further enhance unseen actions' effects.

    If self._include_critic_entropy_term is False, this loss equation becomes:
    ```
    (q(s, a) - (r(s, a) + \gamma q(s', a')))^2
    ```

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute
        elementwise (per-batch-entry) loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      training: Whether this loss is being used for training.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      nest_utils.assert_same_structure(actions, self.action_spec)
      nest_utils.assert_same_structure(time_steps, self.time_step_spec)
      nest_utils.assert_same_structure(next_time_steps, self.time_step_spec)

      # We do not update actor or target networks in critic loss.
      next_actions, next_log_pis = self._actions_and_log_probs(
          next_time_steps, training=False)
      target_input = (next_time_steps.observation, next_actions)
      target_q_values1, unused_network_state1 = self._target_critic_network_1(
          target_input, next_time_steps.step_type, training=False)
      target_q_values2, unused_network_state2 = self._target_critic_network_2(
          target_input, next_time_steps.step_type, training=False)
      target_q_values = tf.minimum(target_q_values1, target_q_values2)

      if self._include_critic_entropy_term:
        target_q_values -= (tf.exp(self._log_alpha) * next_log_pis)

      reward = next_time_steps.reward
      if self._reward_noise_variance > 0:
        reward_noise = tf.random.normal(
            tf.shape(reward),
            0.0,
            self._reward_noise_variance,
            seed=self._reward_seed_stream())
        reward += reward_noise

      td_targets = tf.stop_gradient(reward_scale_factor * reward + gamma *
                                    next_time_steps.discount * target_q_values)

      pred_input = (time_steps.observation, actions)
      pred_td_targets1, _ = self._critic_network_1(
          pred_input, time_steps.step_type, training=training)
      pred_td_targets2, _ = self._critic_network_2(
          pred_input, time_steps.step_type, training=training)
      critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
      critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
      critic_loss = critic_loss1 + critic_loss2

      if critic_loss.shape.rank > 1:
        # Sum over the time dimension.
        critic_loss = tf.reduce_sum(
            critic_loss, axis=range(1, critic_loss.shape.rank))

      agg_loss = common.aggregate_losses(
          per_example_loss=critic_loss,
          sample_weight=weights,
          regularization_loss=(self._critic_network_1.losses +
                               self._critic_network_2.losses))
      critic_loss = agg_loss.total_loss

      self._critic_loss_debug_summaries(td_targets, pred_td_targets1,
                                        pred_td_targets2)

      return critic_loss
