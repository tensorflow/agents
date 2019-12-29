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

"""Implements the Neural + LinUCB bandit algorithm.

  Applies LinUCB on top of an encoding network.
  Since LinUCB is a linear method, the encoding network is used to capture the
  non-linear relationship between the context features and the expected rewards.
  The encoding network may be already trained or not; if not trained, the
  method can optionally train it using epsilon greedy.

  Reference:
  Carlos Riquelme, George Tucker, Jasper Snoek,
  `Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep
  Networks for Thompson Sampling`, ICLR 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import linear_bandit_agent as linear_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.policies import neural_linucb_policy
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import training as training_lib


class NeuralLinUCBVariableCollection(tf.Module):
  """A collection of variables used by `NeuralLinUCBAgent`."""

  def __init__(self, num_actions, encoding_dim, dtype=tf.float64, name=None):
    """Initializes an instance of `NeuralLinUCBVariableCollection`.

    Args:
      num_actions: (int) number of actions the agent acts on.
      encoding_dim: (int) The dimensionality of the output of the encoding
        network.
      dtype: The type of the variables. Should be one of `tf.float32` and
        `tf.float64`.
      name:  (string) the name of this instance.
    """
    tf.Module.__init__(self, name=name)
    self.actions_from_reward_layer = tf.compat.v2.Variable(
        True, dtype=tf.bool, name='is_action_from_reward_layer')

    self.cov_matrix_list = []
    self.data_vector_list = []
    # We keep track of the number of samples per arm.
    self.num_samples_list = []

    for k in range(num_actions):
      self.cov_matrix_list.append(
          tf.compat.v2.Variable(
              tf.zeros([encoding_dim, encoding_dim], dtype=dtype),
              name='a_{}'.format(k)))
      self.data_vector_list.append(
          tf.compat.v2.Variable(
              tf.zeros(encoding_dim, dtype=dtype), name='b_{}'.format(k)))
      self.num_samples_list.append(
          tf.compat.v2.Variable(
              tf.zeros([], dtype=dtype), name='num_samples_{}'.format(k)))


@gin.configurable
class NeuralLinUCBAgent(tf_agent.TFAgent):
  """An agent implementing the LinUCB algorithm on top of a neural network.
  """

  def __init__(
      self,
      time_step_spec,
      action_spec,
      encoding_network,
      encoding_network_num_train_steps,
      encoding_dim,
      optimizer,
      variable_collection=None,
      alpha=1.0,
      gamma=1.0,
      epsilon_greedy=0.0,
      observation_and_action_constraint_splitter=None,
      # Params for training.
      error_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gradient_clipping=None,
      # Params for debugging.
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=None,
      emit_policy_info=(),
      emit_log_probability=False,
      dtype=tf.float64,
      name=None):
    """Initialize an instance of `NeuralLinUCBAgent`.

    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
      encoding_network: a Keras network that encodes the observations.
      encoding_network_num_train_steps: how many training steps to run for
        training the encoding network before switching to LinUCB. If negative,
        the encoding network is assumed to be already trained.
      encoding_dim: the dimension of encoded observations.
      optimizer: The optimizer to use for training.
      variable_collection: Instance of `NeuralLinUCBVariableCollection`.
        Collection of variables to be updated by the agent. If `None`, a new
        instance of `LinearBanditVariables` will be created. Note that this
        collection excludes the variables owned by the encoding network.
      alpha: (float) positive scalar. This is the exploration parameter that
        multiplies the confidence intervals.
      gamma: a float forgetting factor in [0.0, 1.0]. When set to
        1.0, the algorithm does not forget.
      epsilon_greedy: A float representing the probability of choosing a random
        action instead of the greedy action.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      error_loss_fn: A function for computing the error loss, taking parameters
        labels, predictions, and weights (any function from tf.losses would
        work). The default is `tf.losses.mean_squared_error`.
      gradient_clipping: A float representing the norm length to clip gradients
        (or None for no clipping.)
      debug_summaries: A Python bool, default False. When True, debug summaries
        are gathered.
      summarize_grads_and_vars: A Python bool, default False. When True,
        gradients and network variable summaries are written during training.
      train_step_counter: An optional `tf.Variable` to increment every time the
        train op is run.  Defaults to the `global_step`.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      emit_log_probability: Whether the NeuralLinUCBPolicy emits
        log-probabilities or not. Since the policy is deterministic, the
        probability is just 1.
      dtype: The type of the parameters stored and updated by the agent. Should
        be one of `tf.float32` and `tf.float64`. Defaults to `tf.float64`.
      name: a name for this instance of `NeuralLinUCBAgent`.

    Raises:
      TypeError if variable_collection is not an instance of
        `NeuralLinUCBVariableCollection`.
      ValueError if dtype is not one of `tf.float32` or `tf.float64`.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._num_actions = bandit_utils.get_num_actions_from_tensor_spec(
        action_spec)
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)
    if observation_and_action_constraint_splitter is not None:
      context_shape = observation_and_action_constraint_splitter(
          time_step_spec.observation)[0].shape.as_list()
    else:
      context_shape = time_step_spec.observation.shape.as_list()
    self._context_dim = (
        tf.compat.dimension_value(context_shape[0]) if context_shape else 1)
    self._alpha = alpha
    if variable_collection is None:
      variable_collection = NeuralLinUCBVariableCollection(
          self._num_actions, encoding_dim, dtype)
    elif not isinstance(variable_collection, NeuralLinUCBVariableCollection):
      raise TypeError('Parameter `variable_collection` should be '
                      'of type `NeuralLinUCBVariableCollection`.')
    self._variable_collection = variable_collection
    self._gamma = gamma
    if self._gamma < 0.0 or self._gamma > 1.0:
      raise ValueError('Forgetting factor `gamma` must be in [0.0, 1.0].')
    self._dtype = dtype
    if dtype not in (tf.float32, tf.float64):
      raise ValueError(
          'Agent dtype should be either `tf.float32 or `tf.float64`.')
    self._epsilon_greedy = epsilon_greedy

    reward_layer = tf.keras.layers.Dense(
        self._num_actions,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03),
        use_bias=False,
        activation=None,
        name='reward_layer')

    encoding_network.create_variables()
    self._encoding_network = encoding_network
    self._reward_layer = reward_layer
    self._encoding_network_num_train_steps = encoding_network_num_train_steps
    self._encoding_dim = encoding_dim
    self._optimizer = optimizer
    self._error_loss_fn = error_loss_fn
    self._gradient_clipping = gradient_clipping
    train_step_counter = tf.compat.v1.train.get_or_create_global_step()

    policy = neural_linucb_policy.NeuralLinUCBPolicy(
        encoding_network=self._encoding_network,
        encoding_dim=self._encoding_dim,
        reward_layer=self._reward_layer,
        epsilon_greedy=self._epsilon_greedy,
        actions_from_reward_layer=self.actions_from_reward_layer,
        cov_matrix=self.cov_matrix,
        data_vector=self.data_vector,
        num_samples=self.num_samples,
        time_step_spec=time_step_spec,
        alpha=alpha,
        emit_policy_info=emit_policy_info,
        emit_log_probability=emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter))

    super(NeuralLinUCBAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=policy.action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def actions_from_reward_layer(self):
    return self._variable_collection.actions_from_reward_layer

  @property
  def cov_matrix(self):
    return self._variable_collection.cov_matrix_list

  @property
  def data_vector(self):
    return self._variable_collection.data_vector_list

  @property
  def num_samples(self):
    return self._variable_collection.num_samples_list

  @property
  def alpha(self):
    return self._alpha

  @property
  def variables(self):
    return (self.num_samples + self.cov_matrix + self.data_vector +
            self._encoding_network.trainable_weights +
            self._reward_layer.trainable_weights + [self.train_step_counter])

  @alpha.setter
  def update_alpha(self, alpha):
    return tf.compat.v1.assign(self._alpha, alpha)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def compute_summaries(self, loss):
    with tf.name_scope('Losses/'):
      tf.compat.v2.summary.scalar(
          name='total_loss', data=loss, step=self.train_step_counter)

    if self._summarize_grads_and_vars:
      with tf.name_scope('Variables/'):
        trainable_variables = (
            self._encoding_network.trainable_weights +
            self._reward_layer.trainable_weights)
        for var in trainable_variables:
          tf.compat.v2.summary.histogram(
              name=var.name.replace(':', '_'),
              data=var,
              step=self.train_step_counter)

  def loss(self, observations, actions, rewards, weights=None, training=False):
    """Computes loss for reward prediction training.

    Args:
      observations: A batch of observations.
      actions: A batch of actions.
      rewards: A batch of rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `LossInfo` containing the loss for the training step.
    """
    with tf.name_scope('loss'):
      encoded_observation, _ = self._encoding_network(
          observations, training=training)
      predicted_rewards = self._reward_layer(
          encoded_observation, training=training)
      chosen_actions_predicted_rewards = common.index_with_actions(
          predicted_rewards,
          tf.cast(actions, dtype=tf.int32))

      loss = self._error_loss_fn(rewards,
                                 chosen_actions_predicted_rewards,
                                 weights if weights else 1)
      if self._summarize_grads_and_vars:
        with tf.name_scope('Per_arm_loss/'):
          for k in range(self._num_actions):
            loss_mask_for_arm = tf.cast(tf.equal(actions, k), tf.float32)
            loss_for_arm = self._error_loss_fn(
                rewards,
                chosen_actions_predicted_rewards,
                weights=loss_mask_for_arm)
            tf.compat.v2.summary.scalar(
                name='loss_arm_' + str(k),
                data=loss_for_arm,
                step=self.train_step_counter)

    return tf_agent.LossInfo(loss, extra=())

  def compute_loss_using_reward_layer(
      self, observation, action, reward, weights, training=False):
    """Computes loss using the reward layer.

    Args:
      observation: A batch of observations.
      action: A batch of actions.
      reward: A batch of rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output batch loss will be scaled by these weights, and
        the final scalar loss is the mean of these values.
      training: Whether the loss is being used for training.

    Returns:
      loss: A `LossInfo` containing the loss for the training step.
    """
    # Update the neural network params.
    with tf.GradientTape() as tape:
      loss_info = self.loss(
          observation, action, reward, weights, training=training)
    tf.debugging.check_numerics(loss_info[0], 'Loss is inf or nan')
    if self._summarize_grads_and_vars:
      self.compute_summaries(loss_info.loss)
    variables_to_train = (self._encoding_network.trainable_weights +
                          self._reward_layer.trainable_weights)
    if not variables_to_train:
      raise ValueError('No variable to train in the agent.')

    grads = tape.gradient(loss_info.loss, variables_to_train)
    grads_and_vars = tuple(zip(grads, variables_to_train))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(
          grads_and_vars, self._gradient_clipping)

    if self._summarize_grads_and_vars:
      with tf.name_scope('Reward_network/'):
        eager_utils.add_variables_summaries(grads_and_vars,
                                            self.train_step_counter)
        eager_utils.add_gradients_summaries(grads_and_vars,
                                            self.train_step_counter)

    training_lib.apply_gradients(self._optimizer, grads_and_vars,
                                 global_step=self.train_step_counter)
    return loss_info

  def compute_loss_using_linucb(self, observation, action, reward, weights,
                                training=False):
    """Computes the loss using LinUCB.

    Args:
      observation: A batch of observations.
      action: A batch of actions.
      reward: A batch of rewards.
      weights: unused weights.
      training: Whether the loss is being used to train.

    Returns:
      loss: A `LossInfo` containing the loss for the training step.
    """
    del weights  # unused

    # The network is trained now. Update the covariance matrix.
    encoded_observation, _ = self._encoding_network(
        observation, training=training)
    encoded_observation = tf.cast(encoded_observation, dtype=self._dtype)

    for k in range(self._num_actions):
      diag_mask = tf.linalg.tensor_diag(
          tf.cast(tf.equal(action, k), self._dtype))
      observations_for_arm = tf.matmul(diag_mask, encoded_observation)
      rewards_for_arm = tf.matmul(diag_mask, tf.reshape(reward, [-1, 1]))

      num_samples_for_arm_current = tf.reduce_sum(diag_mask)
      tf.compat.v1.assign_add(self.num_samples[k], num_samples_for_arm_current)
      num_samples_for_arm_total = self.num_samples[k].read_value()

      # Update the matrix A and b.
      # pylint: disable=cell-var-from-loop
      def update(cov_matrix, data_vector):
        a_new, b_new, _, _ = linear_agent.update_a_and_b_with_forgetting(
            cov_matrix, data_vector, rewards_for_arm, observations_for_arm,
            self._gamma)
        return a_new, b_new
      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0,
          lambda: update(self.cov_matrix[k], self.data_vector[k]),
          lambda: (self.cov_matrix[k], self.data_vector[k]))

      tf.compat.v1.assign(self.cov_matrix[k], a_new)
      tf.compat.v1.assign(self.data_vector[k], b_new)

    loss_tensor = tf.cast(-1. * tf.reduce_sum(reward), dtype=tf.float32)
    loss_info = tf_agent.LossInfo(loss=loss_tensor, extra=())
    self.train_step_counter.assign_add(1)
    return loss_info

  def _train(self, experience, weights=None):
    """Updates the policy based on the data in `experience`.

    Note that `experience` should only contain data points that this agent has
    not previously seen. If `experience` comes from a replay buffer, this buffer
    should be cleared between each call to `train`.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`.
      weights: (optional) sample weights.

    Returns:
        A `LossInfo` containing the loss *before* the training step is taken.
        In most cases, if `weights` is provided, the entries of this tuple will
        have been calculated with the weights.  Note that each Agent chooses
        its own method of applying weights.
    """

    # If the experience comes from a replay buffer, the reward has shape:
    #     [batch_size, time_steps]
    # where `time_steps` is the number of driver steps executed in each
    # training loop.
    # We flatten the tensors below in order to reflect the effective batch size.

    reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._time_step_spec.reward)
    action, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.action, self._action_spec)
    observation, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.observation, self._time_step_spec.observation)

    if self._observation_and_action_constraint_splitter is not None:
      observation, _ = self._observation_and_action_constraint_splitter(
          observation)
    observation = tf.cast(observation, self._dtype)
    reward = tf.cast(reward, self._dtype)

    tf.compat.v1.assign(
        self.actions_from_reward_layer,
        tf.less(self._train_step_counter,
                self._encoding_network_num_train_steps))
    def use_actions_from_reward_layer():
      return self.compute_loss_using_reward_layer(
          observation, action, reward, weights, training=True)

    def no_actions_from_reward_layer():
      return self.compute_loss_using_linucb(
          observation, action, reward, weights, training=True)

    loss_info = tf.cond(
        self.actions_from_reward_layer,
        use_actions_from_reward_layer,
        no_actions_from_reward_layer)
    return loss_info
