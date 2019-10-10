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

"""Implements the Linear Thompson Sampling bandit algorithm.

  Reference:
  "Thompson Sampling for Contextual Bandits with Linear Payoffs",
  Shipra Agrawal, Navin Goyal, ICML 2013. The actual algorithm implemented is
  `Algorithm 3` from the supplementary material of the paper from
  `http://proceedings.mlr.press/v28/agrawal13-supp.pdf`.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.policies import linear_thompson_sampling_policy as ts_policy
from tf_agents.utils import nest_utils


@gin.configurable
class LinearThompsonSamplingAgent(tf_agent.TFAgent):
  """Linear Thompson Sampling Agent.

  Implements the Linear Thompson Sampling Agent from the following paper:
  "Thompson Sampling for Contextual Bandits with Linear Payoffs",
  Shipra Agrawal, Navin Goyal, ICML 2013. The actual algorithm implemented is
  `Algorithm 3` from the supplementary material of the paper from
  `http://proceedings.mlr.press/v28/agrawal13-supp.pdf`.

  In a nutshell, the agent maintains two parameters `weight_covariances` and
  `parameter_estimators`, and updates them based on experience. The inverse of
  the weight covariance parameters are updated with the outer product of the
  observations using the Woodbury inverse matrix update, while the parameter
  estimators are updated by the reward-weighted observation vectors for every
  action.
  """

  def __init__(self,
               time_step_spec,
               action_spec,
               gamma=1.0,
               observation_and_action_constraint_splitter=None,
               dtype=tf.float32,
               name=None):
    """Initialize an instance of `LinearThompsonSamplingAgent`.

    Args:
      time_step_spec: A `TimeStep` spec describing the expected `TimeStep`s.
      action_spec: A scalar `BoundedTensorSpec` with `int32` or `int64` dtype
        describing the number of actions for this agent.
      gamma: a float forgetting factor in [0.0, 1.0]. When set to
        1.0, the algorithm does not forget.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      dtype: The type of the parameters stored and updated by the agent. Should
        be one of `tf.float32` and `tf.float64`. Defaults to `tf.float32`.
      name: a name for this instance of `LinearThompsonSamplingAgent`.

    Raises:
      ValueError if dtype is not one of `tf.float32` or `tf.float64`.
    """
    tf.Module.__init__(self, name=name)
    self._num_actions = bandit_utils.get_num_actions_from_tensor_spec(
        action_spec)
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)
    if observation_and_action_constraint_splitter:
      context_shape = observation_and_action_constraint_splitter(
          time_step_spec.observation)[0].shape.as_list()
    else:
      context_shape = time_step_spec.observation.shape.as_list()
    self._context_dim = (
        tf.compat.dimension_value(context_shape[0]) if context_shape else 1)
    self._gamma = gamma
    if self._gamma < 0.0 or self._gamma > 1.0:
      raise ValueError('Forgetting factor `gamma` must be in [0.0, 1.0].')

    self._weight_covariances = []
    self._parameter_estimators = []
    self._dtype = dtype
    if dtype not in (tf.float32, tf.float64):
      raise ValueError(
          'Agent dtype should be either `tf.float32 or `tf.float64`.')

    for k in range(self._num_actions):
      self._weight_covariances.append(
          tf.compat.v2.Variable(
              tf.eye(self._context_dim, dtype=dtype), name='a_' + str(k)))
      self._parameter_estimators.append(
          tf.compat.v2.Variable(
              tf.zeros(self._context_dim, dtype=dtype), name='b_' + str(k)))

    policy = ts_policy.LinearThompsonSamplingPolicy(
        action_spec,
        time_step_spec,
        self._weight_covariances,
        self._parameter_estimators,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter))
    super(LinearThompsonSamplingAgent, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=policy.action_spec,
        policy=policy,
        collect_policy=policy,
        train_sequence_length=None)

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def weight_covariances(self):
    return [tf.identity(x) for x in self._weight_covariances]

  @property
  def parameter_estimators(self):
    return [tf.identity(x) for x in self._parameter_estimators]

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  def _train(self, experience, weights=None):
    """Updates the policy based on the data in `experience`.

    Note that `experience` should only contain data points that this agent has
    not previously seen. If `experience` comes from a replay buffer, this buffer
    should be cleared between each call to `train`.

    Args:
      experience: A batch of experience data in the form of a `Trajectory`.
      weights: Unused.

    Returns:
        A `LossInfo` containing the loss *before* the training step is taken.
    """
    del weights  # unused

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
    if self._observation_and_action_constraint_splitter:
      observation, _ = self._observation_and_action_constraint_splitter(
          observation)
    observation = tf.cast(observation, self._dtype)
    reward = tf.cast(reward, self._dtype)

    for k in range(self._num_actions):
      diag_mask = tf.linalg.tensor_diag(
          tf.cast(tf.equal(action, k), self._dtype))
      observations_for_arm = tf.matmul(diag_mask, observation)
      rewards_for_arm = tf.matmul(diag_mask, tf.reshape(reward, [-1, 1]))
      tf.compat.v1.assign(
          self._weight_covariances[k],
          self._gamma * self._weight_covariances[k] + tf.matmul(
              observations_for_arm, observations_for_arm, transpose_a=True))
      tf.compat.v1.assign(
          self._parameter_estimators[k],
          self._gamma * self._parameter_estimators[k] +
          bandit_utils.sum_reward_weighted_observations(rewards_for_arm,
                                                        observations_for_arm))

    batch_size = tf.cast(
        tf.compat.dimension_value(tf.shape(reward)[0]), dtype=tf.int64)
    self._train_step_counter.assign_add(batch_size)

    loss_info = tf_agent.LossInfo(
        loss=(-1. * tf.reduce_sum(experience.reward)), extra=())
    return loss_info
