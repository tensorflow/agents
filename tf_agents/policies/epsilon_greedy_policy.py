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

"""Policy implementation that generates epsilon-greedy actions from a policy.

TODO(kbanoop): Make policy state optional in the action method.
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.policies import policy_utilities
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import nest_utils

tfd = tfp.distributions


@gin.configurable(module='tf_agents', blacklist=['policy'])
class EpsilonGreedyPolicy(tf_policy.TFPolicy):
  """Returns epsilon-greedy samples of a given policy."""

  def __init__(self,
               policy: tf_policy.TFPolicy,
               epsilon: types.FloatOrReturningFloat,
               name: Optional[Text] = None):
    """Builds an epsilon-greedy MixturePolicy wrapping the given policy.

    Args:
      policy: A policy implementing the tf_policy.TFPolicy interface.
      epsilon: The probability of taking the random action represented as a
        float scalar, a scalar Tensor of shape=(), or a callable that returns a
        float scalar or Tensor.
      name: The name of this policy.

    Raises:
      ValueError: If epsilon is invalid.
    """
    observation_and_action_constraint_splitter = getattr(
        policy, 'observation_and_action_constraint_splitter', None)
    accepts_per_arm_features = getattr(policy, 'accepts_per_arm_features',
                                       False)
    self._greedy_policy = greedy_policy.GreedyPolicy(policy)
    self._epsilon = epsilon
    self._random_policy = random_tf_policy.RandomTFPolicy(
        policy.time_step_spec,
        policy.action_spec,
        emit_log_probability=policy.emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        accepts_per_arm_features=accepts_per_arm_features,
        info_spec=policy.info_spec)
    super(EpsilonGreedyPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=policy.emit_log_probability,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  @property
  def wrapped_policy(self) -> tf_policy.TFPolicy:
    return self._greedy_policy.wrapped_policy

  def _variables(self):
    return self._greedy_policy.variables()

  def _get_epsilon(self):
    if callable(self._epsilon):
      return self._epsilon()
    else:
      return self._epsilon

  def _action(self, time_step, policy_state, seed):
    seed_stream = tfp.util.SeedStream(seed=seed, salt='epsilon_greedy')
    greedy_action = self._greedy_policy.action(time_step, policy_state)
    random_action = self._random_policy.action(time_step, (), seed_stream())

    outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
    rng = tf.random.uniform(
        outer_shape, maxval=1.0, seed=seed_stream(), name='epsilon_rng')
    cond = tf.greater(rng, self._get_epsilon())

    # Selects the action/info from the random policy with probability epsilon.
    # TODO(b/133175894): tf.compat.v1.where only supports a condition which is
    # either a scalar or a vector. Use tf.compat.v2 so that it can support any
    # condition whose leading dimensions are the same as the other operands of
    # tf.where.
    outer_ndims = int(outer_shape.shape[0])
    if outer_ndims >= 2:
      raise ValueError(
          'Only supports batched time steps with a single batch dimension')
    action = tf.nest.map_structure(lambda g, r: tf.compat.v1.where(cond, g, r),
                                   greedy_action.action, random_action.action)

    if greedy_action.info:
      if not random_action.info:
        raise ValueError('Incompatible info field')
      info = nest_utils.where(cond, greedy_action.info, random_action.info)
      # Overwrite bandit policy info type.
      if policy_utilities.has_bandit_policy_type(info, check_for_tensor=True):
        # Generate mask of the same shape as bandit_policy_type (batch_size, 1).
        # This is the opposite of `cond`, which is 1-D bool tensor (batch_size,)
        # that is true when greedy policy was used, otherwise `cond` is false.
        random_policy_mask = tf.reshape(tf.logical_not(cond),
                                        tf.shape(info.bandit_policy_type))
        bandit_policy_type = policy_utilities.bandit_policy_uniform_mask(
            info.bandit_policy_type, mask=random_policy_mask)
        info = policy_utilities.set_bandit_policy_type(
            info, bandit_policy_type)
    else:
      if random_action.info:
        raise ValueError('Incompatible info field')
      info = ()

    # The state of the epsilon greedy policy is the state of the underlying
    # greedy policy (the random policy carries no state).
    # It is commonly assumed that the new policy state only depends only
    # on the previous state and "time_step", the action (be it the greedy one
    # or the random one) does not influence the new policy state.
    state = greedy_action.state

    return policy_step.PolicyStep(action, state, info)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'EpsilonGreedyPolicy does not support distributions yet.')
