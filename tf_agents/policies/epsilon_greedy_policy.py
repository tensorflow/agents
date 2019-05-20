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
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.utils import nest_utils

tfd = tfp.distributions


class EpsilonGreedyPolicy(tf_policy.Base):
  """Returns epsilon-greedy samples of a given policy."""

  def __init__(self, policy, epsilon, name=None):
    """Builds an epsilon-greedy MixturePolicy wrapping the given policy.

    Args:
      policy: A policy implementing the tf_policy.Base interface.
      epsilon: The probability of taking the random action represented as a
        float scalar, a scalar Tensor of shape=(), or a callable that returns a
        float scalar or Tensor.
      name: The name of this policy.

    Raises:
      ValueError: If epsilon is invalid.
    """
    self._greedy_policy = greedy_policy.GreedyPolicy(policy)
    self._epsilon = epsilon
    self._random_policy = random_tf_policy.RandomTFPolicy(
        policy.time_step_spec,
        policy.action_spec,
        emit_log_probability=policy.emit_log_probability)
    super(EpsilonGreedyPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=policy.emit_log_probability,
        name=name)

  def _variables(self):
    return self._greedy_policy.variables()

  def _get_epsilon(self):
    if callable(self._epsilon):
      return self._epsilon()
    else:
      return self._epsilon

  def _action(self, time_step, policy_state, seed):
    seed_stream = tfd.SeedStream(seed=seed, salt='epsilon_greedy')
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
    action = tf.compat.v1.where(cond, greedy_action.action,
                                random_action.action)

    if greedy_action.info:
      if not random_action.info:
        raise ValueError('Incompatible info field')
      info = tf.compat.v1.where(cond, greedy_action.info, random_action.info)
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

  def _distribution(self, time_step, policy_states):
    raise NotImplementedError(
        'EpsilonGreedyPolicy does not support distributions yet.')
