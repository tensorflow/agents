# coding=utf-8
# Copyright 2018 The TFAgents Authors.
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

"""Actor RNN Policy based on an actor RNN network.

This is used in e.g. actor-critic algorithms like DDPG.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments import time_step as ts
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_step
from tf_agents.utils import nest_utils

nest = tf.contrib.framework.nest


class ActorRnnPolicy(actor_policy.ActorPolicy):
  """Class to build Actor RNN Policies."""

  # TODO(b/111118505): get rid of policy_state on reset.
  def reset(self, policy_state=None, batch_size=None):
    """Resets the policy with an optional policy_state.

    Args:
      policy_state: An optional initial Tensor, or a nested dict, list or tuple
        of Tensors representing the initial policy_state.
      batch_size: An optional batch size for network_state.

    Returns:
      A policy_state initial Tensor, or a nested dict, list or tuple of Tensors,
        representing the policy policy_state.
      A reset_op.
    Raises:
      ValueError: If batch_size is not int or None.
    """
    if batch_size is None:
      batch_size = 1
    if not isinstance(batch_size, int):
      raise ValueError('batch_size must be int or None, but got %s of type %s' %
                       (batch_size, type(batch_size)))

    def _zeros_from_spec(spec):
      return tf.zeros(
          shape=[batch_size] + spec.shape.as_list(), dtype=spec.dtype)

    policy_state = nest.map_structure(_zeros_from_spec,
                                      self.policy_state_spec())
    return policy_state, tf.no_op()

  def copy(self):
    observation_normalizer = None
    if self._observation_normalizer:
      observation_normalizer = self._observation_normalizer.copy()

    return ActorRnnPolicy(
        time_step_spec=self.time_step_spec(), action_spec=self.action_spec(),
        policy_state_spec=self.policy_state_spec(),
        actor_network=self._actor_network.func,
        template_name=self._actor_network.name,
        observation_normalizer=observation_normalizer, clip=self._clip)

  def _apply_actor_network(self, time_step, policy_state):
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(
          time_step.observation)
      time_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount, observation)
    return self._actor_network(time_step, self._action_spec,
                               policy_state=policy_state)

  def distribution(self, time_step, policy_state=()):
    """Generates the distribution over next actions given the time_step.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of
        Tensors representing the previous policy_state.

    Returns:
      A possibly nested tuple of tf.distribution objects capturing the
        distribution of next actions.  NOTE: These will be batched regardless
        of whether the input is batched or not.
      A policy_state Tensor, or a nested dict, list or tuple of Tensors,
        representing the new policy state.
    """
    # TODO(kbanoop) pass policy_state to networks
    time_step_batched = nest_utils.get_outer_rank(time_step,
                                                  self._time_step_spec)
    if not time_step_batched:
      time_step = nest_utils.batch_nested_tensors(time_step,
                                                  self._time_step_spec)

    # Actor network outputs nested structure of distributions or actions.
    actions_or_distributions, policy_state = self._apply_actor_network(
        time_step, policy_state)
    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = nest.map_structure(_to_distribution,
                                       actions_or_distributions)
    return policy_step.PolicyStep(distributions, policy_state)
