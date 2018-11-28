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

"""Simple Actor Policy based on an actor network.

This is used in e.g. actor-critic algorithms like DDPG.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.environments import time_step as ts
from tf_agents.networks import network
from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tensorflow.python.ops import template  # TF internal

nest = tf.contrib.framework.nest
tfd = tfp.distributions


# TODO(sfishman): Remove old ActorPolicy and rename this one to ActorPolicy
# once all clients have migrated off the old one.
class ActorPolicyKeras(tf_policy.Base):
  """Class to build Actor Policies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               actor_network=None,
               info_spec=(),
               observation_normalizer=None,
               clip=True):
    """Builds an Actor Policy given a actor network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: An instance of a tf_agents.networks.network.Network, with
        call(observation, step_type).
      info_spec: A nest of TensorSpec representing the policy info.
      observation_normalizer: An object to use for obervation normalization.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.

    Raises:
      ValueError: if actor_network is not of type network.Network.
    """
    if not isinstance(actor_network, network.Network):
      raise ValueError('actor_network must be a network.Network. Found '
                       '{}.'.format(type(actor_network)))
    self._actor_network = actor_network
    self._observation_normalizer = observation_normalizer
    self._clip = clip

    super(ActorPolicyKeras, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=actor_network.state_spec,
        info_spec=info_spec)

  def _apply_actor_network(self, time_step, policy_state):
    observation = time_step.observation
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    return self._actor_network(observation, time_step.step_type, policy_state)

  @property
  def observation_normalizer(self):
    return self._observation_normalizer

  def _variables(self):
    return self._actor_network.variables

  def _action(self, time_step, policy_state, seed):
    distribution_step = self.distribution(time_step, policy_state)

    def _sample(dist, action_spec):
      action = dist.sample(seed=seed)
      if self._clip:
        return common.clip_to_spec(action, action_spec)
      return action

    actions = nest.map_structure(_sample, distribution_step.action,
                                 self._action_spec)

    return policy_step.PolicyStep(actions, distribution_step.state,
                                  distribution_step.info)

  def _distribution(self, time_step, policy_state):
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


class ActorPolicy(tf_policy.Base):
  """Class to build Actor Policies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               policy_state_spec=(),
               actor_network=None,
               info_spec=(),
               template_name='actor_policy',
               observation_normalizer=None,
               clip=True):
    """Builds an Actor Policy given a actor network Template or function.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      policy_state_spec: A nest of TensorSpec representing the policy_state.
      actor_network: A Template from tf.make_template or a function. When
        passing a Template the variables will be reused, passing a function it
        will create a new template with a new set variables.  Network should
        return one of the following: 1. a nested tuple of tf.distributions
          objects matching action_spec, or 2. a nested tuple of tf.Tensors
          representing actions.
      info_spec: A nest of TensorSpec representing the policy info.
      template_name: Name to use for the new template. Ignored if actor_network
        is already a template.
      observation_normalizer: An object to use for obervation normalization.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.

    Raises:
      ValueError: if actor_network is not of type callable or
        tensorflow.python.ops.template.Template.
    """
    if isinstance(actor_network, template.Template):
      self._actor_network = actor_network
    elif callable(actor_network):
      self._actor_network = tf.make_template(
          template_name, actor_network, create_scope_now_=True)
    else:
      raise ValueError('Unrecognized type for arg actor_network: %s (%s)' %
                       (actor_network, type(actor_network)))

    self._observation_normalizer = observation_normalizer
    self._clip = clip

    super(ActorPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec)

  def _apply_actor_network(self, time_step):
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(
          time_step.observation)
      time_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount, observation)
    return self._actor_network(time_step, self._action_spec)

  @property
  def observation_normalizer(self):
    return self._observation_normalizer

  def actor_variables(self):
    return self._actor_network.global_variables[:]

  def _variables(self):
    var_list = self._actor_network.global_variables[:]
    if self._observation_normalizer:
      var_list += self._observation_normalizer.variables
    return var_list

  def _action(self, time_step, policy_state, seed):
    distribution_step = self.distribution(time_step, policy_state)
    seed_stream = tfd.SeedStream(seed=seed, salt='actor_policy')

    def _sample(dist, action_spec):
      action = dist.sample(seed=seed_stream())
      if self._clip:
        return common.clip_to_spec(action, action_spec)
      return action

    actions = nest.map_structure(_sample, distribution_step.action,
                                 self._action_spec)

    time_step_batched = nest_utils.get_outer_rank(time_step,
                                                  self._time_step_spec)
    if not time_step_batched:
      actions = nest_utils.unbatch_nested_tensors(actions, self._action_spec)

    return policy_step.PolicyStep(actions, distribution_step.state,
                                  distribution_step.info)

  def _distribution(self, time_step, policy_state):
    # TODO(kbanoop) pass policy_state to networks
    time_step_batched = nest_utils.get_outer_rank(time_step,
                                                  self._time_step_spec)
    if not time_step_batched:
      time_step = nest_utils.batch_nested_tensors(time_step,
                                                  self._time_step_spec)

    # Actor network outputs nested structure of distributions or actions.
    actions_or_distributions = self._apply_actor_network(time_step)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = nest.map_structure(_to_distribution,
                                       actions_or_distributions)
    return policy_step.PolicyStep(distributions, policy_state)
