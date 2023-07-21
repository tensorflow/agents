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

"""Actor Policy based on an actor network.

This is used in e.g. actor-critic algorithms like DDPG.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import network
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import tensor_normalizer


@gin.configurable
class ActorPolicy(tf_policy.TFPolicy):
  """Class to build Actor Policies."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               actor_network: network.Network,
               policy_state_spec: types.NestedTensorSpec = (),
               info_spec: types.NestedTensorSpec = (),
               observation_normalizer: Optional[
                   tensor_normalizer.TensorNormalizer] = None,
               clip: bool = True,
               training: bool = False,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               name: Optional[Text] = None):
    """Builds an Actor Policy given an actor network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      actor_network: An instance of a `tf_agents.networks.network.Network` to be
        used by the policy. The network will be called with `call(observation,
        step_type, policy_state)` and should return `(actions_or_distributions,
        new_state)`.
      policy_state_spec: A nest of TensorSpec representing the policy_state.
        If not set, defaults to actor_network.state_spec.
      info_spec: A nest of `TensorSpec` representing the policy info.
      observation_normalizer: An object to use for observation normalization.
      clip: Whether to clip actions to spec before returning them. Default True.
        Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      training: Whether the network should be called in training mode.
      observation_and_action_constraint_splitter: A function used to process
        observations with action constraints. These constraints can indicate,
        for example, a mask of valid/invalid actions for a given state of the
        environment.
        The function takes in a full observation and returns a tuple consisting
        of 1) the part of the observation intended as input to the network and
        2) the constraint. An example
        `observation_and_action_constraint_splitter` could be as simple as:
        ```
        def observation_and_action_constraint_splitter(observation):
          return observation['network_input'], observation['constraint']
        ```
        *Note*: when using `observation_and_action_constraint_splitter`, make
        sure the provided `actor_network` is compatible with the
        network-specific half of the output of the
        `observation_and_action_constraint_splitter`. In particular,
        `observation_and_action_constraint_splitter` will be called on the
        observation before passing to the network.
        If `observation_and_action_constraint_splitter` is None, action
        constraints are not applied.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      ValueError: if `actor_network` is not of type `network.Network`.
      NotImplementedError: if `observation_and_action_constraint_splitter` is
        not None but `action_spec` is not discrete.
    """
    time_step_spec = tensor_spec.from_spec(time_step_spec)
    action_spec = tensor_spec.from_spec(action_spec)

    if not isinstance(actor_network, network.Network):
      raise ValueError('actor_network must be a network.Network. Found '
                       '{}.'.format(type(actor_network)))

    # Create variables regardless of if we use the output spec.
    actor_output_spec = actor_network.create_variables(
        time_step_spec.observation)

    if isinstance(actor_network, network.DistributionNetwork):
      actor_output_spec = tf.nest.map_structure(
          lambda o: o.sample_spec, actor_network.output_spec)

    distribution_utils.assert_specs_are_compatible(
        actor_output_spec, action_spec,
        'actor_network output spec does not match action spec')

    self._actor_network = actor_network
    self._observation_normalizer = observation_normalizer
    self._training = training

    if observation_and_action_constraint_splitter is not None:
      if len(tf.nest.flatten(action_spec)) > 1 or (
          not tensor_spec.is_discrete(action_spec)):
        raise NotImplementedError(
            'Action constraints for ActorPolicy are currently only supported '
            'for a single spec of discrete actions. Got action_spec {}'.format(
                action_spec))

    if not policy_state_spec:
      policy_state_spec = actor_network.state_spec

    super(ActorPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        clip=clip,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  def _apply_actor_network(self, observation, step_type, policy_state,
                           mask=None):
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    if mask is None:
      return self._actor_network(
          observation, step_type=step_type, network_state=policy_state,
          training=self._training)
    else:
      return self._actor_network(
          observation, step_type=step_type, network_state=policy_state,
          training=self._training,
          mask=mask)

  @property
  def observation_normalizer(
      self) -> Optional[tensor_normalizer.TensorNormalizer]:
    return self._observation_normalizer

  def _variables(self):
    return self._actor_network.variables

  def _distribution(self, time_step, policy_state):
    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)
    network_observation = time_step.observation
    mask = None

    if observation_and_action_constraint_splitter is not None:
      network_observation, mask = observation_and_action_constraint_splitter(
          network_observation)

    # Actor network outputs nested structure of distributions or actions.
    actions_or_distributions, policy_state = self._apply_actor_network(
        network_observation, step_type=time_step.step_type,
        policy_state=policy_state, mask=mask)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)
    return policy_step.PolicyStep(distributions, policy_state)
