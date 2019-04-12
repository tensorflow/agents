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

"""An ActorPolicy that also returns policy_info needed for PPO training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.specs import distribution_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

tfd = tfp.distributions


class PPOPolicy(actor_policy.ActorPolicy):
  """An ActorPolicy that also returns policy_info needed for PPO training."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               actor_network=None,
               value_network=None,
               observation_normalizer=None,
               clip=True,
               collect=True):
    """Builds a PPO Policy given network Templates or functions.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: An instance of a tf_agents.networks.network.Network, with
        call(observation, step_type, network_state).  Network should
        return one of the following: 1. a nested tuple of tfp.distributions
          objects matching action_spec, or 2. a nested tuple of tf.Tensors
          representing actions.
      value_network:  An instance of a tf_agents.networks.network.Network, with
        call(observation, step_type, network_state).  Network should return
        value predictions for the input state.
      observation_normalizer: An object to use for obervation normalization.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      collect: If True, creates ops for actions_log_prob, value_preds, and
        action_distribution_params. (default True)

    Raises:
      ValueError: if actor_network or value_network is not of type callable or
        tensorflow.python.ops.template.Template.
    """
    info_spec = ()
    if collect:
      # TODO(oars): Cleanup how we handle non distribution networks.
      if isinstance(actor_network, network.DistributionNetwork):
        network_output_spec = actor_network.output_spec
      else:
        network_output_spec = tf.nest.map_structure(
            distribution_spec.deterministic_distribution_from_spec, action_spec)
      info_spec = tf.nest.map_structure(lambda spec: spec.input_params_spec,
                                        network_output_spec)

    super(PPOPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec,
        actor_network=actor_network,
        observation_normalizer=observation_normalizer,
        clip=clip)

    self._collect = collect
    self._value_network = value_network

  def apply_value_network(self, observations, step_types, policy_state):
    """Apply value network to time_step, potentially a sequence.

    If observation_normalizer is not None, applies observation normalization.

    Args:
      observations: A (possibly nested) observation tensor with outer_dims
        either (batch_size,) or (batch_size, time_index). If observations is a
        time series and network is RNN, will run RNN steps over time series.
      step_types: A (possibly nested) step_types tensor with same outer_dims as
        observations.
      policy_state: Initial policy state for value_network.

    Returns:
      The output of value_net, which is a tuple of:
        - value_preds with same outer_dims as time_step
        - policy_state at the end of the time series
    """
    if self._observation_normalizer:
      observations = self._observation_normalizer.normalize(observations)
    return self._value_network(observations, step_types, policy_state)

  def _apply_actor_network(self, time_step, policy_state):
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(
          time_step.observation)
      time_step = ts.TimeStep(time_step.step_type, time_step.reward,
                              time_step.discount, observation)
    return self._actor_network(
        time_step.observation, time_step.step_type, network_state=policy_state)

  def _variables(self):
    var_list = self._actor_network.variables[:]
    var_list += self._value_network.variables[:]
    if self._observation_normalizer:
      var_list += self._observation_normalizer.variables
    return var_list

  def _distribution(self, time_step, policy_state):
    # Actor network outputs nested structure of distributions or actions.
    actions_or_distributions, policy_state = self._apply_actor_network(
        time_step, policy_state)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)

    # Prepare policy_info.
    if self._collect:
      policy_info = ppo_utils.get_distribution_params(distributions)
    else:
      policy_info = ()

    return policy_step.PolicyStep(distributions, policy_state, policy_info)
