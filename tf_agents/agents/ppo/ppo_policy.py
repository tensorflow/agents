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

"""An ActorPolicy that also returns policy_info needed for PPO training."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_utils
from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import tensor_normalizer


tfd = tfp.distributions


@gin.configurable(module='tf_agents')
class PPOPolicy(actor_policy.ActorPolicy):
  """An ActorPolicy that also returns policy_info needed for PPO training.

  This policy requires two networks: the usual `actor_network` and the
  additional `value_network`. The value network can be executed with the
  `apply_value_network()` method.

  When the networks have state (RNNs, LSTMs) you must be careful to pass the
  state for the actor network to `action()` and the state of the value network
  to `apply_value_network()`. Use `get_initial_value_state()` to access
  the state of the value network.
  """

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               actor_network: network.Network,
               value_network: network.Network,
               observation_normalizer: Optional[
                   tensor_normalizer.TensorNormalizer] = None,
               clip: bool = True,
               collect: bool = True,
               compute_value_and_advantage_in_train: bool = False):
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
      compute_value_and_advantage_in_train: A bool to indicate where value
        prediction and advantage calculation happen.  If True, both happen in
        agent.train(), therefore no need to save the value prediction inside of
        policy info. If False, value prediction is computed during data
        collection. This argument must be set to `False` if mini batch learning
        is enabled.

    Raises:
      TypeError: if `actor_network` or `value_network` is not of type
        `tf_agents.networks.Network`.
      ValueError: if `actor_network` or `value_network` do not emit
        valid outputs.  For example, `actor_network` must either be
        a (legacy style) `DistributionNetwork`, or explicitly emit
        a nest of `tfp.distribution.Distribution` objects.
    """
    if not isinstance(actor_network, network.Network):
      raise TypeError('actor_network is not of type network.Network')
    if not isinstance(value_network, network.Network):
      raise TypeError('value_network is not of type network.Network')

    actor_output_spec = actor_network.create_variables(
        time_step_spec.observation)

    value_output_spec = value_network.create_variables(
        time_step_spec.observation)

    nest_utils.assert_value_spec(
        value_output_spec, 'value_network')

    distribution_utils.assert_specs_are_compatible(
        actor_output_spec, action_spec,
        'actor_network output spec does not match action spec')

    self._compute_value_and_advantage_in_train = (
        compute_value_and_advantage_in_train)

    if collect:
      if isinstance(actor_network, network.DistributionNetwork):
        # Legacy DistributionNetwork case.  New code can just provide a regular
        # Network that emits a Distribution object; and we use a different
        # code path using DistributionSpecV2 for that.
        network_output_spec = actor_network.output_spec
        info_spec = {
            'dist_params':
                tf.nest.map_structure(lambda spec: spec.input_params_spec,
                                      network_output_spec)
        }
      else:
        # We have a Network that emits a nest of distributions.
        def nested_dist_params(spec):
          if not isinstance(spec, distribution_utils.DistributionSpecV2):
            raise ValueError(
                'Unexpected output from `actor_network`.  Expected '
                '`Distribution` objects, but saw output spec: {}'
                .format(actor_output_spec))
          return distribution_utils.parameters_to_dict(
              spec.parameters, tensors_only=True)

        info_spec = {
            'dist_params':
                tf.nest.map_structure(nested_dist_params,
                                      actor_output_spec)
        }

      if not self._compute_value_and_advantage_in_train:
        info_spec['value_prediction'] = tensor_spec.TensorSpec(
            shape=[], dtype=tf.float32)
    else:
      info_spec = ()

    policy_state_spec = {}
    if actor_network.state_spec:
      policy_state_spec['actor_network_state'] = actor_network.state_spec
    if (collect and value_network.state_spec and
        not self._compute_value_and_advantage_in_train):
      policy_state_spec['value_network_state'] = value_network.state_spec
    if not policy_state_spec:
      policy_state_spec = ()

    super(PPOPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=policy_state_spec,
        info_spec=info_spec,
        actor_network=actor_network,
        observation_normalizer=observation_normalizer,
        clip=clip)

    self._collect = collect
    self._value_network = value_network

  def get_initial_value_state(self,
                              batch_size: types.Int) -> types.NestedTensor:
    """Returns the initial state of the value network.

    Args:
      batch_size: A constant or Tensor holding the batch size. Can be None, in
        which case the state will not have a batch dimension added.

    Returns:
      A nest of zero tensors matching the spec of the value network state.
    """
    return tensor_spec.zero_spec_nest(
        self._value_network.state_spec,
        outer_dims=None if batch_size is None else [batch_size])

  def apply_value_network(self,
                          observations: types.NestedTensor,
                          step_types: types.Tensor,
                          value_state: Optional[types.NestedTensor] = None,
                          training: bool = False) -> types.NestedTensor:
    """Apply value network to time_step, potentially a sequence.

    If observation_normalizer is not None, applies observation normalization.

    Args:
      observations: A (possibly nested) observation tensor with outer_dims
        either (batch_size,) or (batch_size, time_index). If observations is a
        time series and network is RNN, will run RNN steps over time series.
      step_types: A (possibly nested) step_types tensor with same outer_dims as
        observations.
      value_state: Optional. Initial state for the value_network. If not
        provided the behavior depends on the value network itself.
      training: Whether the output value is going to be used for training.

    Returns:
      The output of value_net, which is a tuple of:
        - value_preds with same outer_dims as time_step
        - value_state at the end of the time series
    """
    if self._observation_normalizer:
      observations = self._observation_normalizer.normalize(observations)
    return self._value_network(observations, step_types, value_state,
                               training=training)

  def _apply_actor_network(self, time_step, policy_state, training=False):
    observation = time_step.observation
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)

    return self._actor_network(
        observation,
        time_step.step_type,
        network_state=policy_state,
        training=training)

  def _variables(self):
    var_list = self._actor_network.variables[:]
    var_list += self._value_network.variables[:]
    if self._observation_normalizer:
      var_list += self._observation_normalizer.variables
    return var_list

  def _distribution(self, time_step, policy_state, training=False):
    if not policy_state:
      policy_state = {'actor_network_state': (), 'value_network_state': ()}
    else:
      policy_state = policy_state.copy()

    if 'actor_network_state' not in policy_state:
      policy_state['actor_network_state'] = ()
    if 'value_network_state' not in policy_state:
      policy_state['value_network_state'] = ()

    new_policy_state = {'actor_network_state': (), 'value_network_state': ()}

    (distributions, new_policy_state['actor_network_state']) = (
        self._apply_actor_network(
            time_step, policy_state['actor_network_state'], training=training))

    if self._collect:
      policy_info = {
          'dist_params': ppo_utils.get_distribution_params(
              distributions,
              legacy_distribution_network=isinstance(
                  self._actor_network,
                  network.DistributionNetwork))
      }

      if not self._compute_value_and_advantage_in_train:
        # If value_prediction is not computed in agent.train it needs to be
        # computed and saved here.
        (policy_info['value_prediction'],
         new_policy_state['value_network_state']) = self.apply_value_network(
             time_step.observation,
             time_step.step_type,
             value_state=policy_state['value_network_state'],
             training=False)
    else:
      policy_info = ()

    if (not new_policy_state['actor_network_state'] and
        not new_policy_state['value_network_state']):
      new_policy_state = ()
    elif not new_policy_state['value_network_state']:
      del new_policy_state['value_network_state']
    elif not new_policy_state['actor_network_state']:
      del new_policy_state['actor_network_state']

    return policy_step.PolicyStep(distributions, new_policy_state, policy_info)
