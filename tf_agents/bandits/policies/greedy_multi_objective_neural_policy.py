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

"""Policy for greedy multi-objective prediction."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import copy
from typing import List, Optional, Sequence, Text, Tuple

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.bandits.multi_objective import multi_objective_scalarizer
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import constraints
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks.network import Network
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


@tf.function
def scalarize_objectives(objectives_tensor: tf.Tensor,
                         scalarizer: multi_objective_scalarizer.Scalarizer):
  """Scalarize a rank-3 objectives tensor into a rank-2 tensor.

  Scalarize an objective values tensor shaped as
  [batch_size, num_of_objectives, num_of_actions] along the second dimension
  into a rank-2 tensor shaped as [batch_size, num_of_actions]

  Args:
    objectives_tensor: An objectives tensor to be scalarized.
    scalarizer: A
      `tf_agents.bandits.multi_objective.multi_objective_scalarizer.Scalarizer`
      object that implements scalarization of multiple objectives into a single
      scalar reward.

  Returns:
    A rank-2 tensor of scalarized rewards shaped as
    [batch_size, num_of_actions].

  Raises:
    ValueError: If `objectives_tensor` is not rank-3.
  """
  if objectives_tensor.shape.rank != 3:
    raise ValueError(
        'The objectives_tensor should be rank-3, but is rank-{}'.format(
            objectives_tensor.shape.rank))
  return tf.transpose(
      tf.map_fn(scalarizer, tf.transpose(objectives_tensor, perm=[2, 0, 1])))


@gin.configurable
class GreedyMultiObjectiveNeuralPolicy(tf_policy.TFPolicy):
  """Class to build GreedyMultiObjectiveNeuralPolicy objects."""

  def __init__(
      self,
      time_step_spec: Optional[ts.TimeStep],
      action_spec: Optional[types.NestedBoundedTensorSpec],
      scalarizer: multi_objective_scalarizer.Scalarizer,
      objective_networks: Sequence[Network],
      observation_and_action_constraint_splitter: types.Splitter = None,
      accepts_per_arm_features: bool = False,
      emit_policy_info: Tuple[Text, ...] = (),
      name: Optional[Text] = None):
    """Builds a GreedyMultiObjectiveNeuralPolicy based on multiple networks.

    This policy takes an iterable of `tf_agents.Network`, each responsible for
    predicting a specific objective, along with a `Scalarizer` object to
    generate an action by maximizing the scalarized objective, i.e., the output
    of the `Scalarizer` applied to the multiple predicted objectives by the
    networks.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      scalarizer: A
       `tf_agents.bandits.multi_objective.multi_objective_scalarizer.Scalarizer`
        object that implements scalarization of multiple objectives into a
        single scalar reward.
      objective_networks: A Sequence of `tf_agents.network.Network` objects to
        be used by the policy. Each network will be called with
        call(observation, step_type) and is expected to provide a prediction for
        a specific objective for all actions.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the network and 2) the
        mask.  The mask should be a 0-1 `Tensor` of shape `[batch_size,
        num_actions]`. This function should also work with a `TensorSpec` as
        input, and should output `TensorSpec` objects for the observation and
        mask.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec` or the `BoundedTensorSpec` is not valid.
      NotImplementedError: If `action_spec` is not a `BoundedTensorSpec` of type
        int32 and shape ().
      ValueError: If `objective_networks` has fewer than two networks.
      ValueError: If `accepts_per_arm_features` is true but `time_step_spec` is
        None.
    """
    policy_utilities.check_no_mask_with_arm_features(
        accepts_per_arm_features, observation_and_action_constraint_splitter)
    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise NotImplementedError(
          'action_spec can only contain a single BoundedTensorSpec.')

    action_spec = flat_action_spec[0]
    if (not tensor_spec.is_bounded(action_spec) or
        not tensor_spec.is_discrete(action_spec) or
        action_spec.shape.rank > 1 or
        action_spec.shape.num_elements() != 1):
      raise NotImplementedError(
          'action_spec must be a BoundedTensorSpec of type int32 and shape (). '
          'Found {}.'.format(action_spec))
    self._expected_num_actions = action_spec.maximum - action_spec.minimum + 1
    self._action_offset = action_spec.minimum
    policy_state_spec = []
    for network in objective_networks:
      policy_state_spec.append(network.state_spec)
      network.create_variables()
    self._objective_networks = objective_networks
    self._scalarizer = scalarizer
    self._num_objectives = len(self._objective_networks)
    if self._num_objectives < 2:
      raise ValueError(
          'Number of objectives should be at least two, but found to be {}'
          .format(self._num_objectives))

    self._emit_policy_info = emit_policy_info
    predicted_rewards_mean = ()
    if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info:
      predicted_rewards_mean = tensor_spec.TensorSpec(
          [self._num_objectives, self._expected_num_actions])
    bandit_policy_type = ()
    if policy_utilities.InfoFields.BANDIT_POLICY_TYPE in emit_policy_info:
      bandit_policy_type = (
          policy_utilities.create_bandit_policy_type_tensor_spec(shape=[1]))
    if accepts_per_arm_features:
      if time_step_spec is None:
        raise ValueError(
            'time_step_spec should not be None for per-arm-features policies, '
            'but found to be.')
      # The features for the chosen arm is saved to policy_info.
      chosen_arm_features_info = (
          policy_utilities.create_chosen_arm_features_info_spec(
              time_step_spec.observation))
      info_spec = policy_utilities.PerArmPolicyInfo(
          predicted_rewards_mean=predicted_rewards_mean,
          bandit_policy_type=bandit_policy_type,
          chosen_arm_features=chosen_arm_features_info)
    else:
      info_spec = policy_utilities.PolicyInfo(
          predicted_rewards_mean=predicted_rewards_mean,
          bandit_policy_type=bandit_policy_type)

    self._accepts_per_arm_features = accepts_per_arm_features

    super(GreedyMultiObjectiveNeuralPolicy, self).__init__(
        time_step_spec,
        action_spec,
        policy_state_spec=policy_state_spec,
        clip=False,
        info_spec=info_spec,
        emit_log_probability='log_probability' in emit_policy_info,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  @property
  def accepts_per_arm_features(self) -> bool:
    return self._accepts_per_arm_features

  @property
  def scalarizer(self) -> multi_objective_scalarizer.Scalarizer:
    return self._scalarizer

  def _predict(
      self, observation: types.NestedSpecTensorOrArray,
      step_type: types.SpecTensorOrArray,
      policy_state: Sequence[types.TensorSpec]
  ) -> Tuple[tf.Tensor, List[types.TensorSpec]]:
    """Predict the objectives using the policy's objective networks.

    Args:
      observation: The observation whose objectives are to be predicted.
      step_type: The `tf_agents.trajectories.time_step.StepType` for the input
        observation.
      policy_state: The states for the policy's objective networks.

    Returns:
      A tuple of a rank-3 tensor for the predicted objectives shaped as
      [batch_size, num_of_objectives, num_of_actions] and a list of updated
      states, one for each objective network.

    Raises:
      ValueError: If the output of any objective network is not a rank-2 tensor.
      ValueError: If the output size of any objective network does not match the
        expected number of actions.
    """
    # TODO(b/158804957): Use literal comparison because in some strange cases
    # (tf.function? autograph?) the expression "x in (None, (), [])" gets
    # converted to a tensor.
    if policy_state is None or policy_state is () or policy_state is []:  # pylint: disable=literal-comparison
      policy_state = [()] * self._num_objectives
    predicted_objective_values = []
    updated_policy_state = []
    for idx, (network,
              state) in enumerate(zip(self._objective_networks, policy_state)):
      prediction, state = network(observation, step_type, state)
      updated_policy_state.append(copy.deepcopy(state))
      predicted_value = prediction.q_value_logits if isinstance(
          network,
          heteroscedastic_q_network.HeteroscedasticQNetwork) else prediction
      if predicted_value.shape.rank != 2:
        raise ValueError('Prediction from network {} shoud be a rank-2 tensor, '
                         ' but has shape {}'.format(idx, predicted_value.shape))
      if predicted_value.shape[1] is not None and predicted_value.shape[
          1] != self._expected_num_actions:
        raise ValueError(
            'The number of actions ({}) does not match objective network {}'
            ' output size ({}).'.format(self._expected_num_actions, idx,
                                        predicted_value.shape[1]))
      predicted_objective_values.append(predicted_value)
    # Stack the list of predicted objective tensors into a rank-3 tensor shaped
    # as [batch_size, num_of_objectives, num_of_actions].
    predicted_objectives_tensor = tf.stack(predicted_objective_values, axis=1)
    return predicted_objectives_tensor, updated_policy_state

  def _distribution(
      self, time_step: ts.TimeStep,
      policy_state: Sequence[types.TensorSpec]) -> policy_step.PolicyStep:
    observation = time_step.observation
    if self.observation_and_action_constraint_splitter is not None:
      observation, _ = self.observation_and_action_constraint_splitter(
          observation)
    predicted_objective_values_tensor, policy_state = self._predict(
        observation, time_step.step_type, policy_state)
    scalarized_reward = scalarize_objectives(predicted_objective_values_tensor,
                                             self._scalarizer)
    # Preserve static batch size values when they are available.
    batch_size = (tf.compat.dimension_value(scalarized_reward.shape[0])
                  or tf.shape(scalarized_reward)[0])
    mask = constraints.construct_mask_from_multiple_sources(
        time_step.observation, self._observation_and_action_constraint_splitter,
        (), self._expected_num_actions)

    # Argmax.
    if mask is not None:
      actions = policy_utilities.masked_argmax(
          scalarized_reward, mask, output_type=self.action_spec.dtype)
    else:
      actions = tf.argmax(
          scalarized_reward, axis=-1, output_type=self.action_spec.dtype)

    actions += self._action_offset

    bandit_policy_values = tf.fill([batch_size, 1],
                                   policy_utilities.BanditPolicyType.GREEDY)

    if self._accepts_per_arm_features:
      # Saving the features for the chosen action to the policy_info.
      def gather_observation(obs):
        return tf.gather(params=obs, indices=actions, batch_dims=1)

      chosen_arm_features = tf.nest.map_structure(
          gather_observation,
          observation[bandit_spec_utils.PER_ARM_FEATURE_KEY])
      policy_info = policy_utilities.PerArmPolicyInfo(
          log_probability=tf.zeros([batch_size], tf.float32) if
          policy_utilities.InfoFields.LOG_PROBABILITY in self._emit_policy_info
          else (),
          predicted_rewards_mean=(
              predicted_objective_values_tensor if policy_utilities.InfoFields
              .PREDICTED_REWARDS_MEAN in self._emit_policy_info else ()),
          bandit_policy_type=(bandit_policy_values
                              if policy_utilities.InfoFields.BANDIT_POLICY_TYPE
                              in self._emit_policy_info else ()),
          chosen_arm_features=chosen_arm_features)
    else:
      policy_info = policy_utilities.PolicyInfo(
          log_probability=tf.zeros([batch_size], tf.float32) if
          policy_utilities.InfoFields.LOG_PROBABILITY in self._emit_policy_info
          else (),
          predicted_rewards_mean=(
              predicted_objective_values_tensor if policy_utilities.InfoFields
              .PREDICTED_REWARDS_MEAN in self._emit_policy_info else ()),
          bandit_policy_type=(bandit_policy_values
                              if policy_utilities.InfoFields.BANDIT_POLICY_TYPE
                              in self._emit_policy_info else ()))

    return policy_step.PolicyStep(
        tfp.distributions.Deterministic(loc=actions), policy_state, policy_info)
