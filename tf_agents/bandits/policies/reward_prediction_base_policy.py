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

"""Base policy that samples actions based on predicted rewards."""

import abc
from typing import Iterable, Optional, Text, Tuple

import gin
import six
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import constraints as constr
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


@gin.configurable
@six.add_metaclass(abc.ABCMeta)
class RewardPredictionBasePolicy(tf_policy.TFPolicy):
  """Base class to build policies based on reward predictions."""

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.NestedTensorSpec,
               reward_network: types.Network,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               accepts_per_arm_features: bool = False,
               constraints: Iterable[constr.BaseConstraint] = (),
               emit_policy_info: Tuple[Text, ...] = (),
               name: Optional[Text] = None):
    """Base class for building policies using a reward tf_agents.Network.

    This abstract base class of policies takes a tf_agents.Network predicting
    rewards and generates the action based on predicted rewards. Sub-classes
    need to implement the `_sample_action` abstract method, which defines the
    mapping from predicted rewards to sampled actions.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      reward_network: An instance of a `tf_agents.network.Network`, callable via
        `network(observation, step_type) -> (output, final_state)`.
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
      constraints: Iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.BaseConstraint`.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec` or the `BoundedTensorSpec` is not valid.
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
    reward_network.create_variables()
    self._reward_network = reward_network
    self._constraints = constraints

    self._emit_policy_info = emit_policy_info
    if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info:
      predicted_rewards_mean = tensor_spec.TensorSpec(
          [self._expected_num_actions])
    else:
      predicted_rewards_mean = ()

    if policy_utilities.InfoFields.BANDIT_POLICY_TYPE in emit_policy_info:
      bandit_policy_type = (
          policy_utilities.create_bandit_policy_type_tensor_spec(shape=[1]))
    else:
      bandit_policy_type = ()

    if accepts_per_arm_features:
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

    super(RewardPredictionBasePolicy, self).__init__(
        time_step_spec,
        action_spec,
        policy_state_spec=reward_network.state_spec,
        clip=False,
        info_spec=info_spec,
        emit_log_probability=policy_utilities.InfoFields.LOG_PROBABILITY
        in emit_policy_info,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  @property
  def accepts_per_arm_features(self):
    return self._accepts_per_arm_features

  def _variables(self):
    policy_variables = self._reward_network.variables
    for c in self._constraints:
      policy_variables.append(c.variables)
    return policy_variables

  @abc.abstractmethod
  def _action_distribution(
      self, mask: Optional[types.Tensor], predicted_rewards: types.Tensor
  ) -> Tuple[tfp.distributions.Distribution, types.Tensor]:
    """Returns an action distribution based on predicted rewards.

    Sub-classes are expected to implement this method.

    Args:
      mask: A 2-D tensor of binary masks shaped as [batch_size, num_of_arms].
      predicted_rewards: A 2-D tensor of predicted rewards shaped as
        [batch_size, num_of_arms].

    Returns:
      A tuple of a `tfp.distributions.Distribution` and a tensor of policy types
      shaped as [batch_size].
    """
    pass

  def _maybe_save_chosen_arm_features(
      self, time_step: ts.TimeStep, action: types.Tensor,
      step: policy_step.PolicyStep) -> policy_step.PolicyStep:
    """Extracts and saves the chosen arm features in the policy info.

    If the policy accepts arm features, this method extracts the arm features
    from `time_step.observation` corresponding to the input `action` tensor,
    saves it in the policy info of the input `step` and returns the modified
    step. Otherwise, the method returns the input `step`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      action: A `types.Tensor` of chosen actions.
      step: A `PolicyStep`.

    Returns:
      A `PolicyStep` resulting from saving the chosen arm features in the policy
      info of the input `step` if the policy accepts arm features, or the input
      `step` otherwise.
    """
    if self.accepts_per_arm_features:
      # Saving the features for the chosen action to the policy_info.
      chosen_arm_features = tf.nest.map_structure(
          lambda obs: tf.gather(params=obs, indices=action, batch_dims=1),
          time_step.observation[bandit_spec_utils.PER_ARM_FEATURE_KEY])
      step = step._replace(
          info=step.info._replace(chosen_arm_features=chosen_arm_features))
    return step

  def _action(self,
              time_step: ts.TimeStep,
              policy_state: types.NestedTensor,
              seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
    """Implementation of `action`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seed: Seed to use if action performs sampling (optional).

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: An action Tensor matching the `action_spec`.
        `state`: A policy state tensor to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.
    """
    step = super(RewardPredictionBasePolicy,
                 self)._action(time_step, policy_state, seed)
    return self._maybe_save_chosen_arm_features(time_step, step.action, step)

  def _distribution(self, time_step, policy_state):
    observation = time_step.observation
    if self.observation_and_action_constraint_splitter is not None:
      observation, _ = self.observation_and_action_constraint_splitter(
          observation)

    predictions, policy_state = self._reward_network(
        observation, time_step.step_type, policy_state)

    if isinstance(self._reward_network,
                  heteroscedastic_q_network.HeteroscedasticQNetwork):
      predicted_reward_values = predictions.q_value_logits
    else:
      predicted_reward_values = predictions

    predicted_reward_values.shape.with_rank_at_least(2)
    predicted_reward_values.shape.with_rank_at_most(3)
    if predicted_reward_values.shape[
        -1] is not None and predicted_reward_values.shape[
            -1] != self._expected_num_actions:
      raise ValueError(
          'The number of actions ({}) does not match the reward_network output'
          ' size ({}).'.format(self._expected_num_actions,
                               predicted_reward_values.shape[1]))

    mask = constr.construct_mask_from_multiple_sources(
        time_step.observation, self._observation_and_action_constraint_splitter,
        self._constraints, self._expected_num_actions)
    action_distribution, bandit_policy_values = self._action_distribution(
        mask, predicted_reward_values)

    if self._accepts_per_arm_features:
      # The actual action sampling hasn't happened yet, so we leave
      # `log_probability` empty and set `chosen_arm_features` to dummy values of
      # all zeros. We need to save dummy chosen arm features to make the
      # returned policy step have the same structure as the policy state spec.
      dummy_chosen_arm_features = tf.nest.map_structure(
          lambda obs: tf.zeros_like(obs[:, 0, ...]),
          time_step.observation[bandit_spec_utils.PER_ARM_FEATURE_KEY])
      policy_info = policy_utilities.PerArmPolicyInfo(
          log_probability=(),
          predicted_rewards_mean=(
              predicted_reward_values
              if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN
              in self._emit_policy_info else ()),
          bandit_policy_type=(bandit_policy_values
                              if policy_utilities.InfoFields.BANDIT_POLICY_TYPE
                              in self._emit_policy_info else ()),
          chosen_arm_features=dummy_chosen_arm_features)
    else:
      # The actual action sampling hasn't happened yet, so we leave
      # `log_probability` empty.
      policy_info = policy_utilities.PolicyInfo(
          log_probability=(),
          predicted_rewards_mean=(
              predicted_reward_values
              if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN
              in self._emit_policy_info else ()),
          bandit_policy_type=(bandit_policy_values
                              if policy_utilities.InfoFields.BANDIT_POLICY_TYPE
                              in self._emit_policy_info else ()))

    return policy_step.PolicyStep(action_distribution, policy_state,
                                  policy_info)
