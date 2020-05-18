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

"""Policy for greedy reward prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.networks import heteroscedastic_q_network
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step


@gin.configurable
class GreedyRewardPredictionPolicy(tf_policy.TFPolicy):
  """Class to build GreedyNNPredictionPolicies."""

  def __init__(self,
               time_step_spec=None,
               action_spec=None,
               reward_network=None,
               observation_and_action_constraint_splitter=None,
               accepts_per_arm_features=False,
               constraints=(),
               emit_policy_info=(),
               name=None):
    """Builds a GreedyRewardPredictionPolicy given a reward tf_agents.Network.

    This policy takes a tf_agents.Network predicting rewards and generates the
    action corresponding to the largest predicted reward.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      reward_network: An instance of a `tf_agents.network.Network`,
        callable via `network(observation, step_type) -> (output, final_state)`.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the network and 2) the
        mask.  The mask should be a 0-1 `Tensor` of shape
        `[batch_size, num_actions]`. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      accepts_per_arm_features: (bool) Whether the policy accepts per-arm
        features.
      constraints: iterable of constraints objects that are instances of
        `tf_agents.bandits.agents.NeuralConstraint`.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.

    Raises:
      NotImplementedError: If `action_spec` contains more than one
        `BoundedTensorSpec` or the `BoundedTensorSpec` is not valid.
    """
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
    predicted_rewards_mean = ()
    if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info:
      predicted_rewards_mean = tensor_spec.TensorSpec(
          [self._expected_num_actions])
    bandit_policy_type = ()
    if policy_utilities.InfoFields.BANDIT_POLICY_TYPE in emit_policy_info:
      bandit_policy_type = (
          policy_utilities.create_bandit_policy_type_tensor_spec(shape=[1]))
    if accepts_per_arm_features:
      # The features for the chosen arm is saved to policy_info.
      observation = time_step_spec.observation
      if observation_and_action_constraint_splitter is not None:
        observation = observation_and_action_constraint_splitter(observation)[0]
      arm_spec = observation[bandit_spec_utils.PER_ARM_FEATURE_KEY]
      chosen_arm_features_info = tensor_spec.remove_outer_dims_nest(arm_spec, 1)
      info_spec = policy_utilities.PerArmPolicyInfo(
          predicted_rewards_mean=predicted_rewards_mean,
          bandit_policy_type=bandit_policy_type,
          chosen_arm_features=chosen_arm_features_info)
    else:
      info_spec = policy_utilities.PolicyInfo(
          predicted_rewards_mean=predicted_rewards_mean,
          bandit_policy_type=bandit_policy_type)

    self._accepts_per_arm_features = accepts_per_arm_features

    super(GreedyRewardPredictionPolicy, self).__init__(
        time_step_spec, action_spec,
        policy_state_spec=reward_network.state_spec,
        clip=False,
        info_spec=info_spec,
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

  def _distribution(self, time_step, policy_state):
    observation = time_step.observation
    mask = None
    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)
    if observation_and_action_constraint_splitter is not None:
      observation, mask = observation_and_action_constraint_splitter(
          observation)

    predictions, policy_state = self._reward_network(
        observation, time_step.step_type, policy_state)
    batch_size = tf.shape(predictions)[0]

    if isinstance(self._reward_network,
                  heteroscedastic_q_network.HeteroscedasticQNetwork):
      predicted_reward_values = predictions.q_value_logits
    else:
      predicted_reward_values = predictions

    predicted_reward_values.shape.with_rank_at_least(2)
    predicted_reward_values.shape.with_rank_at_most(3)
    if predicted_reward_values.shape[-1] != self._expected_num_actions:
      raise ValueError(
          'The number of actions ({}) does not match the reward_network output'
          ' size ({}).'.format(self._expected_num_actions,
                               predicted_reward_values.shape[1]))

    if self._constraints:
      # Action feasibility computation.
      feasibility_prob = policy_utilities.compute_feasibility_probability(
          observation, self._constraints, batch_size,
          self._expected_num_actions, mask)
      # Probabilistic masking.
      mask = tfp.distributions.Bernoulli(probs=feasibility_prob).sample()

    # Argmax.
    if self._constraints or (observation_and_action_constraint_splitter
                             is not None):
      actions = policy_utilities.masked_argmax(
          predicted_reward_values, mask, output_type=self.action_spec.dtype)
    else:
      actions = tf.argmax(
          predicted_reward_values, axis=-1, output_type=self.action_spec.dtype)

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
          predicted_rewards_mean=(
              predicted_reward_values if policy_utilities.InfoFields
              .PREDICTED_REWARDS_MEAN in self._emit_policy_info else ()),
          bandit_policy_type=(bandit_policy_values
                              if policy_utilities.InfoFields.BANDIT_POLICY_TYPE
                              in self._emit_policy_info else ()),
          chosen_arm_features=chosen_arm_features)
    else:
      policy_info = policy_utilities.PolicyInfo(
          predicted_rewards_mean=(
              predicted_reward_values if policy_utilities.InfoFields
              .PREDICTED_REWARDS_MEAN in self._emit_policy_info else ()),
          bandit_policy_type=(bandit_policy_values
                              if policy_utilities.InfoFields.BANDIT_POLICY_TYPE
                              in self._emit_policy_info else ()))

    return policy_step.PolicyStep(
        tfp.distributions.Deterministic(loc=actions), policy_state, policy_info)
