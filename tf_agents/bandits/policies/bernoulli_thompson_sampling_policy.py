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

"""Policy for Bernoulli Thompson Sampling."""


from typing import Optional, Sequence, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

tfd = tfp.distributions


@gin.configurable
class BernoulliThompsonSamplingPolicy(tf_policy.TFPolicy):
  """Class to build Bernoulli Thompson Sampling policies."""

  def __init__(self,
               time_step_spec: types.TimeStep,
               action_spec: types.NestedTensorSpec,
               alpha: Sequence[tf.Variable],
               beta: Sequence[tf.Variable],
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               emit_policy_info: Sequence[Text] = (),
               name: Optional[Text] = None):
    """Builds a BernoulliThompsonSamplingPolicy.

    For a reference, see e.g., Chapter 3 in "A Tutorial on Thompson Sampling" by
    Russo et al. (https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf).

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      alpha: list or tuple of tf.Variable's. It holds the `alpha` parameter of
        the beta distribution of each arm.
      beta: list or tuple of tf.Variable's. It holds the `beta` parameter of the
        beta distribution of each arm.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the network and 2) the
        mask.  The mask should be a 0-1 `Tensor` of shape
        `[batch_size, num_actions]`. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
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
          'action_spec must be a BoundedTensorSpec of integer type and '
          'shape (). Found {}.'.format(action_spec))
    self._expected_num_actions = action_spec.maximum - action_spec.minimum + 1

    if len(alpha) != self._expected_num_actions:
      raise ValueError(
          'The size of alpha parameters is expected to be equal '
          'to the number of actions, but found to be {}'.format(len(alpha)))
    self._alpha = alpha
    if len(alpha) != len(beta):
      raise ValueError(
          'The size of alpha parameters is expected to be equal '
          'to the size of beta parameters')
    self._beta = beta

    self._emit_policy_info = emit_policy_info
    predicted_rewards_mean = ()
    if policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN in emit_policy_info:
      predicted_rewards_mean = tensor_spec.TensorSpec(
          [self._expected_num_actions])
    predicted_rewards_sampled = ()
    if policy_utilities.InfoFields.PREDICTED_REWARDS_SAMPLED in (
        emit_policy_info):
      predicted_rewards_sampled = tensor_spec.TensorSpec(
          [self._expected_num_actions])
    info_spec = policy_utilities.PolicyInfo(
        predicted_rewards_mean=predicted_rewards_mean,
        predicted_rewards_sampled=predicted_rewards_sampled)

    super(BernoulliThompsonSamplingPolicy, self).__init__(
        time_step_spec, action_spec,
        info_spec=info_spec,
        emit_log_probability='log_probability' in emit_policy_info,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter),
        name=name)

  def _variables(self):
    return self._alpha + self._beta  # pytype: disable=unsupported-operands  # trace-all-classes

  def _distribution(self, time_step, policy_state):
    if time_step.step_type.shape:
      if tf.is_tensor(
          time_step.step_type) and time_step.step_type.shape.rank > 0:
        batch_size = time_step.step_type.get_shape().as_list()[0]
      else:
        batch_size = 1
    else:
      batch_size = 1
    # Sample from the posterior distribution.
    posterior_dist = tfd.Beta(self._alpha, self._beta)
    predicted_reward_sampled = posterior_dist.sample([batch_size])
    predicted_reward_means_1d = tf.stack([
        self._alpha[k] / (self._alpha[k] + self._beta[k]) for k in range(
            self._expected_num_actions)], axis=-1)
    predicted_reward_means = tf.stack([
        predicted_reward_means_1d for k in range(batch_size)], axis=0)

    mask = None
    if self._observation_and_action_constraint_splitter is not None:
      _, mask = self._observation_and_action_constraint_splitter(
          time_step.observation)

    # Argmax.
    if mask is not None:
      actions = policy_utilities.masked_argmax(
          predicted_reward_sampled, mask, output_type=self.action_spec.dtype)
    else:
      actions = tf.argmax(
          predicted_reward_sampled, axis=-1, output_type=self.action_spec.dtype)

    policy_info = policy_utilities.populate_policy_info(
        arm_observations=(), chosen_actions=actions,
        rewards_for_argmax=tf.cast(predicted_reward_sampled, tf.float32),
        est_rewards=tf.cast(predicted_reward_means, tf.float32),
        emit_policy_info=self._emit_policy_info,
        accepts_per_arm_features=False)
    if policy_utilities.InfoFields.LOG_PROBABILITY in self._emit_policy_info:
      policy_info._replace(
          log_probability=tf.zeros([batch_size], tf.float32))

    return policy_step.PolicyStep(
        tfp.distributions.Deterministic(loc=actions), policy_state, policy_info)
