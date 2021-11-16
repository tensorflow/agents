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

"""Policy implementation that generates random actions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import cast

import tensorflow as tf
from tf_agents.distributions import masked
from tf_agents.policies import tf_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.specs import bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


def _calculate_log_probability(outer_dims, action_spec):
  """Helper function for calculating log prob of a uniform distribution.

  Each item in the returned tensor will be equal to:
  |action_spec.shape| * log_prob_of_each_component_of_action_spec.

  Note that this method expects the same value for all outer_dims because
  we're sampling uniformly from the same distribution for each batch row.

  Args:
    outer_dims: TensorShape.
    action_spec: BoundedTensorSpec.

  Returns:
    A tensor of type float32 with shape outer_dims.
  """
  # Equivalent of what a tfp.distribution.Categorical would return.
  if action_spec.dtype.is_integer:
    log_prob = -tf.math.log(action_spec.maximum - action_spec.minimum + 1.0)
  # Equivalent of what a tfp.distribution.Uniform would return.
  else:
    log_prob = -tf.math.log(action_spec.maximum - action_spec.minimum)

  # Note that log_prob may be a vector. We first reduce it to a scalar, and then
  # adjust by the number of times that vector is repeated in action_spec.
  log_prob = tf.reduce_sum(log_prob) * (
      action_spec.shape.num_elements() / log_prob.shape.num_elements())
  # Regardless of the type of the action, the log_prob should be float32.
  return tf.cast(tf.fill(outer_dims, log_prob), tf.float32)


# TODO(b/161005095): Refactor into RandomTFPolicy and RandomBanditTFPolicy.
class RandomTFPolicy(tf_policy.TFPolicy):
  """Returns random samples of the given action_spec.

  Note: the values in the info_spec (except for the log_probability) are random
    values that have nothing to do with the emitted actions.

  Note: The returned info.log_probabiliy will be an object matching the
  structure of action_spec, where each value is a tensor of size [batch_size].
  """

  def __init__(self, time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec, *args, **kwargs):
    observation_and_action_constraint_splitter = (
        kwargs.get('observation_and_action_constraint_splitter', None))
    self._accepts_per_arm_features = (
        kwargs.pop('accepts_per_arm_features', False))
    self._stationary_mask = kwargs.pop('stationary_mask', None)
    if self._stationary_mask is not None:
      if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
        raise NotImplementedError(
            'RandomTFPolicy only supports action constraints for '
            'BoundedTensorSpec action specs.')
      assert action_spec.dtype.is_integer, ('To use a stationary mask, action '
                                            'dtype must be integer.')
      self._stationary_mask = tf.constant([self._stationary_mask])
      num_actions = action_spec.maximum - action_spec.minimum + 1
      assert (self._stationary_mask.shape[-1] == num_actions), (
          'Stationary mask should have length equal to the number of actions, '
          'but we get {} and {}.'
          .format(num_actions, self._stationary_mask.shape[-1]))

    if observation_and_action_constraint_splitter is not None:
      if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
        raise NotImplementedError(
            'RandomTFPolicy only supports action constraints for '
            'BoundedTensorSpec action specs.')

      action_spec = tensor_spec.from_spec(action_spec)
      action_spec = cast(tensor_spec.BoundedTensorSpec, action_spec)
      scalar_shape = action_spec.shape.rank == 0
      single_dim_shape = (
          action_spec.shape.rank == 1 and action_spec.shape.dims == [1])

      if not scalar_shape and not single_dim_shape:
        raise NotImplementedError(
            'RandomTFPolicy only supports action constraints for action specs '
            'shaped as () or (1,) or their equivalent list forms.')

    super(RandomTFPolicy, self).__init__(time_step_spec, action_spec, *args,
                                         **kwargs)

  def _variables(self):
    return []

  def _action(self, time_step, policy_state, seed):
    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)

    outer_dims = nest_utils.get_outer_shape(time_step, self._time_step_spec)
    if observation_and_action_constraint_splitter is not None:
      observation, mask = observation_and_action_constraint_splitter(
          time_step.observation)

      if self._stationary_mask is not None:
        mask = mask * self._stationary_mask

      action_spec = tensor_spec.from_spec(self.action_spec)
      action_spec = cast(tensor_spec.BoundedTensorSpec, action_spec)
      zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
      masked_categorical = masked.MaskedCategorical(zero_logits, mask)
      action_ = tf.cast(masked_categorical.sample() + action_spec.minimum,
                        action_spec.dtype)

      # If the action spec says each action should be shaped (1,), add another
      # dimension so the final shape is (B, 1) rather than (B,).
      if action_spec.shape.rank == 1:
        action_ = tf.expand_dims(action_, axis=-1)
      policy_info = tensor_spec.sample_spec_nest(
          self._info_spec, outer_dims=outer_dims)
    else:
      observation = time_step.observation
      action_spec = cast(tensor_spec.BoundedTensorSpec, self.action_spec)

      if self._accepts_per_arm_features:
        max_num_arms = action_spec.maximum - action_spec.minimum + 1
        batch_size = tf.shape(time_step.step_type)[0]
        num_actions = observation.get(
            bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY,
            tf.ones(shape=(batch_size,), dtype=tf.int32) * max_num_arms)
        mask = tf.sequence_mask(num_actions, max_num_arms)
        zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
        masked_categorical = masked.MaskedCategorical(zero_logits, mask)
        action_ = tf.nest.map_structure(
            lambda t: tf.cast(masked_categorical.sample() + t.minimum, t.dtype),
            action_spec)
      elif self._stationary_mask is not None:
        batch_size = tf.shape(time_step.step_type)[0]
        mask = tf.tile(self._stationary_mask, [batch_size, 1])
        zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
        masked_categorical = masked.MaskedCategorical(zero_logits, mask)
        action_ = tf.cast(masked_categorical.sample() + action_spec.minimum,
                          action_spec.dtype)
      else:
        action_ = tensor_spec.sample_spec_nest(
            self._action_spec, seed=seed, outer_dims=outer_dims)

      policy_info = tensor_spec.sample_spec_nest(
          self._info_spec, outer_dims=outer_dims)

    # Update policy info with chosen arm features.
    if self._accepts_per_arm_features:
      def _gather_fn(t):
        return tf.gather(params=t, indices=action_, batch_dims=1)
      chosen_arm_features = tf.nest.map_structure(
          _gather_fn, observation[bandit_spec_utils.PER_ARM_FEATURE_KEY])

      if policy_utilities.has_chosen_arm_features(self._info_spec):
        policy_info = policy_info._replace(
            chosen_arm_features=chosen_arm_features)

    # TODO(b/78181147): Investigate why this control dependency is required.
    def _maybe_convert_sparse_tensor(t):
      if isinstance(t, tf.SparseTensor):
        return tf.sparse.to_dense(t)
      else:
        return t
    if time_step is not None:
      with tf.control_dependencies(
          tf.nest.flatten(tf.nest.map_structure(_maybe_convert_sparse_tensor,
                                                time_step))):
        action_ = tf.nest.map_structure(tf.identity, action_)

    if self.emit_log_probability:
      if (self._accepts_per_arm_features
          or observation_and_action_constraint_splitter is not None
          or self._stationary_mask is not None):
        action_spec = cast(tensor_spec.BoundedTensorSpec, self.action_spec)
        log_probability = masked_categorical.log_prob(
            action_ - action_spec.minimum)
      else:
        log_probability = tf.nest.map_structure(
            lambda s: _calculate_log_probability(outer_dims, s),
            self._action_spec)
      policy_info = policy_step.set_log_probability(policy_info,
                                                    log_probability)

    step = policy_step.PolicyStep(action_, policy_state, policy_info)
    return step

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'RandomTFPolicy does not support distributions yet.')
