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

"""Policy implementation that generates random actions."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.distributions import masked
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


def _uniform_probability(action_spec):
  """Helper function for returning probabilities of equivalent distributions."""
  # Equivalent of what a tfp.distribution.Categorical would return.
  if action_spec.dtype.is_integer:
    return 1. / (action_spec.maximum - action_spec.minimum + 1)
  # Equivalent of what a tfp.distribution.Uniform would return.
  return 1. / (action_spec.maximum - action_spec.minimum)


class RandomTFPolicy(tf_policy.TFPolicy):
  """Returns random samples of the given action_spec.

  Note: the values in the info_spec (except for the log_probability) are random
    values that have nothing to do with the emitted actions.
  """

  def __init__(self, time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec, *args, **kwargs):
    observation_and_action_constraint_splitter = (
        kwargs.get('observation_and_action_constraint_splitter', None))
    self._accepts_per_arm_features = (
        kwargs.pop('accepts_per_arm_features', False))

    if observation_and_action_constraint_splitter is not None:
      if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
        raise NotImplementedError(
            'RandomTFPolicy only supports action constraints for '
            'BoundedTensorSpec action specs.')

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

      zero_logits = tf.cast(tf.zeros_like(mask), tf.float32)
      masked_categorical = masked.MaskedCategorical(zero_logits, mask)
      action_ = tf.cast(masked_categorical.sample() + self.action_spec.minimum,
                        self.action_spec.dtype)

      # If the action spec says each action should be shaped (1,), add another
      # dimension so the final shape is (B, 1) rather than (B,).
      if self.action_spec.shape.rank == 1:
        action_ = tf.expand_dims(action_, axis=-1)
      policy_info = tensor_spec.sample_spec_nest(
          self._info_spec, outer_dims=outer_dims)
    else:
      observation = time_step.observation

      action_ = tensor_spec.sample_spec_nest(
          self._action_spec, seed=seed, outer_dims=outer_dims)
      policy_info = tensor_spec.sample_spec_nest(
          self._info_spec, outer_dims=outer_dims)
    if self._accepts_per_arm_features:
      def _gather_fn(t):
        return tf.gather(params=t, indices=action_, batch_dims=1)

      chosen_arm_features = tf.nest.map_structure(_gather_fn,
                                                  observation['per_arm'])
      policy_info = policy_info._replace(
          chosen_arm_features=chosen_arm_features)

    # TODO(b/78181147): Investigate why this control dependency is required.
    if time_step is not None:
      with tf.control_dependencies(tf.nest.flatten(time_step)):
        action_ = tf.nest.map_structure(tf.identity, action_)

    if self.emit_log_probability:
      if observation_and_action_constraint_splitter is not None:
        log_probability = masked_categorical.log_prob(action_ -
                                                      self.action_spec.minimum)
      else:
        action_probability = tf.nest.map_structure(_uniform_probability,
                                                   self._action_spec)
        log_probability = tf.nest.map_structure(tf.math.log, action_probability)
      policy_info = policy_step.set_log_probability(policy_info,
                                                    log_probability)

    step = policy_step.PolicyStep(action_, policy_state, policy_info)
    return step

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'RandomTFPolicy does not support distributions yet.')
