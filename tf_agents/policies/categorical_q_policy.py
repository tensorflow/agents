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

"""Simple Categorical Q-Policy for Q-Learning with Categorical DQN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


@gin.configurable()
class CategoricalQPolicy(tf_policy.Base):
  """Class to build categorical Q-policies."""

  def __init__(self,
               min_q_value,
               max_q_value,
               q_network,
               action_spec,
               temperature=1.0):
    """Builds a categorical Q-policy given a categorical Q-network.

    Args:
      min_q_value: A float specifying the minimum Q-value, used for setting up
        the support.
      max_q_value: A float specifying the maximum Q-value, used for setting up
        the support.
      q_network: A network.Network to use for our policy.
      action_spec: A `BoundedTensorSpec` representing the actions.
      temperature: temperature for sampling, when close to 0.0 is arg_max.

    Raises:
      ValueError: if `q_network` does not have property `num_atoms`.
      TypeError: if `action_spec` is not a `BoundedTensorSpec`.
    """
    num_atoms = getattr(q_network, 'num_atoms', None)
    if num_atoms is None:
      raise ValueError('Expected q_network to have property `num_atoms`, but '
                       'it doesn\'t. Network is: %s' % q_network)

    time_step_spec = ts.time_step_spec(q_network.input_tensor_spec)
    super(CategoricalQPolicy, self).__init__(
        time_step_spec, action_spec, q_network.state_spec)

    if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
      raise TypeError('action_spec must be a BoundedTensorSpec. Got: %s' % (
          action_spec,))

    self._temperature = tf.convert_to_tensor(temperature, dtype=tf.float32)
    self._min_q_value = min_q_value
    self._max_q_value = max_q_value
    self._num_atoms = q_network.num_atoms
    self._q_network = q_network
    self._support = tf.linspace(min_q_value, max_q_value, self._num_atoms)
    self._action_dtype = action_spec.dtype

  def _variables(self):
    return self._q_network.variables

  def _distribution(self, time_step, policy_state):
    """Generates the distribution over next actions given the time_step.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of
        Tensors representing the previous policy_state.

    Returns:
      A tfp.distributions.Categorical capturing the distribution of next
        actions.
      A policy_state Tensor, or a nested dict, list or tuple of Tensors,
        representing the new policy state.
    """
    q_logits, policy_state = self._q_network(time_step.observation,
                                             time_step.step_type,
                                             policy_state)
    q_logits.shape.assert_has_rank(3)
    q_values = common.convert_q_logits_to_values(q_logits, self._support)

    logits = q_values
    mask_split_fn = self._q_network.mask_split_fn

    if mask_split_fn:
      _, mask = mask_split_fn(time_step.observation)
      # Overwrite the logits for invalid actions to -inf.
      neg_inf = tf.constant(-np.inf, dtype=logits.dtype)
      logits = tf.compat.v2.where(tf.cast(mask, tf.bool), logits, neg_inf)

    dist = tfp.distributions.Categorical(
        logits=logits, dtype=self.action_spec.dtype)
    return policy_step.PolicyStep(dist, policy_state)
