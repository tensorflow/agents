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
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, cast

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common


@gin.configurable()
class CategoricalQPolicy(tf_policy.TFPolicy):
  """Class to build categorical Q-policies."""

  def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedTensorSpec,
               q_network: network.Network,
               min_q_value: float,
               max_q_value: float,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None,
               temperature: types.Float = 1.0):
    """Builds a categorical Q-policy given a categorical Q-network.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A `BoundedTensorSpec` representing the actions.
      q_network: A network.Network to use for our policy.
      min_q_value: A float specifying the minimum Q-value, used for setting up
        the support.
      max_q_value: A float specifying the maximum Q-value, used for setting up
        the support.
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
        sure the provided `q_network` is compatible with the network-specific
        half of the output of the `observation_and_action_constraint_splitter`.
        In particular, `observation_and_action_constraint_splitter` will be
        called on the observation before passing to the network.
        If `observation_and_action_constraint_splitter` is None, action
        constraints are not applied.
      temperature: temperature for sampling, when close to 0.0 is arg_max.

    Raises:
      ValueError: if `q_network` does not have property `num_atoms`.
      TypeError: if `action_spec` is not a `BoundedTensorSpec`.
    """
    network_action_spec = getattr(q_network, 'action_spec', None)

    if network_action_spec is not None:
      action_spec = cast(tf.TypeSpec, action_spec)
      if not action_spec.is_compatible_with(network_action_spec):
        raise ValueError(
            'action_spec must be compatible with q_network.action_spec; '
            'instead got action_spec=%s, q_network.action_spec=%s' % (
                action_spec, network_action_spec))

    if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
      raise TypeError('action_spec must be a BoundedTensorSpec. Got: %s' % (
          action_spec,))

    action_spec = cast(tensor_spec.BoundedTensorSpec, action_spec)
    if action_spec.minimum != 0:
      raise ValueError(
          'Action specs should have minimum of 0, but saw: {0}.  If collecting '
          'from a python environment, consider using '
          'tf_agents.environments.wrappers.ActionOffsetWrapper.'
          .format(action_spec))

    num_actions = action_spec.maximum - action_spec.minimum + 1
    try:
      num_atoms = q_network.num_atoms
    except AttributeError:
      raise ValueError('Expected q_network to have property `num_atoms`, but '
                       'it doesn\'t. (Note: you likely want to use a '
                       'CategoricalQNetwork.) Network is: %s' % q_network)
    self._num_atoms = num_atoms

    network_utils.check_single_floating_network_output(
        q_network.create_variables(), (num_actions, num_atoms), str(q_network))

    super(CategoricalQPolicy, self).__init__(
        time_step_spec,
        action_spec,
        policy_state_spec=q_network.state_spec,
        observation_and_action_constraint_splitter=(
            observation_and_action_constraint_splitter))

    self._temperature = tf.convert_to_tensor(temperature, dtype=tf.float32)
    self._q_network = q_network

    # Generate support in numpy so that we can assign it to a constant and avoid
    # having a tensor property.
    support = np.linspace(min_q_value, max_q_value, self._num_atoms,
                          dtype=np.float32)
    self._support = tf.constant(support, dtype=tf.float32)
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
    network_observation = time_step.observation
    observation_and_action_constraint_splitter = (
        self.observation_and_action_constraint_splitter)

    if observation_and_action_constraint_splitter is not None:
      network_observation, mask = (
          observation_and_action_constraint_splitter(network_observation))

    q_logits, policy_state = self._q_network(
        network_observation, step_type=time_step.step_type,
        network_state=policy_state)
    q_logits.shape.assert_has_rank(3)
    q_values = common.convert_q_logits_to_values(q_logits, self._support)

    logits = q_values

    if observation_and_action_constraint_splitter is not None:
      # Overwrite the logits for invalid actions to -inf.
      neg_inf = tf.constant(-np.inf, dtype=logits.dtype)
      logits = tf.compat.v2.where(tf.cast(mask, tf.bool), logits, neg_inf)

    action_spec = cast(tf.TensorSpec, self.action_spec)
    dist = tfp.distributions.Categorical(
        logits=logits, dtype=action_spec.dtype)

    return policy_step.PolicyStep(dist, policy_state)
