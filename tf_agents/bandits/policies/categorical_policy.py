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

"""Policy that chooses actions based on a categorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.utils import common
from tf_agents.utils import nest_utils

tfd = tfp.distributions


def _validate_weights(weights):
  if len(weights.shape) != 1:
    raise ValueError(
        'Expected a 1D `Tensor` of weights; got {}.'.format(weights))


class CategoricalPolicy(tf_policy.TFPolicy):
  """Policy that chooses an action based on a categorical distribution.

  The distribution is specified by a set of weights for each action and an
  inverse temperature. The unnormalized probability distribution is given by
  `exp(weight * inv_temp)`. Weights and inverse temperature are typically
  maintained as `Variable`s, and are updated by an `Agent`.

  Note that this policy does not make use of `time_step.observation` at all.
  That is, it is a non-contextual bandit policy.
  """

  def __init__(self,
               weights,
               time_step_spec,
               action_spec,
               inverse_temperature=1.,
               emit_log_probability=True,
               name=None):
    """Initializes `CategoricalPolicy`.

    The `weights` and `inverse_temperature` arguments may be either `Tensor`s or
      `tf.Variable`s. If they are variables, then any assignments to those
      variables will be reflected in the output of the policy. The shape of
      `weights` is used to determine the action_spec for this policy;
      `action_spec.maximum = weights.shape[0]`.


    If `emit_log_probability=True`, the info field of the `PolicyStep` returned
      will be a `PolicyInfo` tuple with log-probability that can be accessed
      using `policy_step.get_log_probability(step.info)`.

    Args:
      weights: a vector of weights, corresponding to the unscaled log
        probabilities of a categorical distribution.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A `tensor_spec` of action specification.
      inverse_temperature: a float value used to scale `weights`. Lower values
        will induce a more uniform distribution over actions; higher values will
        result in a sharper distribution.
      emit_log_probability: Whether to emit log probabilities or not.
      name: The name of this policy.

    Raises:
      ValueError: If the number of actions specified by the action_spec does not
        match the dimension of weights.
    """
    _validate_weights(weights)
    self._weights = weights
    self._inverse_temperature = inverse_temperature
    if action_spec.maximum + 1 != tf.compat.dimension_value(weights.shape[0]):
      raise ValueError(
          'Number of actions ({}) does not match weights dimension ({}).'
          .format(action_spec.maximum + 1,
                  tf.compat.dimension_value(weights.shape[0])))
    super(CategoricalPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        emit_log_probability=True,
        name=name)

  def _variables(self):
    return [v for v in [self._weights, self._inverse_temperature]
            if isinstance(v, tf.Variable)]

  def _distribution(self, time_step, policy_state):
    """Implementation of `distribution`. Returns a `Categorical` distribution.

    The returned `Categorical` distribution has (unnormalized) probabilities
    `exp(inverse_temperature * weights)`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: Unused in `CategoricalPolicy`. It is simply passed through.

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: A (optionally nested) of tfp.distribution.Distribution
          capturing the distribution of next actions.
        `state`: A policy state tensor for the next call to distribution.
        `info`: Optional side information such as action log probabilities.
    """
    outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
    logits = (
        self._inverse_temperature *
        common.replicate(self._weights, outer_shape))
    action_distribution = tfd.Independent(
        tfd.Categorical(logits=logits, dtype=self._action_spec.dtype))
    return policy_step.PolicyStep(action_distribution, policy_state)
