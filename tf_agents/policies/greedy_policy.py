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

"""Policy implementation that generates greedy actions from another policy."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Text

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step


# TODO(b/131405384): Remove this once Deterministic does casting internally.
@tfp.experimental.register_composite
class DeterministicWithLogProb(tfp.distributions.Deterministic):
  """Thin wrapper around Deterministic that supports taking log_prob."""

  def _log_prob(self, x):
    """Takes log-probs by casting to tf.float32 instead of self.dtype."""
    return tf.math.log(tf.cast(self.prob(x), dtype=tf.float32))


@gin.configurable(module='tf_agents', denylist=['policy'])
class GreedyPolicy(tf_policy.TFPolicy):
  """Returns greedy samples of a given policy."""

  def __init__(self, policy: tf_policy.TFPolicy, name: Optional[Text] = None):
    """Builds a greedy TFPolicy wrapping the given policy.

    Args:
      policy: A policy implementing the tf_policy.TFPolicy interface.
      name: The name of this policy. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    super(GreedyPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=policy.emit_log_probability,
        name=name)
    self._wrapped_policy = policy

  @property
  def wrapped_policy(self) -> tf_policy.TFPolicy:
    return self._wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _distribution(self, time_step, policy_state):
    def dist_fn(dist):
      try:
        greedy_action = dist.mode()
      except NotImplementedError:
        raise ValueError("Your network's distribution does not implement mode "
                         "making it incompatible with a greedy policy.")

      return DeterministicWithLogProb(loc=greedy_action)

    distribution_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    return policy_step.PolicyStep(
        tf.nest.map_structure(dist_fn, distribution_step.action),
        distribution_step.state, distribution_step.info)
