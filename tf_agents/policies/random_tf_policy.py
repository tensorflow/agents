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
from __future__ import print_function

import tensorflow as tf

from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


class RandomTFPolicy(tf_policy.Base):
  """Returns random samples of the given action_spec."""

  def _variables(self):
    return []

  def _action(self, time_step, policy_state, seed):
    outer_dims = nest_utils.get_outer_shape(time_step, self._time_step_spec)

    action_ = tensor_spec.sample_spec_nest(
        self._action_spec, seed=seed, outer_dims=outer_dims)
    # TODO(b/78181147): Investigate why this control dependency is required.
    if time_step is not None:
      with tf.control_dependencies(tf.nest.flatten(time_step)):
        action_ = tf.nest.map_structure(tf.identity, action_)
    return policy_step.PolicyStep(action_, policy_state)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'RandomTFPolicy does not support distributions yet.')
