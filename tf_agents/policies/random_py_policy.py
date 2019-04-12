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

import numpy as np
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils


class RandomPyPolicy(py_policy.Base):
  """Returns random samples of the given action_spec."""

  def __init__(self, time_step_spec, action_spec, seed=None, outer_dims=None):
    """Initializes the RandomPyPolicy.

    Args:
      time_step_spec: Reference `time_step_spec`. If not None and outer_dims
        is not provided this is used to infer the outer_dims required for the
        given time_step when action is called.
      action_spec: A nest of BoundedArraySpec representing the actions to sample
        from.
      seed: Optional seed used to instantiate a random number generator.
      outer_dims: An optional list/tuple specifying outer dimensions to add to
        the spec shape before sampling. If unspecified the outer_dims are
        derived from the outer_dims in the given observation when `action` is
        called.
    """

    self._seed = seed
    self._outer_dims = outer_dims
    self._rng = np.random.RandomState(seed)
    if time_step_spec is None:
      time_step_spec = ts.time_step_spec()

    super(RandomPyPolicy, self).__init__(
        time_step_spec=time_step_spec, action_spec=action_spec)

  def _action(self, time_step, policy_state):
    outer_dims = self._outer_dims
    if outer_dims is None:
      if self.time_step_spec.observation:
        outer_dims = nest_utils.get_outer_array_shape(
            time_step.observation, self.time_step_spec.observation)
      else:
        outer_dims = ()

    random_action = array_spec.sample_spec_nest(
        self._action_spec, self._rng, outer_dims=outer_dims)
    return policy_step.PolicyStep(random_action, policy_state)
