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

"""Policy implementation that steps over a given configuration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence, Tuple, Optional
from absl import logging

import numpy as np

from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


class ScriptedPyPolicy(py_policy.PyPolicy):
  """Returns actions from the given configuration."""

  def __init__(self, time_step_spec: ts.TimeStep,
               action_spec: types.NestedArraySpec,
               action_script: Sequence[Tuple[int, types.NestedArray]]):
    """Instantiates the scripted policy.

    The Action  script can be configured through gin. e.g:

    ScriptedPyPolicy.action_script = [
        (1, {  "action1": [[5, 2], [1, 3]],
               "action2": [[4, 6]]},),
        (0, {  "action1": [[8, 1], [9, 2]],
               "action2": [[1, 2]]},),
        (2, {  "action1": [[1, 1], [3, 2]],
               "action2": [[8, 2]]},),
    ]

    In this case the first action is executed once, the second scripted action
    is disabled and skipped. Then the third listed action is executed for two
    steps.

    Args:
      time_step_spec: A time_step_spec for the policy will interact
        with.
      action_spec: An action_spec for the environment the policy will interact
        with.
      action_script: A list of 2-tuples of the form (n, nest) where the nest of
        actions follow the action_spec. Each action will be executed for n
        steps.
    """
    if time_step_spec is None:
      time_step_spec = ts.time_step_spec()
    super(ScriptedPyPolicy, self).__init__(
        time_step_spec=time_step_spec, action_spec=action_spec)

    self._action_script = action_script

  def _get_initial_state(self, batch_size):
    del batch_size
    # We use the state to keep track of the action index to execute and to count
    # how many times it has been performed.
    return [0, 0]

  def _action(self, time_step, policy_state, seed: Optional[types.Seed] = None):
    del time_step  # Unused.
    del seed  # Unused.
    if policy_state is None:
      policy_state = [0, 0]

    action_index, num_repeats = policy_state  #  pylint: disable=unpacking-non-sequence

    def _check_episode_length():
      if action_index >= len(self._action_script):
        raise ValueError(
            "Episode is longer than the provided scripted policy. Consider "
            "setting a TimeLimit wrapper that stops episodes within the length"
            " of your scripted policy.")

    _check_episode_length()
    n, current_action = self._action_script[action_index]

    # If the policy has been executed n times get the next scripted action.
    # Allow users to disable entries in the scripted policy by setting n <= 0.
    while num_repeats >= n:
      action_index += 1
      num_repeats = 0
      _check_episode_length()
      n, current_action = self._action_script[action_index]

    num_repeats += 1

    # To make it easier for the user we allow the actions in the script to be
    # lists instead of numpy arrays. Checking the arrays_nest requires us to
    # have the leaves be objects and not lists so we lift them into numpy
    # arrays.
    def actions_as_array(action_spec, action):
      return np.asarray(action, dtype=action_spec.dtype)

    current_action = nest.map_structure_up_to(
        self._action_spec, actions_as_array, self._action_spec, current_action)

    if not array_spec.check_arrays_nest(current_action, self._action_spec):
      raise ValueError(
          "Action at index {} does not match the environment's action_spec. "
          "Got: {}. Expected {}.".format(action_index, current_action,
                                         self._action_spec))

    logging.info("Policy_state: %r", policy_state)
    return policy_step.PolicyStep(current_action, [action_index, num_repeats])
