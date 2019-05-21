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

"""Utility functions for testing with trajectories."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.trajectories import trajectory


def stacked_trajectory_from_transition(time_step, action_step, next_time_step):
  """Given transitions, returns a time stacked `Trajectory`.

  The tensors of the produced `Trajectory` will have a time dimension added
  (i.e., a shape of `[B, T, ...]` where T = 2 in this case). The `Trajectory`
  can be used when calling `agent.train()` or passed directly to `to_transition`
  without the need for a `next_trajectory` argument.

  Args:
    time_step: A `time_step.TimeStep` representing the first step in a
      transition.
    action_step: A `policy_step.PolicyStep` representing actions corresponding
      to observations from time_step.
    next_time_step: A `time_step.TimeStep` representing the second step in a
      transition.

  Returns:
    A time stacked `Trajectory`.
  """
  # Note that we reuse action_step and next_time_step in experience2 in order to
  # ensure the action, policy_info, next_step_type, reward, and discount match
  # for both values of the time dimension.
  experience1 = trajectory.from_transition(
      time_step, action_step, next_time_step)
  experience2 = trajectory.from_transition(
      next_time_step, action_step, next_time_step)
  return tf.nest.map_structure(lambda x, y: tf.stack([x, y], axis=1),
                               experience1, experience2)

