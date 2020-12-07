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

"""Driver utilities for use with bandit policies and environments."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.trajectories import trajectory
from tf_agents.typing import types

nest = tf.compat.v2.nest


def trajectory_for_bandit(initial_step: types.TimeStep,
                          action_step: types.PolicyStep,
                          final_step: types.TimeStep) -> types.NestedTensor:
  """Builds a trajectory from a single-step bandit episode.

  Since all episodes consist of a single step, the returned `Trajectory` has no
  time dimension. All input and output `Tensor`s/arrays are expected to have
  shape `[batch_size, ...]`.

  Args:
    initial_step: A `TimeStep` returned from `environment.step(...)`.
    action_step: A `PolicyStep` returned by `policy.action(...)`.
    final_step: A `TimeStep` returned from `environment.step(...)`.
  Returns:
    A `Trajectory` containing zeros for discount value and `StepType.LAST` for
    both `step_type` and `next_step_type`.
  """
  return trajectory.Trajectory(observation=initial_step.observation,
                               action=action_step.action,
                               policy_info=action_step.info,
                               reward=final_step.reward,
                               discount=final_step.discount,
                               step_type=initial_step.step_type,
                               next_step_type=final_step.step_type)
