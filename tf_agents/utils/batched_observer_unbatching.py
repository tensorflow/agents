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

# coding=utf-8
# Copyright 2022 The TF-Agents Authors.
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
"""Adapter for using unbatched observers in batched contexts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable

from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.utils import nest_utils


class BatchedObserverUnbatching(object):
  """Creates an unbatching observer.

  Creates an observer that takes batched trajectories, unbatches them, and
  delegates them to multiple observers.

  The unbatched trajectories are delegated to observers that don't support
  batch dimensions (e.g. ReverbAddEpisodeObserver).

  Note that the batch size is assumed to be fixed and it is not validated.
  """

  def __init__(self, create_delegated_observer_fn: Callable[[], Callable[
      [trajectory_lib.Trajectory], None]], batch_size: int):
    self._delegated_observers = [
        create_delegated_observer_fn() for _ in range(batch_size)
    ]

  def __call__(self, batched_trajectory: trajectory_lib.Trajectory):
    unbatched_trajectories = nest_utils.unstack_nested_arrays(
        batched_trajectory)
    for obs, traj in zip(self._delegated_observers, unbatched_trajectories):
      # The for loop can be optimized by parallelizing running delegated
      # observers in the future.
      obs(traj)
