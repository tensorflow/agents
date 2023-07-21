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

"""Tests for tf_agents.drivers.test_utils."""
from absl.testing import parameterized
import numpy as np

from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils


class TestUtilsTest(parameterized.TestCase, test_utils.TestCase):

  @parameterized.named_parameters([
      ('BatchOfOneTrajectoryOfLengthThree', 1, 3),
      ('BatchOfOneTrajectoryOfLengthSeven', 1, 7),
      ('BatchOfOneTrajectoryOfLengthNine', 1, 9),
      ('BatchOfTwoTrajectorieOfLengthThree', 2, 3),
      ('BatchOfTwoTrajectorieOfLengthSeven', 2, 7),
      ('BatchOfTwoTrajectorieOfLengthNine', 2, 9),
      ('BatchOfFiveTrajectorieOfLengthThree', 5, 3),
      ('BatchOfFiveTrajectorieOfLengthSeven', 5, 7),
      ('BatchOfFiveTrajectorieOfLengthNine', 5, 9)
  ])
  def testNumEpisodesObserverEpisodeTotal(self, batch_size, traj_len):
    single_trajectory = np.concatenate([[ts.StepType.FIRST],
                                        np.repeat(ts.StepType.MID,
                                                  traj_len - 2),
                                        [ts.StepType.LAST]])
    step_type = np.tile(single_trajectory, (batch_size, 1))

    traj = trajectory.Trajectory(
        observation=np.random.rand(batch_size, traj_len),
        action=np.random.rand(batch_size, traj_len),
        policy_info=(),
        reward=np.random.rand(batch_size, traj_len),
        discount=np.ones((batch_size, traj_len)),
        step_type=step_type,
        next_step_type=np.zeros((batch_size, traj_len)))

    observer = driver_test_utils.NumEpisodesObserver()
    observer(traj)
    self.assertEqual(observer.num_episodes, batch_size)
