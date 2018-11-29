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

"""Tests for tf_agents.drivers.trajectory_replay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.environments import time_step as ts
from tf_agents.environments import trajectory
from tf_agents.environments import trajectory_replay
from tf_agents.specs import tensor_spec

nest = tf.contrib.framework.nest


def make_random_trajectory():
  time_step_spec = ts.time_step_spec(
      tensor_spec.TensorSpec([], tf.int64, name='observation'))
  action_spec = tensor_spec.BoundedTensorSpec([],
                                              tf.int32,
                                              minimum=1,
                                              maximum=2,
                                              name='action')
  # info and policy state specs match that of TFPolicyMock.
  outer_dims = [1, 6]  # (batch_size, time)
  traj = trajectory.Trajectory(
      observation=tensor_spec.sample_spec_nest(
          time_step_spec.observation, outer_dims=outer_dims),
      action=tensor_spec.sample_bounded_spec(
          action_spec, outer_dims=outer_dims),
      policy_info=tensor_spec.sample_bounded_spec(
          action_spec, outer_dims=outer_dims),
      reward=tf.fill(outer_dims, 0.0),
      # step_type is F M L F M L.
      step_type=tf.reshape(tf.range(0, 6) % 3, outer_dims),
      # next_step_type is M L F M L F.
      next_step_type=tf.reshape(tf.range(1, 7) % 3, outer_dims),
      discount=tf.fill(outer_dims, 1.0),
  )
  return traj, time_step_spec, action_spec


class TrajectoryReplayTest(tf.test.TestCase):

  def _compare_to_original(self,
                           output_actions,
                           output_policy_info,
                           traj):
    # policy_info & action between collected & original is different because the
    # policy will be emitting different outputs.
    self.assertFalse(
        np.all(np.isclose(output_policy_info,
                          traj.policy_info)))
    self.assertFalse(
        np.all(np.isclose(output_actions,
                          traj.action)))

  def testReplayBufferObservers(self):
    traj, time_step_spec, action_spec = make_random_trajectory()
    policy = driver_test_utils.TFPolicyMock(time_step_spec, action_spec)
    replay = trajectory_replay.TrajectoryReplay(policy)
    output_actions, output_policy_info, _ = replay.run(traj)
    new_traj = traj._replace(
        action=output_actions,
        policy_info=output_policy_info)
    repeat_output_actions, repeat_output_policy_info, _ = replay.run(new_traj)
    self.evaluate(tf.global_variables_initializer())
    (output_actions, output_policy_info, traj,
     repeat_output_actions, repeat_output_policy_info) = self.evaluate(
         (output_actions, output_policy_info, traj,
          repeat_output_actions, repeat_output_policy_info))

    # Ensure output actions & policy info don't match original trajectory.
    self._compare_to_original(output_actions, output_policy_info, traj)

    # Ensure repeated run with the same deterministic policy recreates the same
    # actions & policy info.
    nest.map_structure(
        self.assertAllEqual, output_actions, repeat_output_actions)
    nest.map_structure(
        self.assertAllEqual, output_policy_info, repeat_output_policy_info)

  def testReplayBufferObserversWithInitialState(self):
    traj, time_step_spec, action_spec = make_random_trajectory()
    policy = driver_test_utils.TFPolicyMock(time_step_spec, action_spec)
    policy_state = policy.get_initial_state(1)
    replay = trajectory_replay.TrajectoryReplay(policy)
    output_actions, output_policy_info, _ = replay.run(
        traj, policy_state=policy_state)
    new_traj = traj._replace(
        action=output_actions,
        policy_info=output_policy_info)
    repeat_output_actions, repeat_output_policy_info, _ = replay.run(
        new_traj, policy_state=policy_state)
    self.evaluate(tf.global_variables_initializer())
    (output_actions, output_policy_info, traj,
     repeat_output_actions, repeat_output_policy_info) = self.evaluate(
         (output_actions, output_policy_info, traj,
          repeat_output_actions, repeat_output_policy_info))

    # Ensure output actions & policy info don't match original trajectory.
    self._compare_to_original(output_actions, output_policy_info, traj)

    # Ensure repeated run with the same deterministic policy recreates the same
    # actions & policy info.
    nest.map_structure(
        self.assertAllEqual, output_actions, repeat_output_actions)
    nest.map_structure(
        self.assertAllEqual, output_policy_info, repeat_output_policy_info)


if __name__ == '__main__':
  tf.test.main()
