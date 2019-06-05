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

"""A Driver-like object that replays Trajectories."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class TrajectoryReplay(object):
  """A helper that replays a policy against given `Trajectory` observations.

  """

  def __init__(
      self,
      policy,
      time_major=False):
    """Creates a TrajectoryReplay object.

    TrajectoryReplay.run returns the actions and policy info of the new policy
    assuming it saw the observations from the given trajectory.

    Args:
      policy: A tf_policy.Base policy.
      time_major: If `True`, the tensors in `trajectory` passed to method `run`
        are assumed to have shape `[time, batch, ...]`.  Otherwise (default)
        they are assumed to have shape `[batch, time, ...]`.

    Raises:
      ValueError:
        If policy is not an instance of tf_policy.Base.
    """
    self._policy = policy
    self._time_major = time_major

  def run(self, trajectory, policy_state=None):
    """Apply the policy to trajectory steps and store actions/info.

    If `self.time_major == True`, the tensors in `trajectory` are assumed to
    have shape `[time, batch, ...]`.  Otherwise they are assumed to
    have shape `[batch, time, ...]`.

    Args:
      trajectory: The `Trajectory` to run against.
        If the replay class was created with `time_major=True`, then
        the tensors in trajectory must be shaped `[time, batch, ...]`.
        Otherwise they must be shaped `[batch, time, ...]`.
      policy_state: (optional) A nest Tensor with initial step policy state.

    Returns:
      output_actions: A nest of the actions that the policy took.
        If the replay class was created with `time_major=True`, then
        the tensors here will be shaped `[time, batch, ...]`.  Otherwise
        they'll be shaped `[batch, time, ...]`.
      output_policy_info: A nest of the policy info that the policy emitted.
        If the replay class was created with `time_major=True`, then
        the tensors here will be shaped `[time, batch, ...]`.  Otherwise
        they'll be shaped `[batch, time, ...]`.
      policy_state: A nest Tensor with final step policy state.

    Raises:
      TypeError: If `policy_state` structure doesn't match
        `self.policy.policy_state_spec`, or `trajectory` structure doesn't
        match `self.policy.trajectory_spec`.
      ValueError: If `policy_state` doesn't match
        `self.policy.policy_state_spec`, or `trajectory` structure doesn't
        match `self.policy.trajectory_spec`.
      ValueError: If `trajectory` lacks two outer dims.
    """
    trajectory_spec = self._policy.trajectory_spec
    outer_dims = nest_utils.get_outer_shape(trajectory, trajectory_spec)

    if tf.compat.dimension_value(outer_dims.shape[0]) != 2:
      raise ValueError(
          "Expected two outer dimensions, but saw '{}' dimensions.\n"
          "Trajectory:\n{}.\nTrajectory spec from policy:\n{}.".format(
              tf.compat.dimension_value(outer_dims.shape[0]), trajectory,
              trajectory_spec))
    if self._time_major:
      sequence_length = outer_dims[0]
      batch_size = outer_dims[1]
      static_batch_size = tf.compat.dimension_value(
          trajectory.discount.shape[1])
    else:
      batch_size = outer_dims[0]
      sequence_length = outer_dims[1]
      static_batch_size = tf.compat.dimension_value(
          trajectory.discount.shape[0])

    if policy_state is None:
      policy_state = self._policy.get_initial_state(batch_size)
    else:
      tf.nest.assert_same_structure(policy_state,
                                    self._policy.policy_state_spec)

    if not self._time_major:
      # Make trajectory time-major.
      trajectory = tf.nest.map_structure(common.transpose_batch_time,
                                         trajectory)

    trajectory_tas = tf.nest.map_structure(
        lambda t: tf.TensorArray(t.dtype, size=sequence_length).unstack(t),
        trajectory)

    def create_output_ta(spec):
      return tf.TensorArray(
          spec.dtype, size=sequence_length,
          element_shape=(tf.TensorShape([static_batch_size])
                         .concatenate(spec.shape)))

    output_action_tas = tf.nest.map_structure(create_output_ta,
                                              trajectory_spec.action)
    output_policy_info_tas = tf.nest.map_structure(create_output_ta,
                                                   trajectory_spec.policy_info)

    read0 = lambda ta: ta.read(0)
    zeros_like0 = lambda t: tf.zeros_like(t[0])
    ones_like0 = lambda t: tf.ones_like(t[0])
    time_step = ts.TimeStep(
        step_type=read0(trajectory_tas.step_type),
        reward=tf.nest.map_structure(zeros_like0, trajectory.reward),
        discount=ones_like0(trajectory.discount),
        observation=tf.nest.map_structure(read0, trajectory_tas.observation))

    def process_step(time, time_step, policy_state,
                     output_action_tas, output_policy_info_tas):
      """Take an action on the given step, and update output TensorArrays.

      Args:
        time: Step time.  Describes which row to read from the trajectory
          TensorArrays and which location to write into in the output
          TensorArrays.
        time_step: Previous step's `TimeStep`.
        policy_state: Policy state tensor or nested structure of tensors.
        output_action_tas: Nest of `tf.TensorArray` containing new actions.
        output_policy_info_tas: Nest of `tf.TensorArray` containing new
          policy info.

      Returns:
        policy_state: The next policy state.
        next_output_action_tas: Updated `output_action_tas`.
        next_output_policy_info_tas: Updated `output_policy_info_tas`.
      """
      action_step = self._policy.action(time_step, policy_state)
      policy_state = action_step.state
      write_ta = lambda ta, t: ta.write(time - 1, t)
      next_output_action_tas = tf.nest.map_structure(
          write_ta, output_action_tas, action_step.action)
      next_output_policy_info_tas = tf.nest.map_structure(
          write_ta, output_policy_info_tas, action_step.info)

      return (action_step.state,
              next_output_action_tas,
              next_output_policy_info_tas)

    def loop_body(time, time_step, policy_state,
                  output_action_tas, output_policy_info_tas):
      """Runs a step in environment.

      While loop will call multiple times.

      Args:
        time: Step time.
        time_step: Previous step's `TimeStep`.
        policy_state: Policy state tensor or nested structure of tensors.
        output_action_tas: Updated nest of `tf.TensorArray`, the new actions.
        output_policy_info_tas: Updated nest of `tf.TensorArray`, the new
          policy info.

      Returns:
        loop_vars for next iteration of tf.while_loop.
      """
      policy_state, next_output_action_tas, next_output_policy_info_tas = (
          process_step(time, time_step, policy_state,
                       output_action_tas,
                       output_policy_info_tas))

      ta_read = lambda ta: ta.read(time)
      ta_read_prev = lambda ta: ta.read(time - 1)
      time_step = ts.TimeStep(
          step_type=ta_read(trajectory_tas.step_type),
          observation=tf.nest.map_structure(ta_read,
                                            trajectory_tas.observation),
          reward=tf.nest.map_structure(ta_read_prev, trajectory_tas.reward),
          discount=ta_read_prev(trajectory_tas.discount))

      return (time + 1, time_step, policy_state,
              next_output_action_tas, next_output_policy_info_tas)

    time = tf.constant(1)
    time, time_step, policy_state, output_action_tas, output_policy_info_tas = (
        tf.while_loop(
            cond=lambda time, *_: time < sequence_length,
            body=loop_body,
            loop_vars=[time, time_step, policy_state,
                       output_action_tas, output_policy_info_tas],
            back_prop=False,
            name="trajectory_replay_loop"))

    # Run the last time step
    last_policy_state, output_action_tas, output_policy_info_tas = (
        process_step(time, time_step, policy_state,
                     output_action_tas, output_policy_info_tas))

    def stack_ta(ta):
      t = ta.stack()
      if not self._time_major:
        t = common.transpose_batch_time(t)
      return t

    stacked_output_actions = tf.nest.map_structure(stack_ta, output_action_tas)
    stacked_output_policy_info = tf.nest.map_structure(stack_ta,
                                                       output_policy_info_tas)

    return (stacked_output_actions,
            stacked_output_policy_info,
            last_policy_state)
