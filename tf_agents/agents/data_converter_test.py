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

"""Tests for agents.tf_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tf_agents.agents import data_converter

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils


class AsTrajectoryTest(tf.test.TestCase):

  def setUp(self):
    super(AsTrajectoryTest, self).setUp()
    self._data_context = data_converter.DataContext(
        time_step_spec=ts.TimeStep(step_type=(),
                                   reward=tf.TensorSpec((), tf.float32),
                                   discount=tf.TensorSpec((), tf.float32),
                                   observation=()),
        action_spec={'action1': tf.TensorSpec((), tf.float32)},
        info_spec=())

  def testSimple(self):
    converter = data_converter.AsTrajectory(self._data_context)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    converted = converter(traj)
    (traj, converted) = self.evaluate((traj, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, traj)

  def testPrunes(self):
    converter = data_converter.AsTrajectory(self._data_context)
    my_spec = self._data_context.trajectory_spec.replace(
        action={
            'action1': tf.TensorSpec((), tf.float32),
            'action2': tf.TensorSpec([4], tf.int32)
        })
    traj = tensor_spec.sample_spec_nest(my_spec, outer_dims=[2, 3])
    converted = converter(traj)
    expected = tf.nest.map_structure(lambda x: x, traj)
    del expected.action['action2']
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testFromBatchTimeTransition(self):
    converter = data_converter.AsTrajectory(self._data_context)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    transition = trajectory.to_transition(traj, traj)
    converted = converter(transition)
    (traj, converted) = self.evaluate((traj, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, traj)

  def testNoTimeDimensionRaises(self):
    converter = data_converter.AsTrajectory(self._data_context)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[3])
    with self.assertRaisesRegex(
        ValueError, r'tensors must have shape \`\[B, T\] \+ spec.shape\`'):
      converter(traj)

  def testTransitionNoTimeDimensionRaises(self):
    converter = data_converter.AsTrajectory(self._data_context)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2])
    transition = trajectory.to_transition(traj, traj)
    with self.assertRaisesRegex(
        ValueError, r'tensors must have shape \`\[B, T\] \+ spec.shape\`'):
      converter(transition)

  def testInvalidTimeDimensionRaises(self):
    converter = data_converter.AsTrajectory(
        self._data_context, sequence_length=4)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    with self.assertRaisesRegex(
        ValueError, r'has a time axis dim value \'3\' vs the expected \'4\''):
      converter(traj)


class AsTransitionTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._data_context = data_converter.DataContext(
        time_step_spec=ts.TimeStep(step_type=(),
                                   reward=tf.TensorSpec((), tf.float32),
                                   discount=tf.TensorSpec((), tf.float32),
                                   observation=()),
        action_spec={'action1': tf.TensorSpec((), tf.float32)},
        policy_state_spec=(),
        info_spec=())

    self._data_context_with_state = data_converter.DataContext(
        time_step_spec=ts.TimeStep(step_type=(),
                                   reward=tf.TensorSpec((), tf.float32),
                                   discount=tf.TensorSpec((), tf.float32),
                                   observation=tf.TensorSpec((2,), tf.float32)),
        action_spec={'action1': tf.TensorSpec((), tf.float32)},
        policy_state_spec=[tf.TensorSpec((2,), tf.float32),
                           tf.TensorSpec((2,), tf.float32)],
        info_spec=())

  def testSimple(self):
    converter = data_converter.AsTransition(
        self._data_context, squeeze_time_dim=True)
    transition = tensor_spec.sample_spec_nest(
        self._data_context.transition_spec, outer_dims=[2])
    converted = converter(transition)
    (transition, converted) = self.evaluate((transition, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, transition)

  def testPrunes(self):
    converter = data_converter.AsTransition(
        self._data_context, squeeze_time_dim=True)
    my_spec = self._data_context.transition_spec.replace(
        action_step=self._data_context.transition_spec.action_step.replace(
            action={
                'action1': tf.TensorSpec((), tf.float32),
                'action2': tf.TensorSpec([4], tf.int32)
            }))
    transition = tensor_spec.sample_spec_nest(my_spec, outer_dims=[2])
    converted = converter(transition)
    expected = tf.nest.map_structure(lambda x: x, transition)
    del expected.action_step.action['action2']
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testFromBatchTimeTrajectory(self):
    converter = data_converter.AsTransition(
        self._data_context, squeeze_time_dim=True)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[4, 2])  # [B, T=2]
    converted = converter(traj)
    expected = trajectory.to_transition(traj)
    # Remove the now-singleton time dim.
    expected = tf.nest.map_structure(lambda x: tf.squeeze(x, 1), expected)
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testTrajectoryInvalidTimeDimensionRaises(self):
    converter = data_converter.AsTransition(
        self._data_context, squeeze_time_dim=True)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    with self.assertRaisesRegex(
        ValueError, r'has a time axis dim value \'3\' vs the expected \'2\''):
      converter(traj)

  def testTrajectoryNotSingleStepTransition(self):
    converter = data_converter.AsTransition(self._data_context)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    converted = converter(traj)
    expected = trajectory.to_transition(traj)
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testValidateTransitionWithState(self):
    converter = data_converter.AsTransition(
        self._data_context_with_state, squeeze_time_dim=False)
    transition = tensor_spec.sample_spec_nest(
        self._data_context_with_state.transition_spec, outer_dims=[1, 2])
    pruned_action_step = transition.action_step._replace(
        state=tf.nest.map_structure(
            lambda t: t[:, 0, ...], transition.action_step.state))
    transition = transition._replace(action_step=pruned_action_step)
    converted = converter(transition)
    (transition, converted) = self.evaluate((transition, converted))


class AsNStepTransitionTest(tf.test.TestCase):

  def setUp(self):
    super(AsNStepTransitionTest, self).setUp()
    self._data_context = data_converter.DataContext(
        time_step_spec=ts.TimeStep(step_type=(),
                                   reward=tf.TensorSpec((), tf.float32),
                                   discount=tf.TensorSpec((), tf.float32),
                                   observation=()),
        action_spec={'action1': tf.TensorSpec((), tf.float32)},
        info_spec=())

  def testSimple(self):
    converter = data_converter.AsNStepTransition(
        self._data_context, gamma=0.5)
    transition = tensor_spec.sample_spec_nest(
        self._data_context.transition_spec, outer_dims=[2])
    converted = converter(transition)
    (transition, converted) = self.evaluate((transition, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, transition)

  def testPrunes(self):
    converter = data_converter.AsNStepTransition(
        self._data_context, gamma=0.5)
    my_spec = self._data_context.transition_spec.replace(
        action_step=self._data_context.transition_spec.action_step.replace(
            action={
                'action1': tf.TensorSpec((), tf.float32),
                'action2': tf.TensorSpec([4], tf.int32)
            }))
    transition = tensor_spec.sample_spec_nest(my_spec, outer_dims=[2])
    converted = converter(transition)
    expected = tf.nest.map_structure(lambda x: x, transition)
    del expected.action_step.action['action2']
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testFromBatchTimeTrajectory(self):
    converter = data_converter.AsNStepTransition(
        self._data_context, gamma=0.5)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[4, 2])  # [B, T=2]
    converted = converter(traj)
    expected = trajectory.to_n_step_transition(traj, gamma=0.5)
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testTrajectoryInvalidTimeDimensionRaises(self):
    converter = data_converter.AsNStepTransition(
        self._data_context, gamma=0.5, n=4)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    with self.assertRaisesRegex(
        ValueError, r'has a time axis dim value \'3\' vs the expected \'5\''):
      converter(traj)

  def testTrajectoryNotSingleStepTransition(self):
    converter = data_converter.AsNStepTransition(
        self._data_context, gamma=0.5)
    traj = tensor_spec.sample_spec_nest(self._data_context.trajectory_spec,
                                        outer_dims=[2, 3])
    converted = converter(traj)
    expected = trajectory.to_n_step_transition(traj, gamma=0.5)
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)


class AsHalfTransitionTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._data_context = data_converter.DataContext(
        time_step_spec=ts.TimeStep(step_type=(),
                                   reward=tf.TensorSpec((), tf.float32),
                                   discount=tf.TensorSpec((), tf.float32),
                                   observation=()),
        action_spec={'action1': tf.TensorSpec((), tf.float32)},
        info_spec=(),
        policy_state_spec=(),
        use_half_transition=True)

    self._data_context_with_state = data_converter.DataContext(
        time_step_spec=ts.TimeStep(step_type=(),
                                   reward=tf.TensorSpec((), tf.float32),
                                   discount=tf.TensorSpec((), tf.float32),
                                   observation=tf.TensorSpec((2,), tf.float32)),
        action_spec={'action1': tf.TensorSpec((), tf.float32)},
        info_spec=(),
        policy_state_spec=[tf.TensorSpec((2,), tf.float32),
                           tf.TensorSpec((2,), tf.float32)],
        use_half_transition=True)

  def testSimple(self):
    converter = data_converter.AsHalfTransition(
        self._data_context, squeeze_time_dim=True)
    transition = tensor_spec.sample_spec_nest(
        self._data_context.transition_spec, outer_dims=[1])
    converted = converter(transition)
    (transition, converted) = self.evaluate((transition, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, transition)

  def testPrunes(self):
    converter = data_converter.AsHalfTransition(
        self._data_context, squeeze_time_dim=True)
    my_spec = self._data_context.transition_spec.replace(
        action_step=self._data_context.transition_spec.action_step.replace(
            action={
                'action1': tf.TensorSpec((), tf.float32),
                'action2': tf.TensorSpec([4], tf.int32)
            }))
    transition = tensor_spec.sample_spec_nest(my_spec, outer_dims=[1])
    converted = converter(transition)
    expected = tf.nest.map_structure(lambda x: x, transition)
    del expected.action_step.action['action2']
    (expected, converted) = self.evaluate((expected, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, expected)

  def testValidateTransitionWithState(self):
    converter = data_converter.AsHalfTransition(
        self._data_context_with_state, squeeze_time_dim=False)
    transition = tensor_spec.sample_spec_nest(
        self._data_context_with_state.transition_spec, outer_dims=[1, 2])
    pruned_action_step = transition.action_step._replace(
        state=tf.nest.map_structure(
            lambda t: t[:, 0, ...], transition.action_step.state))
    transition = transition._replace(action_step=pruned_action_step)
    converted = converter(transition)
    (transition, converted) = self.evaluate((transition, converted))
    tf.nest.map_structure(self.assertAllEqual, converted, transition)

if __name__ == '__main__':
  test_utils.main()
