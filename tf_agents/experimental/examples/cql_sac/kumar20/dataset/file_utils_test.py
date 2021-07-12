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

"""Tests for file_utils."""
import mock
import numpy as np
import tensorflow as tf

from tf_agents.experimental.examples.cql_sac.kumar20.dataset.file_utils import create_trajectory
from tf_agents.experimental.examples.cql_sac.kumar20.dataset.file_utils import create_transition
from tf_agents.experimental.examples.cql_sac.kumar20.dataset.file_utils import write_samples_to_tfrecord

from tf_agents.utils import test_utils

TFRECORD_OBSERVER_PREFIX = ('tf_agents.experimental.examples.cql_sac.' +
                            'kumar20.dataset.file_utils.' +
                            'example_encoding_dataset.TFRecordObserver')


class FileUtilsTest(test_utils.TestCase):

  @mock.patch('%s.__init__' % TFRECORD_OBSERVER_PREFIX)
  @mock.patch('%s.__call__' % TFRECORD_OBSERVER_PREFIX)
  @mock.patch('%s.close' % TFRECORD_OBSERVER_PREFIX)
  def test_write_transitions_to_tfrecord(self, mock_close, mock_call,
                                         mock_init):
    mock_init.return_value = None
    mock_close.return_value = None
    episode_dict = {
        'states':
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.],
                      [11., 12.], [13., 14.]]),
        'actions':
            np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.]]),
        'rewards':
            np.array([[0.], [1.], [0.], [1.], [0.], [0.], [1.]]),
        'discounts':
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        'episode_start_index':
            np.array([0, 2, 4])
    }

    first_transition = create_transition(
        state=np.array([5., 6.]),
        action=np.array([3.]),
        next_state=np.array([7., 8.]),
        discount=np.array(1.),
        reward=np.array([0.]),
        step_type=np.array(0),
        next_step_type=np.array(2))

    write_samples_to_tfrecord(
        episode_dict,
        first_transition,  # used as a dummy collect_data_spec
        dataset_path='dataset_path',
        start_episode=1,
        end_episode=3,
        use_trajectories=False)

    expected_transitions = [
        first_transition,
        create_transition(
            state=np.array([7., 8.]),
            action=np.array([4.]),
            next_state=np.array([0., 0.]),
            discount=np.array(0.),
            reward=np.array([1.]),
            step_type=np.array(2),
            next_step_type=np.array(0)),
        create_transition(
            state=np.array([9., 10.]),
            action=np.array([5.]),
            next_state=np.array([11., 12.]),
            discount=np.array(1.),
            reward=np.array([0.]),
            step_type=np.array(0),
            next_step_type=np.array(1)),
        create_transition(
            state=np.array([11., 12.]),
            action=np.array([6.]),
            next_state=np.array([13., 14.]),
            discount=np.array(1.),
            reward=np.array([0.]),
            step_type=np.array(1),
            next_step_type=np.array(2)),
        create_transition(
            state=np.array([13., 14.]),
            action=np.array([7.]),
            next_state=np.array([0., 0.]),
            discount=np.array(0.),
            reward=np.array([1.]),
            step_type=np.array(2),
            next_step_type=np.array(0))
    ]

    # Check that the transitions passed to the mock TFRecordObserver
    # match what we expect.
    num_transitions = len(expected_transitions)
    self.assertEqual(len(mock_call.call_args_list), num_transitions)
    for i in range(num_transitions):
      actual_transition = mock_call.call_args_list[i][0][0]
      ts_actual, ps_actual, next_ts_actual = actual_transition
      ts_expected, ps_expected, next_ts_expected = expected_transitions[i]

      self.assertAllEqual(ts_actual.step_type, ts_expected.step_type)
      self.assertAllEqual(ts_actual.observation, ts_expected.observation)
      self.assertAllEqual(ts_actual.reward, ts_expected.reward)
      self.assertAllEqual(ts_actual.discount, ts_expected.discount)

      self.assertAllEqual(next_ts_actual.step_type, next_ts_expected.step_type)
      self.assertAllEqual(next_ts_actual.observation,
                          next_ts_expected.observation)
      self.assertAllEqual(next_ts_actual.reward, next_ts_expected.reward)
      self.assertAllEqual(next_ts_actual.discount, next_ts_expected.discount)

      self.assertAllEqual(ps_actual.action, ps_expected.action)
      self.assertAllEqual(ps_actual.info, ps_expected.info)

  @mock.patch('%s.__init__' % TFRECORD_OBSERVER_PREFIX)
  @mock.patch('%s.__call__' % TFRECORD_OBSERVER_PREFIX)
  @mock.patch('%s.close' % TFRECORD_OBSERVER_PREFIX)
  def test_write_trajectories_to_tfrecord(self, mock_close, mock_call,
                                          mock_init):
    mock_init.return_value = None
    mock_close.return_value = None
    episode_dict = {
        'states':
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.],
                      [11., 12.], [13., 14.]]),
        'actions':
            np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.]]),
        'rewards':
            np.array([[0.], [1.], [0.], [1.], [0.], [0.], [1.]]),
        'discounts':
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        'episode_start_index':
            np.array([0, 2, 4])
    }

    first_trajectory = create_trajectory(
        state=np.array([5., 6.]),
        action=np.array([3.]),
        discount=np.array(1.),
        reward=np.array([0.]),
        step_type=np.array(0),
        next_step_type=np.array(2))

    write_samples_to_tfrecord(
        episode_dict,
        first_trajectory,  # used as a dummy collect_data_spec
        dataset_path='dataset_path',
        start_episode=1,
        end_episode=3,
        use_trajectories=True)

    expected_trajectories = [
        first_trajectory,
        create_trajectory(
            state=np.array([7., 8.]),
            action=np.array([4.]),
            discount=np.array(0.),
            reward=np.array([1.]),
            step_type=np.array(2),
            next_step_type=np.array(0)),
        create_trajectory(
            state=np.array([9., 10.]),
            action=np.array([5.]),
            discount=np.array(1.),
            reward=np.array([0.]),
            step_type=np.array(0),
            next_step_type=np.array(1)),
        create_trajectory(
            state=np.array([11., 12.]),
            action=np.array([6.]),
            discount=np.array(1.),
            reward=np.array([0.]),
            step_type=np.array(1),
            next_step_type=np.array(2)),
        create_trajectory(
            state=np.array([13., 14.]),
            action=np.array([7.]),
            discount=np.array(0.),
            reward=np.array([1.]),
            step_type=np.array(2),
            next_step_type=np.array(0))
    ]

    # Check that the trajectories passed to the mock TFRecordObserver
    # match what we expect.
    num_trajectories = len(expected_trajectories)
    self.assertEqual(len(mock_call.call_args_list), num_trajectories)
    for i in range(num_trajectories):
      actual_traj = mock_call.call_args_list[i][0][0]
      expected_traj = expected_trajectories[i]

      self.assertAllEqual(actual_traj.step_type, expected_traj.step_type)
      self.assertAllEqual(actual_traj.observation, expected_traj.observation)
      self.assertAllEqual(actual_traj.action, expected_traj.action)
      self.assertAllEqual(actual_traj.policy_info, expected_traj.policy_info)
      self.assertAllEqual(actual_traj.next_step_type,
                          expected_traj.next_step_type)
      self.assertAllEqual(actual_traj.reward, expected_traj.reward)
      self.assertAllEqual(actual_traj.discount, expected_traj.discount)


if __name__ == '__main__':
  tf.test.main()
