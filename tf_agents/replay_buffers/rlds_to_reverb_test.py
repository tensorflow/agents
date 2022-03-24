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

"""Tests for tf_agents.replay_buffers.rlds_to_reverb."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List, Tuple

from absl.testing import parameterized

import reverb
from rlds import rlds_types
import tensorflow as tf

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import rlds_to_reverb
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils

SEQUENCE_LENGTH = 2
STRIDE_LENGTH = 1
REVERB_TABLE_SIZE = 10000
REVERB_RATE_LIMITER = 1
OBSERVATIONS = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
ACTIONS = [[11.0], [21.0], [31.0]]
REWARDS = [1.0, 2.0, 3.0]
DISCOUNTS = [1.0, 1.0, 1.0]


def generate_valid_episodes(
) -> Dict[str, Tuple[tf.data.Dataset, List[trajectory.Trajectory]]]:
  """Get test data for valid RLDS datasets to be used across differrent tests.

  Returns:
    A dict representing all valid cases as keys and tuples of valid input RLDS
    and expected TF-Agents trajectories as values.
  """
  complete_steps = tf.data.Dataset.from_tensor_slices({
      rlds_types.OBSERVATION: OBSERVATIONS,
      rlds_types.ACTION: ACTIONS,
      rlds_types.REWARD: REWARDS,
      rlds_types.DISCOUNT: DISCOUNTS,
      rlds_types.IS_TERMINAL: [False, False, True],
      rlds_types.IS_LAST: [False, False, True],
      rlds_types.IS_FIRST: [True, False, False],
  })
  complete_episode_trajectories = [
      trajectory.Trajectory(0, [1.0, 2.0], [11.0], (), 1, 1.0, 1.0),
      trajectory.Trajectory(1, [3.0, 4.0], [21.0], (), 2, 2.0, 1.0),
      trajectory.Trajectory(2, [5.0, 6.0], [31.0], (), 0, 3.0, 0.0),
  ]

  truncated_steps = tf.data.Dataset.from_tensor_slices({
      rlds_types.OBSERVATION: OBSERVATIONS,
      rlds_types.ACTION: ACTIONS,
      rlds_types.REWARD: REWARDS,
      rlds_types.DISCOUNT: DISCOUNTS,
      rlds_types.IS_TERMINAL: [False, False, False],
      rlds_types.IS_LAST: [False, False, True],
      rlds_types.IS_FIRST: [True, False, False],
  })
  truncated_episode_trajectories = [
      trajectory.Trajectory(0, [1.0, 2.0], [11.0], (), 1, 1.0, 1.0),
      trajectory.Trajectory(1, [3.0, 4.0], [21.0], (), 2, 2.0, 1.0),
      trajectory.Trajectory(2, [5.0, 6.0], [31.0], (), 0, 3.0, 1.0),
  ]

  single_step = tf.data.Dataset.from_tensor_slices({
      rlds_types.OBSERVATION: [[1.0, 2.0]],
      rlds_types.ACTION: [[11.0]],
      rlds_types.REWARD: [1.0],
      rlds_types.DISCOUNT: [1.0],
      rlds_types.IS_TERMINAL: [False],
      rlds_types.IS_LAST: [True],
      rlds_types.IS_FIRST: [True],
  })
  single_step_episode_trajectories = [
      trajectory.Trajectory(2, [1.0, 2.0], [11.0], (), 2, 1.0, 1.0),
  ]

  return {
      'complete_episode': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [complete_steps],
      }), complete_episode_trajectories),
      'truncated_episode': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [truncated_steps],
      }), truncated_episode_trajectories),
      'single_step_episode': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [single_step],
      }), single_step_episode_trajectories),
      'multiple_episodes': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [complete_steps, single_step, truncated_steps],
      }), complete_episode_trajectories + single_step_episode_trajectories +
                            truncated_episode_trajectories)
  }


def generate_invalid_episodes() -> Dict[str, Tuple[tf.data.Dataset, str]]:
  """Get test data for invalid RLDS datasets to be used across differrent tests.

  Returns:
    A dict representing all invalid cases as keys and tuples of invalid input
    RLDS and expected error messages as values.
  """
  return {
      'incorrect_ending': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [
              tf.data.Dataset.from_tensor_slices({
                  rlds_types.OBSERVATION: OBSERVATIONS,
                  rlds_types.ACTION: ACTIONS,
                  rlds_types.REWARD: REWARDS,
                  rlds_types.DISCOUNT: DISCOUNTS,
                  rlds_types.IS_TERMINAL: [False, False, False],
                  rlds_types.IS_LAST: [False, False, False],
                  rlds_types.IS_FIRST: [True, False, True],
              })
          ],
      }), 'Mid step should not be followed by a first step.'),
      'incorrect_termination': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [
              tf.data.Dataset.from_tensor_slices({
                  rlds_types.OBSERVATION: OBSERVATIONS,
                  rlds_types.ACTION: ACTIONS,
                  rlds_types.REWARD: REWARDS,
                  rlds_types.DISCOUNT: DISCOUNTS,
                  rlds_types.IS_TERMINAL: [False, False, True],
                  rlds_types.IS_LAST: [False, False, False],
                  rlds_types.IS_FIRST: [True, False, False],
              })
          ],
      }), 'Terminal step must be the last step of an episode.'),
      'incorrect_beginning': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [
              tf.data.Dataset.from_tensor_slices({
                  rlds_types.OBSERVATION: OBSERVATIONS,
                  rlds_types.ACTION: ACTIONS,
                  rlds_types.REWARD: REWARDS,
                  rlds_types.DISCOUNT: DISCOUNTS,
                  rlds_types.IS_TERMINAL: [False, True, False],
                  rlds_types.IS_LAST: [False, True, False],
                  rlds_types.IS_FIRST: [True, False, False],
              })
          ],
      }), 'Last step of an episode must be followed by a first step.'),
      'different_spec_episode': (tf.data.Dataset.from_tensor_slices({
          rlds_types.STEPS: [
              tf.data.Dataset.from_tensor_slices({
                  rlds_types.OBSERVATION: [[1.0], [3.0], [5.0]],
                  rlds_types.ACTION: ACTIONS,
                  rlds_types.REWARD: REWARDS,
                  rlds_types.DISCOUNT: DISCOUNTS,
                  rlds_types.IS_TERMINAL: [False, True, False],
                  rlds_types.IS_LAST: [False, True, False],
                  rlds_types.IS_FIRST: [True, False, False],
              })
          ],
      }), 'Replay buffer table signature spec should match RLDS data spec.'),
      'no_step_episode':
          (tf.data.Dataset.from_tensors({'random': True}),
           f'No dataset representing RLDS {rlds_types.STEPS} exist in the data.'
          ),
      'incorrect_step_spec': (tf.data.Dataset.from_tensors({
          rlds_types.STEPS:
              tf.data.Dataset.from_tensors({
                  'random1': [True, True],
                  'random2': [False, False]
              })
      }), f'Invalid RLDS step spec. Features expected are {rlds_to_reverb.get_rlds_step_features()}'
                              ', but found [\'random1\', \'random2\']')
  }


class RldsToReverbTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(RldsToReverbTest, self).setUp()

    self._valid_episodes = generate_valid_episodes()
    self._invalid_episodes = generate_invalid_episodes()

    # Data spec corresponding to our test data. This data spec is used for
    #  1) Validation of create_trajectory_data_spec.
    #  2) Initializing Reverb server and Reverb Replay Buffer.
    self._data_spec = trajectory.Trajectory(
        tensor_spec.TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
        tensor_spec.TensorSpec(shape=(2,), dtype=tf.float32, name=None),
        tensor_spec.TensorSpec(shape=(1,), dtype=tf.float32, name=None), (),
        tensor_spec.TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
        tensor_spec.TensorSpec(shape=(), dtype=tf.float32, name=None),
        tensor_spec.BoundedTensorSpec(
            shape=(),
            dtype=tf.float32,
            name='discount',
            minimum=[0.],
            maximum=[1.]))

    self._table_name = 'test_table'

    # Initialize Reverb server, Reverb Replay Buffer and Reverb Observer.
    uniform_table = reverb.Table(
        name=self._table_name,
        max_size=REVERB_TABLE_SIZE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(REVERB_RATE_LIMITER),
        signature=tensor_spec.add_outer_dim(self._data_spec))

    self._server = reverb.Server([uniform_table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        self._data_spec,
        sequence_length=SEQUENCE_LENGTH,
        table_name=self._table_name,
        local_server=self._server)

    self._reverb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        self._table_name,
        sequence_length=SEQUENCE_LENGTH,
        stride_length=STRIDE_LENGTH)

  def tearDown(self):
    if self._server:
      # Stop the Reverb server if it is running.
      self._server.stop()
      self._server = None
    super(RldsToReverbTest, self).tearDown()

  @parameterized.named_parameters(
      ('_complete_episode', 'complete_episode'),
      ('_truncated_episode', 'truncated_episode'),
      ('_single_step_episode', 'single_step_episode'),
      ('_multiple_episodes', 'multiple_episodes'))
  def test_trajectory_data_spec_valid_episodes(self, episode):
    rlds_data, _ = self._valid_episodes[episode]
    self.assertEqual(
        rlds_to_reverb.create_trajectory_data_spec(rlds_data), self._data_spec)

  @parameterized.named_parameters(
      ('_no_step_episode', 'no_step_episode'),
      ('_incorrect_step_spec', 'incorrect_step_spec'))
  def test_trajectory_data_spec_no_step_episode(self, episode):
    rlds_data, error_message = self._invalid_episodes[episode]
    with self.assertRaises(ValueError) as err:
      rlds_to_reverb.create_trajectory_data_spec(rlds_data)
    self.assertEqual(str(err.exception), error_message)

  @parameterized.named_parameters(
      ('_complete_episode', 'complete_episode', 3),
      ('_truncated_episode', 'truncated_episode', 3),
      ('_single_step_episode', 'single_step_episode', 1),
      ('_multiple_episodes', 'multiple_episodes', 7))
  def test_push_to_reverb_valid_episodes(self, episode,
                                         expected_trajectories_pushed):
    rlds_data, _ = self._valid_episodes[episode]
    trajectories_pushed = rlds_to_reverb.push_rlds_to_reverb(
        rlds_data, self._reverb_observer)  # type: int
    self.assertEqual(trajectories_pushed, expected_trajectories_pushed)

  @parameterized.named_parameters(
      ('_incorrect_ending', 'incorrect_ending'),
      ('_incorrect_termination', 'incorrect_termination'),
      ('_incorrect_beginning', 'incorrect_beginning'))
  def test_push_to_reverb_invalid_episodes(self, episode):
    rlds_data, error_message = self._invalid_episodes[episode]
    with self.assertRaises(tf.errors.InvalidArgumentError) as err:
      rlds_to_reverb.push_rlds_to_reverb(rlds_data, self._reverb_observer)
    self.assertRegex(str(err.exception), error_message)

  @parameterized.named_parameters(
      ('_different_spec_episode', 'different_spec_episode'),
      ('_no_step_episode', 'no_step_episode'),
      ('_incorrect_step_spec', 'incorrect_step_spec'))
  def test_push_to_reverb_invalid_episodes_value_errors(self, episode):
    rlds_data, error_message = self._invalid_episodes[episode]
    with self.assertRaises(ValueError) as err:
      rlds_to_reverb.push_rlds_to_reverb(rlds_data, self._reverb_observer)
    self.assertEqual(str(err.exception), error_message)


class RldsToTrajectoriesTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(RldsToTrajectoriesTest, self).setUp()

    self._valid_episodes = generate_valid_episodes()
    self._invalid_episodes = generate_invalid_episodes()

  @parameterized.named_parameters(
      ('_complete_episode', 'complete_episode'),
      ('_truncated_episode', 'truncated_episode'),
      ('_single_step_episode', 'single_step_episode'),
      ('_multiple_episodes', 'multiple_episodes'))
  def test_conversion_valid_episodes(self, episode):
    rlds_data, expected_trajectories = self._valid_episodes[episode]
    generated_trajectories = rlds_to_reverb.convert_rlds_to_trajectories(
        rlds_data)  # type: tf.data.Dataset
    for generated_trajectory, expected_trajectory in zip(
        list(generated_trajectories.as_numpy_iterator()),
        expected_trajectories):
      self.assertEqual(generated_trajectory.step_type,
                       expected_trajectory.step_type)
      self.assertEqual(generated_trajectory.next_step_type,
                       expected_trajectory.next_step_type)
      self.assertAllEqual(generated_trajectory.observation,
                          expected_trajectory.observation)
      self.assertAllEqual(generated_trajectory.action,
                          expected_trajectory.action)
      self.assertAllEqual(generated_trajectory.discount,
                          expected_trajectory.discount)
      self.assertAllEqual(generated_trajectory.reward,
                          expected_trajectory.reward)

  @parameterized.named_parameters(
      ('_incorrect_ending', 'incorrect_ending'),
      ('_incorrect_termination', 'incorrect_termination'),
      ('_incorrect_beginning', 'incorrect_beginning'))
  def test_conversion_invalid_episodes(self, episode):
    rlds_data, error_message = self._invalid_episodes[episode]
    with self.assertRaises(tf.errors.InvalidArgumentError) as err:
      list(
          rlds_to_reverb.convert_rlds_to_trajectories(
              rlds_data).as_numpy_iterator())
    self.assertRegex(str(err.exception), error_message)

  @parameterized.named_parameters(
      ('_no_step_episode', 'no_step_episode'),
      ('_incorrect_step_spec', 'incorrect_step_spec'))
  def test_conversion_no_step_episodes(self, episode):
    rlds_data, error_message = self._invalid_episodes[episode]
    with self.assertRaises(ValueError) as err:
      rlds_to_reverb.convert_rlds_to_trajectories(rlds_data)
    self.assertEqual(str(err.exception), error_message)


if __name__ == '__main__':
  test_utils.main()
