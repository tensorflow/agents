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

"""Tests for dataset_utils."""
import numpy as np
import tensorflow as tf

from tf_agents.experimental.examples.cql_sac.kumar20.dataset.dataset_utils import create_collect_data_spec
from tf_agents.experimental.examples.cql_sac.kumar20.dataset.dataset_utils import create_episode_dataset

from tf_agents.specs.array_spec import ArraySpec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory

from tf_agents.utils import test_utils


class UtilsTest(test_utils.TestCase):

  def assertDictEqual(self, a, b, msg=None):
    self.assertEqual(a.keys(), b.keys())

    for key in a:
      self.assertAllEqual(a[key], b[key])

  def test_create_episode_dataset(self):
    d4rl_dataset = {
        'observations': [[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.],
                         [11., 12.], [13., 14.]],
        'actions': [[1.], [2.], [3.], [4.], [5.], [6.], [7.]],
        'rewards': [[0.], [1.], [0.], [1.], [0.], [0.], [1.]],
        'terminals': [False, True, False, True, False, False, True],
        'timeouts': [False, False, False, False, False, False, False],
        'infos/goal': [[0.]] * 8
    }

    episode_dict = create_episode_dataset(d4rl_dataset, exclude_timeouts=True)
    expected_dict = {
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
    self.assertDictEqual(episode_dict, expected_dict)

  def test_create_episode_dataset_exclude_timeout(self):
    d4rl_dataset = {
        'observations': [[1., 2.], [3., 4.]],
        'actions': [[1.], [2.]],
        'rewards': [[0.], [0.]],
        'terminals': [False, False],
        'timeouts': [False, True],
        'infos/goal': [[10., 10.], [10., 10.]]
    }

    episode_dict = create_episode_dataset(d4rl_dataset, exclude_timeouts=True)

    # Threw out timeout step.
    expected_dict = {
        'states': np.array([[1., 2.]]),
        'actions': np.array([[1.]]),
        'rewards': np.array([[0.]]),
        'discounts': np.array([1.0]),
        'episode_start_index': np.array([0])
    }
    self.assertDictEqual(episode_dict, expected_dict)

  def test_create_episode_dataset_include_timeout(self):
    d4rl_dataset = {
        'observations': [[1., 2.], [3., 4.]],
        'actions': [[1.], [2.]],
        'rewards': [[0.], [0.]],
        'terminals': [False, False],
        'timeouts': [False, True],
        'infos/goal': [[10., 10.], [10., 10.]]
    }

    episode_dict = create_episode_dataset(d4rl_dataset, exclude_timeouts=False)

    expected_dict = {
        'states': np.array([[1., 2.], [3., 4.]]),
        'actions': np.array([[1.], [2.]]),
        'rewards': np.array([[0.], [0.]]),
        'discounts': np.array([1.0, 1.0]),
        'episode_start_index': np.array([0])
    }
    self.assertDictEqual(episode_dict, expected_dict)

  def test_create_episode_dataset_from_terminal(self):
    d4rl_dataset = {
        'observations': [[10., 10.]],
        'actions': [[5.]],
        'rewards': [[1.]],
        'terminals': [True],
        'timeouts': [False],
        'infos/goal': [[10., 10.]]
    }

    episode_dict = create_episode_dataset(d4rl_dataset, exclude_timeouts=True)

    # Immediately started at the goal. Not thrown out.
    expected_dict = {
        'states': np.array([[10., 10.]]),
        'actions': np.array([[5.]]),
        'rewards': np.array([[1.]]),
        'discounts': np.array([0.0]),
        'episode_start_index': np.array([0])
    }
    self.assertDictEqual(episode_dict, expected_dict)

  def test_collect_data_spec_transition(self):
    episode_dict = {
        'states':
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                     dtype=np.float32),
        'actions':
            np.array([[1.], [2.], [3.], [4.]], dtype=np.float32),
        'rewards':
            np.array([[0.], [1.], [0.], [1.]], dtype=np.float32),
        'discounts':
            np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        'episode_start_index':
            np.array([0, 2], dtype=np.int32)
    }

    time_step_spec = time_step.TimeStep(
        step_type=ArraySpec(shape=[], dtype=np.int32),
        reward=ArraySpec(shape=[1], dtype=np.float32),
        discount=ArraySpec(shape=[], dtype=np.float32),
        observation=ArraySpec(shape=[2], dtype=np.float32))
    action_spec = policy_step.PolicyStep(
        action=ArraySpec(shape=[1], dtype=np.float32), state=(), info=())
    expected_spec = trajectory.Transition(
        time_step=time_step_spec,
        action_step=action_spec,
        next_time_step=time_step_spec)
    actual_spec = create_collect_data_spec(episode_dict, use_trajectories=False)
    self.assertEqual(actual_spec, expected_spec)

  def test_collect_data_spec_trajectory(self):
    episode_dict = {
        'states':
            np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]],
                     dtype=np.float32),
        'actions':
            np.array([[1.], [2.], [3.], [4.]], dtype=np.float32),
        'rewards':
            np.array([[0.], [1.], [0.], [1.]], dtype=np.float32),
        'discounts':
            np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        'episode_start_index':
            np.array([0, 2], dtype=np.int32)
    }

    expected_spec = trajectory.Trajectory(
        step_type=ArraySpec(shape=[], dtype=np.int32),
        observation=ArraySpec(shape=[2], dtype=np.float32),
        action=ArraySpec(shape=[1], dtype=np.float32),
        policy_info=(),
        next_step_type=ArraySpec(shape=[], dtype=np.int32),
        reward=ArraySpec(shape=[1], dtype=np.float32),
        discount=ArraySpec(shape=[], dtype=np.float32))
    actual_spec = create_collect_data_spec(episode_dict, use_trajectories=True)
    self.assertEqual(actual_spec, expected_spec)


if __name__ == '__main__':
  tf.test.main()
