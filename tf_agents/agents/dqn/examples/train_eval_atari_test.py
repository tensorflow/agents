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

"""Tests for tf_agents.agents.dqn.examples.train_eval_atari."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing.absltest import mock
import numpy as np
import tensorflow as tf

from tf_agents.agents.dqn.examples import train_eval_atari
from tf_agents.environments import time_step as ts
from tf_agents.environments import trajectory
from tf_agents.policies import policy_step

FLAGS = flags.FLAGS


# TimeStep and Trajectory constructors that only take observations (for
# simplicity):


def ts_restart(observation):
  return ts.restart(observation=observation)


def ts_transition(observation):
  return ts.transition(
      observation=observation, reward=np.array(1, dtype=np.float32))


def ts_termination(observation):
  return ts.termination(
      observation=observation, reward=np.array(1, dtype=np.float32))


def trajectory_first(observation):
  return trajectory.first(
      observation=observation, action=1, policy_info=(),
      reward=np.array(1, dtype=np.float32), discount=1.0)


def trajectory_mid(observation):
  return trajectory.mid(
      observation=observation, action=1, policy_info=(),
      reward=np.array(1, dtype=np.float32), discount=1.0)


def trajectory_last(observation, discount=0.0):
  return trajectory.last(
      observation=observation, action=1, policy_info=(),
      reward=np.array(1, dtype=np.float32), discount=discount)


class AtariTerminalOnLifeLossTest(tf.test.TestCase):

  def _setup_mocks(self):
    self.trainer = train_eval_atari.TrainEval(
        self.get_temp_dir(), 'Pong-v0', terminal_on_life_loss=True)

    self.trainer._env = mock.MagicMock()
    self.trainer._env.envs[0].game_over = False
    self.trainer._replay_buffer = mock.MagicMock()
    self.trainer._collect_policy = mock.MagicMock()
    action_step = policy_step.PolicyStep(action=1)
    self.trainer._collect_policy.action.return_value = action_step
    self.observer = mock.MagicMock()
    self.metric_observers = [self.observer]

  def testRegularStep(self):
    self._setup_mocks()

    with self.cached_session() as sess:
      self.trainer._initialize_graph(sess)

      time_step = ts_restart(0)

      # Run a regular step.
      self.trainer._env.step.return_value = ts_transition(1)
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)
      self.assertTrue(time_step.is_mid())
      self.trainer._replay_buffer.add_batch.assert_called_with(
          trajectory_first(0))
      self.observer.assert_called_with(trajectory_first(0))

  def testLifeLoss(self):
    self._setup_mocks()

    with self.cached_session() as sess:
      self.trainer._initialize_graph(sess)

      time_step = ts_restart(0)

      # Run a regular step.
      self.trainer._env.step.return_value = ts_transition(1)
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)

      self.trainer._replay_buffer.add_batch.reset_mock()
      self.observer.reset_mock()

      # Lose a life, but not the end of a game.
      self.trainer._env.step.side_effect = [
          ts_termination(2),
          ts_transition(3),
      ]
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)
      self.assertTrue(time_step.is_mid())
      expected_rb_calls = [
          mock.call(trajectory_last(1, discount=1.0)),
          mock.call(trajectory_first(2))
      ]
      self.assertEqual(
          expected_rb_calls,
          self.trainer._replay_buffer.add_batch.call_args_list)
      expected_observer_calls = [
          mock.call(trajectory_mid(1)),
          mock.call(trajectory_mid(2)),
      ]
      self.assertEqual(expected_observer_calls, self.observer.call_args_list)

  def testRegularStepAfterLifeLoss(self):
    self._setup_mocks()

    with self.cached_session() as sess:
      self.trainer._initialize_graph(sess)

      time_step = ts_restart(0)

      # Run a regular step.
      self.trainer._env.step.return_value = ts_transition(1)
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)

      # Lose a life, but not the end of a game.
      self.trainer._env.step.return_value = None
      self.trainer._env.step.side_effect = [
          ts_termination(2),
          ts_transition(3),
      ]
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)

      self.trainer._replay_buffer.add_batch.reset_mock()
      self.observer.reset_mock()

      # Run a regular step.
      self.trainer._env.step.return_value = ts_transition(4)
      self.trainer._env.step.side_effect = None
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)
      self.assertTrue(time_step.is_mid())
      self.trainer._replay_buffer.add_batch.assert_called_with(
          trajectory_mid(3))
      self.observer.assert_called_with(trajectory_mid(3))

  def testGameOver(self):
    self._setup_mocks()

    with self.cached_session() as sess:
      self.trainer._initialize_graph(sess)

      time_step = ts_restart(0)

      # Run a regular step.
      self.trainer._env.step.return_value = ts_transition(1)
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)

      # Lose a life, but not the end of a game.
      self.trainer._env.step.return_value = None
      self.trainer._env.step.side_effect = [
          ts_termination(2),
          ts_transition(3),
      ]
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)

      # Run a regular step.
      self.trainer._env.step.return_value = ts_transition(4)
      self.trainer._env.step.side_effect = None
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)

      self.trainer._replay_buffer.add_batch.reset_mock()
      self.observer.reset_mock()

      # Set game_over.
      self.trainer._env.envs[0].game_over = True
      self.trainer._env.step.return_value = ts_termination(5)
      time_step = self.trainer._collect_step(
          time_step, self.metric_observers, train=True)
      self.assertTrue(time_step.is_last())
      self.trainer._replay_buffer.add_batch.assert_called_with(
          trajectory_last(4))
      self.observer.assert_called_with(trajectory_last(4))


if __name__ == '__main__':
  tf.test.main()
