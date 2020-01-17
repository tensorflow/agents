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

"""Tests for episodic_replay_buffer using driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import test_utils as driver_test_utils
from tf_agents.environments import batched_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import test_utils


class EpisodicReplayBufferDriverTest(test_utils.TestCase):

  # Creates a test EpisodicReplayBuffer.
  def _make_replay_buffer(self, tf_env):
    """Default replay buffer factory."""

    time_step_spec = tf_env.time_step_spec()
    action_spec = tf_env.action_spec()
    action_step_spec = policy_step.PolicyStep(
        action_spec, (), tensor_spec.TensorSpec((), tf.int32))
    trajectory_spec = trajectory.from_transition(time_step_spec,
                                                 action_step_spec,
                                                 time_step_spec)
    return episodic_replay_buffer.EpisodicReplayBuffer(
        trajectory_spec, end_episode_fn=lambda _: False)

  def testMultiStepEpisodicReplayBuffer(self):
    num_episodes = 5
    num_driver_episodes = 5

    # Create mock environment.
    py_env = batched_py_environment.BatchedPyEnvironment([
        driver_test_utils.PyEnvironmentMock(final_state=i+1)
        for i in range(num_episodes)
    ])
    env = tf_py_environment.TFPyEnvironment(py_env)

    # Creat mock policy.
    policy = driver_test_utils.TFPolicyMock(
        env.time_step_spec(), env.action_spec(), batch_size=num_episodes)

    # Create replay buffer and driver.
    replay_buffer = self._make_replay_buffer(env)
    stateful_buffer = episodic_replay_buffer.StatefulEpisodicReplayBuffer(
        replay_buffer, num_episodes)
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env, policy, num_episodes=num_driver_episodes,
        observers=[stateful_buffer.add_batch])

    run_driver = driver.run()

    end_episodes = replay_buffer._maybe_end_batch_episodes(
        stateful_buffer.episode_ids, end_episode=True)

    completed_episodes = replay_buffer._completed_episodes()

    self.evaluate([
        tf.compat.v1.local_variables_initializer(),
        tf.compat.v1.global_variables_initializer()
    ])

    self.evaluate(run_driver)

    self.evaluate(end_episodes)
    completed_episodes = self.evaluate(completed_episodes)
    eps = [replay_buffer._get_episode(ep) for ep in completed_episodes]
    eps = self.evaluate(eps)

    episodes_length = [tf.nest.flatten(ep)[0].shape[0] for ep in eps]

    # Compare with expected output.
    self.assertAllEqual(completed_episodes, [3, 4, 5, 6, 7])
    self.assertAllEqual(episodes_length, [4, 4, 2, 1, 1])

    first = ts.StepType.FIRST
    mid = ts.StepType.MID
    last = ts.StepType.LAST

    step_types = [ep.step_type for ep in eps]
    observations = [ep.observation for ep in eps]
    rewards = [ep.reward for ep in eps]
    actions = [ep.action for ep in eps]

    self.assertAllClose([[first, mid, mid, last], [first, mid, mid, mid],
                         [first, last], [first], [first]], step_types)

    self.assertAllClose([
        [0, 1, 3, 4],
        [0, 1, 3, 4],
        [0, 1],
        [0],
        [0],
    ], observations)

    self.assertAllClose([
        [1, 2, 1, 2],
        [1, 2, 1, 2],
        [1, 2],
        [1],
        [1],
    ], actions)

    self.assertAllClose([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 0],
        [1],
        [1],
    ], rewards)


if __name__ == '__main__':
  tf.test.main()
