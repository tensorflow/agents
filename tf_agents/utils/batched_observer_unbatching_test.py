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
"""Tests for tf_agents.utils.batched_observer_unbatching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import reverb
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import py_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gym
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory as trajectory_lib
from tf_agents.utils import batched_observer_unbatching


class BatchedObserverUnbatchingTest(tf.test.TestCase):

  def test_call(self):
    trajectories = []

    def observer(traj):
      trajectories.append(traj)

    def observer_fn():
      return observer

    unbatcher = batched_observer_unbatching.BatchedObserverUnbatching(
        observer_fn, batch_size=2)

    trajectory = trajectory_lib.Trajectory(
        action=tf.constant([0, 1]),
        discount=tf.constant([0, 0]),
        next_step_type=tf.constant([1, 2]),
        observation={
            "a": tf.constant([24, 42]),
            "b": tf.constant([100, 200]),
        },
        policy_info=tf.constant([500, 1000]),
        reward=tf.constant([25, 50]),
        step_type=tf.constant([13, 26]),
    )
    unbatcher(trajectory)

    self.assertEqual(
        trajectories,
        [
            trajectory_lib.Trajectory(
                action=tf.constant([0]),
                discount=tf.constant([0]),
                next_step_type=tf.constant([1]),
                observation={
                    "a": tf.constant([24]),
                    "b": tf.constant([100]),
                },
                policy_info=tf.constant([500]),
                reward=tf.constant([25]),
                step_type=tf.constant([13]),
            ),
            trajectory_lib.Trajectory(
                action=tf.constant([1]),
                discount=tf.constant([0]),
                next_step_type=tf.constant([2]),
                observation={
                    "a": tf.constant([42]),
                    "b": tf.constant([200]),
                },
                policy_info=tf.constant([1000]),
                reward=tf.constant([50]),
                step_type=tf.constant([26]),
            ),
        ],
    )

  def test_reverb_integration(self):
    num_envs = 3
    env = parallel_py_environment.ParallelPyEnvironment(
        [lambda: suite_gym.load("CartPole-v0")] * num_envs)

    policy = random_py_policy.RandomPyPolicy(env.time_step_spec(),
                                             env.action_spec())

    replay_buffer_signature = tensor_spec.from_spec(policy.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
    table = reverb.Table(
        "experience",
        max_size=100,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )
    reverb_server = reverb.Server([table])

    def create_add_episode_observer():
      return reverb_utils.ReverbAddEpisodeObserver(
          reverb_server.localhost_client(),
          table_name="experience",
          max_sequence_length=200,
      )

    rb_observer = batched_observer_unbatching.BatchedObserverUnbatching(
        create_add_episode_observer, batch_size=num_envs)

    driver = py_driver.PyDriver(
        env, policy, observers=[rb_observer], max_episodes=30)
    driver.run(env.reset())


if __name__ == "__main__":
  tf.test.main()
