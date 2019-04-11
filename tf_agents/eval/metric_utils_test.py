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

"""Test for tf_agents.eval.metric_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.environments import random_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec


class MetricUtilsTest(tf.test.TestCase):

  def testMetricIsComputedCorrectly(self):

    def reward_fn(*unused_args):
      reward = np.random.uniform()
      reward_fn.total_reward += reward
      return reward

    reward_fn.total_reward = 0

    action_spec = array_spec.BoundedArraySpec((1,), np.int32, -10, 10)
    observation_spec = array_spec.BoundedArraySpec((1,), np.int32, -10, 10)
    env = random_py_environment.RandomPyEnvironment(
        observation_spec, action_spec, reward_fn=reward_fn)
    policy = random_py_policy.RandomPyPolicy(
        time_step_spec=None, action_spec=action_spec)

    average_return = py_metrics.AverageReturnMetric()

    num_episodes = 10
    results = metric_utils.compute([average_return], env, policy, num_episodes)
    self.assertAlmostEqual(reward_fn.total_reward / num_episodes,
                           results[average_return.name], places=5)

if __name__ == '__main__':
  tf.test.main()
