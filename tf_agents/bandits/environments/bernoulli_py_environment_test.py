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

"""Tests for the Bernoulli Bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import bernoulli_py_environment


class BernoulliBanditPyEnvironmentTest(tf.test.TestCase):

  def test_bernoulli_bandit_py_environment(self):

    env = bernoulli_py_environment.BernoulliPyEnvironment(
        [0.1, 0.2, 0.3], batch_size=2)
    observation_step = env.reset()
    self.assertAllEqual(observation_step.observation.shape, [2])
    reward_step = env.step([0, 1])
    self.assertAllEqual(len(reward_step.reward), 2)

  def test_out_of_bound_parameter(self):
    with self.assertRaisesRegexp(
        ValueError, r'All parameters should be floats in \[0, 1\]\.'):
      bernoulli_py_environment.BernoulliPyEnvironment(
          [0.1, 1.2, 0.3], batch_size=1)


if __name__ == '__main__':
  tf.test.main()
