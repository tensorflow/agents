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

"""Tests for the Bernoulli Bandit environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import piecewise_bernoulli_py_environment as pbe


class PiecewiseBernoulliBanditPyEnvironmentTest(tf.test.TestCase,
                                                parameterized.TestCase):

  def deterministic_duration_generator(self):
    while True:
      yield 10

  def test_out_of_bound_parameter(self):
    with self.assertRaisesRegexp(
        ValueError, r'All parameters should be floats in \[0, 1\]\.'):
      pbe.PiecewiseBernoulliPyEnvironment(
          [[0.1, 1.2, 0.3]], self.deterministic_duration_generator())

  @parameterized.named_parameters(
      dict(testcase_name='_batch_1',
           batch_size=1),
      dict(testcase_name='_batch_4',
           batch_size=4),
  )
  def test_correct_piece(self, batch_size):
    env = pbe.PiecewiseBernoulliPyEnvironment(
        [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.1, 0.12, 0.14]],
        self.deterministic_duration_generator(), batch_size)
    for t in range(100):
      env.reset()
      self.assertEqual(int(t / 10) % 3, env._current_piece)
      _ = env.step([0])


if __name__ == '__main__':
  tf.test.main()
