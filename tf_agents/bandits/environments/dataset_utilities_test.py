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

"""Tests for tf_agents.bandits.environments.dataset_utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.bandits.environments import dataset_utilities


class DatasetUtilitiesTest(tf.test.TestCase):

  def testOneHot(self):
    data = np.array([[1, 2], [1, 3], [2, 2], [1, 1]], dtype=np.int32)
    encoded = dataset_utilities._one_hot(data)
    expected = [[1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0]]
    np.testing.assert_array_equal(encoded, expected)

  def testRewardDistribution(self):
    reward_distr = dataset_utilities.mushroom_reward_distribution(
        r_noeat=0.0,
        r_eat_safe=5.0,
        r_eat_poison_bad=-35.0,
        r_eat_poison_good=5.0,
        prob_poison_bad=0.5)
    self.assertAllEqual(reward_distr.mean(), [[0, -15.], [0, 5.]])


if __name__ == '__main__':
  tf.test.main()
