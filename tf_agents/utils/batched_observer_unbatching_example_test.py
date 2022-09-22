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

"""Tests for tf_agents.utils.batched_observer_unbatching_example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.utils import batched_observer_unbatching_example


class BatchedObserverUnbatchingExampleTest(tf.test.TestCase):

    def test_collect_random(self):
        batched_observer_unbatching_example.collect_random(
            num_episodes=10, num_envs=5)


if __name__ == '__main__':
    tf.test.main()
