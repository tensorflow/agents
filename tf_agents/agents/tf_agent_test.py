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

"""Tests for agents.tf_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.agents import tf_agent


class LossInfoTest(tf.test.TestCase):

  def testBaseLossInfo(self):
    loss_info = tf_agent.LossInfo(0.0, ())
    self.assertEqual(loss_info.loss, 0.0)
    self.assertTrue(isinstance(loss_info, tf_agent.LossInfo))


if __name__ == '__main__':
  tf.test.main()
