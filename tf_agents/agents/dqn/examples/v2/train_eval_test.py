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

"""Tests for tf_agents.agents.dqn.examples.v2.train_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.dqn.examples.v2 import train_eval

FLAGS = flags.FLAGS


class TrainEval(tf.test.TestCase):

  def testDQNCartPole(self):
    if not tf.executing_eagerly():
      self.skipTest('Binary is eager-only.')

    root_dir = self.get_temp_dir()
    train_loss = train_eval.train_eval(root_dir,
                                       num_iterations=1,
                                       num_eval_episodes=1,
                                       initial_collect_steps=10)
    self.assertGreater(train_loss.loss, 0.0)

  def testRNNDQNMaskedCartPole(self):
    if not tf.executing_eagerly():
      self.skipTest('Binary is eager-only.')

    root_dir = self.get_temp_dir()
    train_loss = train_eval.train_eval(
        root_dir,
        env_name='MaskedCartPole-v0',
        train_sequence_length=2,
        initial_collect_steps=10,
        num_eval_episodes=1,
        num_iterations=1)
    self.assertGreater(train_loss.loss, 0.0)

if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
