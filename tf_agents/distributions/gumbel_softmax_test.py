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

"""Tests for tf_agents.distributions.gumbel_softmax."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.distributions import gumbel_softmax


class GumbelSoftmaxTest(tf.test.TestCase):

  def testLogProb(self):
    temperature = 0.8
    logits = [.3, .1, .4]
    dist = gumbel_softmax.GumbelSoftmax(
        temperature, logits, validate_args=True)
    x = tf.constant([0, 0, 1])
    log_prob = self.evaluate(dist.log_prob(x))
    expected_log_prob = -0.972918868065
    self.assertAllClose(expected_log_prob, log_prob)

  def testSample(self):
    temperature = 0.8
    logits = [.3, .1, .4]
    dist = gumbel_softmax.GumbelSoftmax(
        temperature, logits, dtype=tf.int64, validate_args=True)
    actions = dist.convert_to_one_hot(dist.sample())
    self.assertEqual(actions.dtype, tf.int64)
    self.assertEqual(self.evaluate(tf.reduce_sum(actions, axis=-1)), 1)

  def testMode(self):
    temperature = 1.0
    logits = [.3, .1, .4]
    dist = gumbel_softmax.GumbelSoftmax(
        temperature, logits, validate_args=True)
    self.assertAllEqual(self.evaluate(dist.mode()),
                        self.evaluate(tf.constant([0, 0, 1])))


if __name__ == '__main__':
  tf.test.main()
