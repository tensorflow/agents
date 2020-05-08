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

"""Tests for tf_agents.bandits.agents.loss_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import loss_utils


tf.compat.v1.enable_v2_behavior()


class LossUtilsTest(tf.test.TestCase):

  def testBaseCase(self):
    # Example taken from:
    # https://en.wikipedia.org/wiki/Quantile_regression
    # Random variable takes values 1...9 with equal probability.
    y_true = tf.constant(np.arange(1, 10), dtype=tf.float32)

    # Compute the loss for the median.
    # We see that the value `y_pred = 5` minimizes the loss.

    p_loss = loss_utils.pinball_loss(
        y_true, y_pred=3 * tf.ones_like(y_true), quantile=0.5)
    self.assertNear(24.0, 9.0 / 0.5 * self.evaluate(p_loss), err=1e-3)

    p_loss = loss_utils.pinball_loss(
        y_true, y_pred=4 * tf.ones_like(y_true), quantile=0.5)
    self.assertNear(21.0, 9.0 / 0.5 * self.evaluate(p_loss), err=1e-3)

    p_loss = loss_utils.pinball_loss(
        y_true, y_pred=5 * tf.ones_like(y_true), quantile=0.5)
    self.assertNear(20.0, 9.0 / 0.5 * self.evaluate(p_loss), err=1e-3)

    p_loss = loss_utils.pinball_loss(
        y_true, y_pred=6 * tf.ones_like(y_true), quantile=0.5)
    self.assertNear(21.0, 9.0 / 0.5 * self.evaluate(p_loss), err=1e-3)

    p_loss = loss_utils.pinball_loss(
        y_true, y_pred=7 * tf.ones_like(y_true), quantile=0.5)
    self.assertNear(24.0, 9.0 / 0.5 * self.evaluate(p_loss), err=1e-3)


if __name__ == '__main__':
  tf.test.main()
