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

"""Tests tf_agents.utils.network_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import utils


class NetworkUtilsTest(tf.test.TestCase):

  def test_flatten_and_unflatten_ops(self):
    batch_squash = utils.BatchSquash(2)

    tensor = tf.constant(0, shape=(5, 4, 3, 2, 1))
    flat = batch_squash.flatten(tensor)
    unflat = batch_squash.unflatten(flat)

    self.assertAllEqual((20, 3, 2, 1), flat.shape)
    self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_and_unflatten_ops_no_batch_dims(self):
    batch_squash = utils.BatchSquash(0)

    tensor = tf.constant(0, shape=(5, 4, 3, 2, 1))
    flat = batch_squash.flatten(tensor)
    unflat = batch_squash.unflatten(flat)

    self.assertAllEqual((1, 5, 4, 3, 2, 1), flat.shape)
    self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_flatten_and_unflatten_ops_one_batch_dims(self):
    batch_squash = utils.BatchSquash(1)

    tensor = tf.constant(0, shape=(5, 4, 3, 2, 1))
    flat = batch_squash.flatten(tensor)
    unflat = batch_squash.unflatten(flat)

    self.assertAllEqual((5, 4, 3, 2, 1), flat.shape)
    self.assertAllEqual((5, 4, 3, 2, 1), unflat.shape)

  def test_encode_state(self):
    state = tf.constant(0, shape=(5, 100, 100, 3), dtype=tf.float32)
    states = utils.encode_state(
        state, conv_layers=[(15, 3, 2)], fc_layers=(20, 10))
    self.assertAllEqual((5, 10), states.shape)

  def test_encode_state_no_batch_dim(self):
    state = tf.constant(0, shape=(100, 100, 3), dtype=tf.float32)
    states = utils.encode_state(
        state, conv_layers=[(15, 3, 2)], fc_layers=(20, 10))
    self.assertAllEqual((10,), states.shape)

  def test_encode_state_with_time_dim(self):
    state = tf.constant(0, shape=(5, 4, 100, 100, 3), dtype=tf.float32)
    states = utils.encode_state(
        state, conv_layers=[(15, 3, 2)], fc_layers=(20, 10))
    self.assertAllEqual((5, 4, 10), states.shape)

  def test_encode_state_no_conv(self):
    state = tf.constant(0, shape=(5, 100), dtype=tf.float32)
    states = utils.encode_state(state, fc_layers=(20, 10))
    self.assertAllEqual((5, 10), states.shape)

  def test_encode_state_no_batch_dim_no_conv(self):
    state = tf.constant(0, shape=(100,), dtype=tf.float32)
    states = utils.encode_state(state, fc_layers=(20, 10))
    self.assertAllEqual((10,), states.shape)

  def test_encode_state_with_time_dim_no_conv(self):
    state = tf.constant(0, shape=(5, 4, 100), dtype=tf.float32)
    states = utils.encode_state(state, fc_layers=(20, 10))
    self.assertAllEqual((5, 4, 10), states.shape)


if __name__ == '__main__':
  tf.test.main()
