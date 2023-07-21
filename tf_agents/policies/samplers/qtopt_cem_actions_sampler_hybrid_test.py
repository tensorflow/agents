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

"""Tests for tf_agents.policies.samplers.qtopt_cem_actions_sampler_hybrid."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies.samplers import qtopt_cem_actions_sampler_hybrid
from tf_agents.specs import tensor_spec


_BATCH = 2
_NUM_SAMPLES = 10
_ACTION_SIZE = 3
_MEAN = [0., 0., 0.0]
_VAR = [0.09, 0.03, 0.05]


class ActionsSamplerTest(tf.test.TestCase):

  def testSampleBatch(self):
    action_spec = (
        tensor_spec.BoundedTensorSpec([_ACTION_SIZE], tf.float32, 0.0, 1.0),
        tensor_spec.BoundedTensorSpec([_ACTION_SIZE], tf.int32, 0, 1))
    sampler = qtopt_cem_actions_sampler_hybrid.GaussianActionsSampler(
        action_spec=action_spec)

    mean = tf.constant(_MEAN)
    var = tf.constant(_VAR)
    mean = (tf.broadcast_to(mean, [_BATCH, _ACTION_SIZE]),
            tf.broadcast_to(mean, [_BATCH, _ACTION_SIZE]))
    var = (tf.broadcast_to(var, [_BATCH, _ACTION_SIZE]),
           tf.broadcast_to(var, [_BATCH, _ACTION_SIZE]))

    actions = sampler.sample_batch_and_clip(_NUM_SAMPLES, mean, var)
    self.assertEqual([_BATCH, _NUM_SAMPLES, _ACTION_SIZE], actions[0].shape)
    self.assertEqual([_BATCH, _NUM_SAMPLES, _ACTION_SIZE], actions[1].shape)

    actions_ = self.evaluate(actions)
    flat_actions = tf.nest.flatten(actions_)
    flat_action_spec = tf.nest.flatten(action_spec)
    for i in range(len(flat_action_spec)):
      self.assertTrue((flat_actions[i] <= flat_action_spec[i].maximum).all())
      self.assertTrue((flat_actions[i] >= flat_action_spec[i].minimum).all())

  def testInvalidActionSpec(self):
    action_spec = [
        tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE, _ACTION_SIZE], tf.float32, 0., 1.)]
    with self.assertRaisesRegex(
        ValueError, 'Only 1d action is supported by this sampler.*'):
      qtopt_cem_actions_sampler_hybrid.GaussianActionsSampler(
          action_spec=action_spec)

if __name__ == '__main__':
  tf.test.main()
