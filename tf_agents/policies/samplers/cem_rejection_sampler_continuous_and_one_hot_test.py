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

"""Tests for tf_agents.policies.samplers.cem_rejection_sampler_continuous_and_one_hot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tf_agents.policies.samplers import cem_rejection_sampler_continuous_and_one_hot
from tf_agents.specs import tensor_spec


_BATCH = 2
_NUM_SAMPLES = 10
_ACTION_SIZE_CONTINUOUS = 3
_ACTION_SIZE_DISCRETE = 1
_MEAN = [0., 0., 0.0]
_VAR = [0.09, 0.03, 0.05]


class ActionsSamplerTest(tf.test.TestCase):

  def _sample_validator(self, num_samples_continuous, mean, var, state):
    del state
    def sample(mean, var):
      dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))
      return dist.sample(num_samples_continuous)

    # [N, B, A]
    samples_continuous = tf.nest.map_structure(
        sample, mean, var)
    return samples_continuous

  def testSampleBatch(self):
    action_spec = {
        'continuous': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0),
        'discrete': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_DISCRETE], tf.int32, 0, 1)}
    sampler = cem_rejection_sampler_continuous_and_one_hot.GaussianActionsSampler(
        action_spec=action_spec, sample_validator={
            'continuous': self._sample_validator,
            'discrete': None},
        sub_actions_fields=[['discrete'], ['continuous']])

    mean = tf.constant(_MEAN)
    var = tf.constant(_VAR)
    mean = {
        'continuous': tf.broadcast_to(mean, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'discrete': tf.zeros([_BATCH, _ACTION_SIZE_DISCRETE])}
    var = {
        'continuous': tf.broadcast_to(var, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'discrete': tf.zeros([_BATCH, _ACTION_SIZE_DISCRETE])}

    actions = sampler.sample_batch_and_clip(_NUM_SAMPLES, mean, var)

    self.assertEqual((_BATCH, _NUM_SAMPLES, _ACTION_SIZE_CONTINUOUS),
                     actions['continuous'].shape)
    self.assertEqual((_BATCH, _NUM_SAMPLES, _ACTION_SIZE_DISCRETE),
                     actions['discrete'].shape)

    actions_ = self.evaluate(actions)
    flat_actions = tf.nest.flatten(actions_)
    flat_action_spec = tf.nest.flatten(action_spec)
    for i in range(len(flat_action_spec)):
      self.assertTrue((flat_actions[i] <= flat_action_spec[i].maximum).all())
      self.assertTrue((flat_actions[i] >= flat_action_spec[i].minimum).all())

  def testInvalidActionSpec(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0)
    with self.assertRaisesRegex(
        ValueError, r'Only continuous action \+ 1 one_hot action is supported'
        ' by this sampler.*'):
      cem_rejection_sampler_continuous_and_one_hot.GaussianActionsSampler(
          action_spec=action_spec, sample_validator={
              'continuous': self._sample_validator,
              'discrete': None},
          sub_actions_fields=[['discrete'], ['continuous']])

    action_spec = [
        tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_DISCRETE, _ACTION_SIZE_DISCRETE],
            tf.float32, 0.0, 1.0),
        tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0)]
    with self.assertRaisesRegex(
        ValueError, 'Only 1d action is supported by this sampler.*'):
      cem_rejection_sampler_continuous_and_one_hot.GaussianActionsSampler(
          action_spec=action_spec, sample_validator={
              'continuous': self._sample_validator,
              'discrete': None},
          sub_actions_fields=[['discrete'], ['continuous']])

if __name__ == '__main__':
  tf.test.main()
