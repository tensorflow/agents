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

"""Tests for tf_agents.policies.samplers.qtopt_cem_actions_sampler_continuous_and_one_hot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tf_agents.policies.samplers import qtopt_cem_actions_sampler_continuous_and_one_hot
from tf_agents.specs import tensor_spec


_BATCH = 2
_NUM_SAMPLES = 5
_ACTION_SIZE_CONTINUOUS = 3
_ACTION_SIZE_DISCRETE = 3
_MEAN = [0., 0., 0.0]
_VAR = [0.09, 0.03, 0.05]


def dummy_sample_rejecter(samples, state_sample):
  del state_sample
  batch_size = samples['continuous1'].shape[0]
  num_samples = samples['continuous1'].shape[1]
  return tf.ones([batch_size, num_samples], dtype=tf.bool)


class ActionsSamplerTest(tf.test.TestCase):

  def testSampleBatch(self):
    action_spec = {
        'continuous1': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0),
        'continuous2': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0),
        'discrete': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_DISCRETE], tf.int32, 0, 1)}
    sampler = qtopt_cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
        action_spec=action_spec,
        sample_clippers=[[], [], []],
        sub_actions_fields=[['continuous1'], ['discrete'], ['continuous2']],
        sample_rejecters=[None, None, dummy_sample_rejecter])

    mean = tf.constant(_MEAN)
    var = tf.constant(_VAR)
    mean = {
        'continuous1': tf.broadcast_to(mean, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'continuous2': tf.broadcast_to(mean, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'discrete': tf.zeros([_BATCH, _ACTION_SIZE_DISCRETE])}
    var = {
        'continuous1': tf.broadcast_to(var, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'continuous2': tf.broadcast_to(var, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'discrete': tf.zeros([_BATCH, _ACTION_SIZE_DISCRETE])}

    actions = sampler.sample_batch_and_clip(_NUM_SAMPLES, mean, var)

    self.assertEqual((_BATCH, _NUM_SAMPLES, _ACTION_SIZE_CONTINUOUS),
                     actions['continuous1'].shape)
    self.assertEqual((_BATCH, _NUM_SAMPLES, _ACTION_SIZE_CONTINUOUS),
                     actions['continuous2'].shape)
    self.assertEqual((_BATCH, _NUM_SAMPLES, _ACTION_SIZE_DISCRETE),
                     actions['discrete'].shape)

    actions_ = self.evaluate(actions)
    flat_actions = tf.nest.flatten(actions_)
    flat_action_spec = tf.nest.flatten(action_spec)
    for i in range(len(flat_action_spec)):
      self.assertTrue((flat_actions[i] <= flat_action_spec[i].maximum).all())
      self.assertTrue((flat_actions[i] >= flat_action_spec[i].minimum).all())

    # make sure discrete part
    # make sure 0 for continuous part
    expected_actions_discrete = tf.broadcast_to(
        tf.constant([[1., 0., 0.],
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.],
                     [0., 0., 1.]]),
        tf.constant([_BATCH, _NUM_SAMPLES, _ACTION_SIZE_DISCRETE]))

    self.assertAllClose(expected_actions_discrete, actions_['discrete'])

    expected_actions_continuous1 = tf.zeros(
        [_BATCH, 2, _ACTION_SIZE_CONTINUOUS])
    self.assertAllClose(
        expected_actions_continuous1, actions_['continuous1'][:, 3:, :])

    self.assertAllGreaterEqual(
        actions_['continuous1'][:, 0:3, :], 0.0)
    self.assertAllLessEqual(
        actions_['continuous1'][:, 0:3, :], 1.0)

    expected_actions_continuous2 = tf.zeros(
        [_BATCH, 2, _ACTION_SIZE_CONTINUOUS])
    self.assertAllClose(
        expected_actions_continuous2, actions_['continuous2'][:, 1:3, :])
    self.assertAllGreaterEqual(
        actions_['continuous2'][:, 2:, :], 0.0)
    self.assertAllLessEqual(
        actions_['continuous2'][:, 2:, :], 1.0)

  def testInvalidActionSpec(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0)
    with self.assertRaisesRegex(
        ValueError, r'Only continuous action \+ 1 one_hot action is supported'
        ' by this sampler.*'):
      qtopt_cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
          action_spec=action_spec)

    action_spec = [
        tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_DISCRETE, _ACTION_SIZE_DISCRETE],
            tf.float32, 0.0, 1.0),
        tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0)]
    with self.assertRaisesRegex(
        ValueError, 'Only 1d action is supported by this sampler.*'):
      qtopt_cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
          action_spec=action_spec)

  def testRefitDistribution(self):

    def dummy_sample_clipper(actions, state):
      del state
      return actions

    action_spec = {
        'continuous': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_CONTINUOUS], tf.float32, 0.0, 1.0),
        'discrete': tensor_spec.BoundedTensorSpec(
            [_ACTION_SIZE_DISCRETE], tf.int32, 0, 1)}
    sampler = qtopt_cem_actions_sampler_continuous_and_one_hot.GaussianActionsSampler(
        action_spec=action_spec,
        sample_clippers=[[], [dummy_sample_clipper]],
        sub_actions_fields=[['discrete'], ['continuous']])

    mean = tf.constant(_MEAN)
    var = tf.constant(_VAR)
    mean = {
        'continuous': tf.broadcast_to(mean, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'discrete': tf.zeros([_BATCH, _ACTION_SIZE_DISCRETE])}
    var = {
        'continuous': tf.broadcast_to(var, [_BATCH, _ACTION_SIZE_CONTINUOUS]),
        'discrete': tf.zeros([_BATCH, _ACTION_SIZE_DISCRETE])}
    actions = sampler.sample_batch_and_clip(3, mean, var)

    # [B, M, A] = [2, 3, 3]
    actions['continuous'] = tf.constant(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
         [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])

    ind = tf.constant([[0, 0, 2], [1, 1, 2]])

    mean, var = sampler.refit_distribution_to(ind, actions)

    self.assertAllClose(self.evaluate(mean)['continuous'],
                        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    self.assertAllClose(self.evaluate(var)['continuous'],
                        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

if __name__ == '__main__':
  tf.test.main()
