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

"""Test for tf_agents.utils.random_tf_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils


@parameterized.named_parameters(
    ('_int32', tf.int32),
    ('_int64', tf.int64),
    ('_float32', tf.float32),
    ('_float64', tf.float64),
)
class RandomTFPolicyTest(test_utils.TestCase, parameterized.TestCase):

  def create_batch(self, single_time_step, batch_size):
    batch_time_step = nest_utils.stack_nested_tensors([single_time_step] *
                                                      batch_size)
    return batch_time_step

  def create_time_step(self):
    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observation)

    observation_spec = tensor_spec.TensorSpec(observation.shape.as_list(),
                                              tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)

    return time_step_spec, time_step

  def testGeneratesBoundedActions(self, dtype):
    action_spec = [
        tensor_spec.BoundedTensorSpec((2, 3), dtype, -10, 10),
        tensor_spec.BoundedTensorSpec((1, 2), dtype, -10, 10)
    ]
    time_step_spec, time_step = self.create_time_step()
    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    action_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(action_[0] >= -10))
    self.assertTrue(np.all(action_[0] <= 10))
    self.assertTrue(np.all(action_[1] >= -10))
    self.assertTrue(np.all(action_[1] <= 10))

  def testGeneratesUnBoundedActions(self, dtype):
    action_spec = [
        tensor_spec.TensorSpec((2, 3), dtype),
        tensor_spec.TensorSpec((1, 2), dtype)
    ]
    bounded = tensor_spec.BoundedTensorSpec.from_spec(action_spec[0])
    time_step_spec, time_step = self.create_time_step()
    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    action_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(action_[0] >= bounded.minimum))
    self.assertTrue(np.all(action_[0] <= bounded.maximum))
    self.assertTrue(np.all(action_[1] >= bounded.minimum))
    self.assertTrue(np.all(action_[1] <= bounded.maximum))

  def testGeneratesBatchedActionsImplicitBatchSize(self, dtype):
    action_spec = [
        tensor_spec.BoundedTensorSpec((2, 3), dtype, -10, 10),
        tensor_spec.BoundedTensorSpec((1, 2), dtype, -10, 10)
    ]
    time_step_spec, time_step = self.create_time_step()
    time_step = self.create_batch(time_step, 2)
    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    action_ = self.evaluate(action_step.action)
    self.assertTrue(np.all(action_[0] >= -10))
    self.assertTrue(np.all(action_[0] <= 10))
    self.assertTrue(np.all(action_[1] >= -10))
    self.assertTrue(np.all(action_[1] <= 10))

    self.assertEqual((2, 2, 3), action_[0].shape)
    self.assertEqual((2, 1, 2), action_[1].shape)

  def testEmitLogProbability(self, dtype):
    action_spec = [
        tensor_spec.BoundedTensorSpec((2, 3), dtype, -10, 10),
        tensor_spec.BoundedTensorSpec((1, 2), dtype, -10, 10)
    ]
    time_step_spec, time_step = self.create_time_step()
    batch_size = 3
    time_step = self.create_batch(time_step, batch_size)

    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        emit_log_probability=True)

    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    step = self.evaluate(action_step)
    action_ = step.action
    # For integer specs, boundaries are inclusive.
    p = 1. / 21 if dtype.is_integer else 1. / 20
    np.testing.assert_allclose(
        np.array(step.info.log_probability, dtype=np.float32),
        np.array(
            np.log([[math.pow(p, 6) for _ in range(3)],
                    [math.pow(p, 2) for _ in range(3)]]),
            dtype=np.float32),
        rtol=1e-5)
    self.assertTrue(np.all(action_[0] >= -10))
    self.assertTrue(np.all(action_[0] <= 10))
    self.assertTrue(np.all(action_[1] >= -10))
    self.assertTrue(np.all(action_[1] <= 10))

  def testMasking(self, dtype):
    if not dtype.is_integer:
      self.skipTest('testMasking only applies to integer dtypes')

    batch_size = 1000

    action_spec = tensor_spec.BoundedTensorSpec((), dtype, -5, 5)
    time_step_spec, time_step = self.create_time_step()
    time_step = self.create_batch(time_step, batch_size)

    # We create a fixed mask here for testing purposes. Normally the mask would
    # be part of the observation.
    mask = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    np_mask = np.array(mask)
    tf_mask = tf.constant([mask for _ in range(batch_size)])

    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        emit_log_probability=True,
        observation_and_action_constraint_splitter=lambda obs: (obs, tf_mask))

    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)

    # Sample from the policy 1000 times, and ensure that actions considered
    # invalid according to the mask are never chosen.
    step = self.evaluate(action_step)
    action_ = step.action
    self.assertTrue(np.all(action_ >= -5))
    self.assertTrue(np.all(action_ <= 5))
    self.assertAllEqual(np_mask[action_ - action_spec.minimum],
                        np.ones([batch_size]))

    # Ensure that all valid actions occur somewhere within the batch. Because we
    # sample 1000 times, the chance of this failing for any particular action is
    # (2/3)^1000, roughly 1e-176.
    for index in range(action_spec.minimum, action_spec.maximum + 1):
      if np_mask[index - action_spec.minimum]:
        self.assertIn(index, action_)

    # With only three valid actions, all of the probabilities should be 1/3.
    self.assertAllClose(step.info.log_probability,
                        tf.constant(np.log(1. / 3), shape=[batch_size]))

  def testInfoSpec(self, dtype):
    action_spec = [
        tensor_spec.BoundedTensorSpec((2, 3), dtype, -10, 10),
        tensor_spec.BoundedTensorSpec((1, 2), dtype, -10, 10)
    ]
    info_spec = [
        tensor_spec.TensorSpec([1], dtype=tf.float32, name='loc'),
        tensor_spec.TensorSpec([1], dtype=tf.float32, name='scale')
    ]
    time_step_spec, time_step = self.create_time_step()
    policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        info_spec=info_spec)

    # Test without batch
    action_step = policy.action(time_step)
    tf.nest.assert_same_structure(action_spec, action_step.action)
    self.assertEqual((2, 3,), action_step.action[0].shape)
    self.assertEqual((1, 2,), action_step.action[1].shape)
    tf.nest.assert_same_structure(info_spec, action_step.info)
    self.assertEqual((1,), action_step.info[0].shape)
    self.assertEqual((1,), action_step.info[1].shape)

    # Test with batch, we should see the additional outer batch dim for both
    # `action` and `info`.
    batch_size = 2
    batched_time_step = self.create_batch(time_step, batch_size)
    batched_action_step = policy.action(batched_time_step)
    tf.nest.assert_same_structure(action_spec, batched_action_step.action)
    self.assertEqual((batch_size, 2, 3,), batched_action_step.action[0].shape)
    self.assertEqual((batch_size, 1, 2,), batched_action_step.action[1].shape)
    tf.nest.assert_same_structure(info_spec, batched_action_step.info)
    self.assertEqual((batch_size, 1,), batched_action_step.info[0].shape)
    self.assertEqual((batch_size, 1,), batched_action_step.info[1].shape)


if __name__ == '__main__':
  tf.test.main()
