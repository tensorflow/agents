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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils
from tensorflow.python.eager import context  # TF internal


@parameterized.named_parameters(
    ('tf.int32', tf.int32, context.graph_mode),
    ('tf.int32_eager', tf.int32, context.eager_mode),
    ('tf.int64', tf.int64, context.graph_mode),
    ('tf.int64_eager', tf.int64, context.eager_mode),
    ('tf.float32', tf.float32, context.graph_mode),
    ('tf.float32_eager', tf.float32, context.eager_mode),
    ('tf.float64', tf.float64, context.graph_mode),
    ('tf.float64_eager', tf.float64, context.eager_mode),
)
class RandomTFPolicyTest(test_utils.TestCase, parameterized.TestCase):

  def create_batch(self, single_time_step, batch_size):
    batch_time_step = nest_utils.stack_nested_tensors(
        [single_time_step] * batch_size)
    return batch_time_step

  def create_time_step(self):
    observation = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_step = ts.restart(observation)

    observation_spec = tensor_spec.TensorSpec(
        observation.shape.as_list(), tf.float32)
    time_step_spec = ts.time_step_spec(observation_spec)

    return time_step_spec, time_step

  def testGeneratesBoundedActions(self, dtype, run_mode):
    with run_mode():
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

  def testGeneratesUnBoundedActions(self, dtype, run_mode):
    with run_mode():
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
      # TODO(kbanoop) assertWithinBounds to test_utils.
      self.assertTrue(np.all(action_[0] >= bounded.minimum))
      self.assertTrue(np.all(action_[0] <= bounded.maximum))
      self.assertTrue(np.all(action_[1] >= bounded.minimum))
      self.assertTrue(np.all(action_[1] <= bounded.maximum))

  def testGeneratesBatchedActionsImplicitBatchSize(self, dtype, run_mode):
    with run_mode():
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

  def testEmitLogProbability(self, dtype, run_mode):
    with run_mode():
      action_spec = [
          tensor_spec.BoundedTensorSpec((2, 3), dtype, -10, 10),
          tensor_spec.BoundedTensorSpec((1, 2), dtype, -10, 10)
      ]
      time_step_spec, time_step = self.create_time_step()
      policy = random_tf_policy.RandomTFPolicy(
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          emit_log_probability=True)

      action_step = policy.action(time_step)
      tf.nest.assert_same_structure(action_spec, action_step.action)

      step = self.evaluate(action_step)
      action_ = step.action
      # For integer specs, boundaries are inclusive.
      p = 1./21 if dtype.is_integer else 1./20
      np.testing.assert_allclose(
          np.array(step.info.log_probability, dtype=np.float32),
          np.array(np.log([p, p], dtype=np.float32)),
          rtol=1e-5)
      self.assertTrue(np.all(action_[0] >= -10))
      self.assertTrue(np.all(action_[0] <= 10))
      self.assertTrue(np.all(action_[1] >= -10))
      self.assertTrue(np.all(action_[1] <= 10))


if __name__ == '__main__':
  tf.test.main()
