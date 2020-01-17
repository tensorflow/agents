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

"""Tests for tf_agents.policies.eager_tf_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import actor_network
from tf_agents.environments import random_py_environment
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils


class PyTFEagerPolicyTest(test_utils.TestCase):

  def testPyEnvCompatible(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    observation_spec = array_spec.ArraySpec([2], np.float32)
    action_spec = array_spec.BoundedArraySpec([1], np.float32, 2, 3)

    observation_tensor_spec = tensor_spec.from_spec(observation_spec)
    action_tensor_spec = tensor_spec.from_spec(action_spec)
    time_step_tensor_spec = ts.time_step_spec(observation_tensor_spec)

    actor_net = actor_network.ActorNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(10,),
    )

    tf_policy = actor_policy.ActorPolicy(
        time_step_tensor_spec, action_tensor_spec, actor_network=actor_net)

    py_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_policy)
    # Env will validate action types automaticall since we provided the
    # action_spec.
    env = random_py_environment.RandomPyEnvironment(observation_spec,
                                                    action_spec)

    time_step = env.reset()

    for _ in range(100):
      action_step = py_policy.action(time_step)
      time_step = env.step(action_step.action)

  def testRandomTFPolicyCompatibility(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    observation_spec = array_spec.ArraySpec([2], np.float32)
    action_spec = array_spec.BoundedArraySpec([1], np.float32, 2, 3)

    observation_tensor_spec = tensor_spec.from_spec(observation_spec)
    action_tensor_spec = tensor_spec.from_spec(action_spec)
    time_step_tensor_spec = ts.time_step_spec(observation_tensor_spec)

    tf_policy = random_tf_policy.RandomTFPolicy(time_step_tensor_spec,
                                                action_tensor_spec)

    py_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_policy)
    env = random_py_environment.RandomPyEnvironment(observation_spec,
                                                    action_spec)
    time_step = env.reset()

    for _ in range(100):
      action_step = py_policy.action(time_step)
      time_step = env.step(action_step.action)


class SavedModelPYTFEagerPolicyTest(test_utils.TestCase,
                                    parameterized.TestCase):

  def setUp(self):
    super(SavedModelPYTFEagerPolicyTest, self).setUp()
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    observation_spec = array_spec.ArraySpec([2], np.float32)
    self.action_spec = array_spec.BoundedArraySpec([1], np.float32, 2, 3)
    self.time_step_spec = ts.time_step_spec(observation_spec)

    observation_tensor_spec = tensor_spec.from_spec(observation_spec)
    action_tensor_spec = tensor_spec.from_spec(self.action_spec)
    time_step_tensor_spec = tensor_spec.from_spec(self.time_step_spec)

    actor_net = actor_network.ActorNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(10,),
    )

    self.tf_policy = actor_policy.ActorPolicy(
        time_step_tensor_spec, action_tensor_spec, actor_network=actor_net)

  def testSavedModel(self):

    path = os.path.join(self.get_temp_dir(), 'saved_policy')
    saver = policy_saver.PolicySaver(self.tf_policy)
    saver.save(path)

    eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        path, self.time_step_spec, self.action_spec)
    rng = np.random.RandomState()
    sample_time_step = array_spec.sample_spec_nest(self.time_step_spec, rng)
    batched_sample_time_step = nest_utils.batch_nested_array(sample_time_step)

    original_action = self.tf_policy.action(batched_sample_time_step)
    unbatched_original_action = nest_utils.unbatch_nested_tensors(
        original_action)
    original_action_np = tf.nest.map_structure(lambda t: t.numpy(),
                                               unbatched_original_action)
    saved_policy_action = eager_py_policy.action(sample_time_step)

    tf.nest.assert_same_structure(saved_policy_action.action, self.action_spec)

    np.testing.assert_array_almost_equal(original_action_np.action,
                                         saved_policy_action.action)

  @parameterized.parameters(None, 0, 100, 200000)
  def testGetTrainStep(self, train_step):
    path = os.path.join(self.get_temp_dir(), 'saved_policy')
    if train_step is None:
      # Use the default argument, which should set the train step to be -1.
      saver = policy_saver.PolicySaver(self.tf_policy)
      expected_train_step = -1
    else:
      saver = policy_saver.PolicySaver(
          self.tf_policy, train_step=tf.constant(train_step))
      expected_train_step = train_step
    saver.save(path)

    eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        path, self.time_step_spec, self.action_spec)

    self.assertEqual(expected_train_step, eager_py_policy.get_train_step())


if __name__ == '__main__':
  tf.test.main()
