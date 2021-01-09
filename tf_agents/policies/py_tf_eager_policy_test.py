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

"""Tests for tf_agents.policies.eager_tf_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.environments import batched_py_environment
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

  def setUp(self):
    super(PyTFEagerPolicyTest, self).setUp()
    self._observation_spec = array_spec.ArraySpec([2], np.float32)
    self._action_spec = array_spec.BoundedArraySpec([1], np.float32, 2, 3)
    self._observation_tensor_spec = tensor_spec.from_spec(
        self._observation_spec)
    self._action_tensor_spec = tensor_spec.from_spec(self._action_spec)
    self._time_step_tensor_spec = ts.time_step_spec(
        self._observation_tensor_spec)
    info_spec = {
        'a': array_spec.BoundedArraySpec([1], np.float32, 0, 1),
        'b': array_spec.BoundedArraySpec([1], np.float32, 100, 101)
    }
    self._info_tensor_spec = tensor_spec.from_spec(info_spec)
    # Env will validate action types automaticall since we provided the
    # action_spec.
    self._env = random_py_environment.RandomPyEnvironment(
        self._observation_spec, self._action_spec)

  def testPyEnvCompatible(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    actor_net = actor_network.ActorNetwork(
        self._observation_tensor_spec,
        self._action_tensor_spec,
        fc_layer_params=(10,),
    )

    tf_policy = actor_policy.ActorPolicy(
        self._time_step_tensor_spec,
        self._action_tensor_spec,
        actor_network=actor_net)

    py_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_policy)
    time_step = self._env.reset()

    for _ in range(100):
      action_step = py_policy.action(time_step)
      time_step = self._env.step(action_step.action)

  def testBatchedPyEnvCompatible(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    actor_net = actor_network.ActorNetwork(
        self._observation_tensor_spec,
        self._action_tensor_spec,
        fc_layer_params=(10,),
    )

    tf_policy = actor_policy.ActorPolicy(
        self._time_step_tensor_spec,
        self._action_tensor_spec,
        actor_network=actor_net)

    py_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_policy, batch_time_steps=False)

    env_ctr = lambda: random_py_environment.RandomPyEnvironment(  # pylint: disable=g-long-lambda
        self._observation_spec, self._action_spec)

    env = batched_py_environment.BatchedPyEnvironment(
        [env_ctr() for _ in range(3)])
    time_step = env.reset()

    for _ in range(20):
      action_step = py_policy.action(time_step)
      time_step = env.step(action_step.action)

  def testActionWithSeed(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    tf_policy = random_tf_policy.RandomTFPolicy(
        self._time_step_tensor_spec,
        self._action_tensor_spec,
        info_spec=self._info_tensor_spec)

    py_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_policy)
    time_step = self._env.reset()
    tf.random.set_seed(100)
    action_step_1 = py_policy.action(time_step, seed=100)
    time_step = self._env.reset()
    tf.random.set_seed(100)
    action_step_2 = py_policy.action(time_step, seed=100)
    time_step = self._env.reset()
    tf.random.set_seed(200)
    action_step_3 = py_policy.action(time_step, seed=200)
    self.assertEqual(action_step_1.action[0], action_step_2.action[0])
    self.assertNotEqual(action_step_1.action[0], action_step_3.action[0])

  def testRandomTFPolicyCompatibility(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in eager.')

    tf_policy = random_tf_policy.RandomTFPolicy(
        self._time_step_tensor_spec,
        self._action_tensor_spec,
        info_spec=self._info_tensor_spec)

    py_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_policy)
    time_step = self._env.reset()

    def _check_action_step(action_step):
      self.assertIsInstance(action_step.action, np.ndarray)
      self.assertEqual(action_step.action.shape, (1,))
      self.assertBetween(action_step.action[0], 2.0, 3.0)

      self.assertIsInstance(action_step.info['a'], np.ndarray)
      self.assertEqual(action_step.info['a'].shape, (1,))
      self.assertBetween(action_step.info['a'][0], 0.0, 1.0)

      self.assertIsInstance(action_step.info['b'], np.ndarray)
      self.assertEqual(action_step.info['b'].shape, (1,))
      self.assertBetween(action_step.info['b'][0], 100.0, 101.0)

    for _ in range(100):
      action_step = py_policy.action(time_step)
      _check_action_step(action_step)
      time_step = self._env.step(action_step.action)


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

  def testSavedModelLoadingSpecs(self):
    path = os.path.join(self.get_temp_dir(), 'saved_policy')
    saver = policy_saver.PolicySaver(self.tf_policy)
    saver.save(path)

    eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        path, load_specs_from_pbtxt=True)

    # Bounded specs get converted to regular specs when saved into a proto.
    def assert_specs_mostly_equal(loaded_spec, expected_spec):
      self.assertEqual(loaded_spec.shape, expected_spec.shape)
      self.assertEqual(loaded_spec.dtype, expected_spec.dtype)

    tf.nest.map_structure(assert_specs_mostly_equal,
                          eager_py_policy.time_step_spec, self.time_step_spec)
    tf.nest.map_structure(assert_specs_mostly_equal,
                          eager_py_policy.action_spec, self.action_spec)

  @parameterized.parameters(None, 0, 100, 200000)
  def testGetTrainStep(self, train_step):
    path = os.path.join(self.get_temp_dir(), 'saved_policy')
    if train_step is None:
      # Use the default argument, which should set the train step to be -1.
      saver = policy_saver.PolicySaver(self.tf_policy)
      expected_train_step = -1
    else:
      saver = policy_saver.PolicySaver(
          self.tf_policy,
          train_step=common.create_variable(
              'train_step', initial_value=train_step))
      expected_train_step = train_step
    saver.save(path)

    eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        path, self.time_step_spec, self.action_spec)

    self.assertEqual(expected_train_step, eager_py_policy.get_train_step())

  def testUpdateFromCheckpoint(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x.')

    path = os.path.join(self.get_temp_dir(), 'saved_policy')
    saver = policy_saver.PolicySaver(self.tf_policy)
    saver.save(path)
    self.evaluate(
        tf.nest.map_structure(lambda v: v.assign(v * 0 + -1),
                              self.tf_policy.variables()))
    checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoint')
    saver.save_checkpoint(checkpoint_path)

    eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        path, self.time_step_spec, self.action_spec)

    # Use evaluate to force a copy.
    saved_model_variables = self.evaluate(eager_py_policy.variables())

    eager_py_policy.update_from_checkpoint(checkpoint_path)

    assert_np_not_equal = lambda a, b: self.assertFalse(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_not_equal, saved_model_variables,
                          self.evaluate(eager_py_policy.variables()))

    assert_np_all_equal = lambda a, b: self.assertTrue(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_all_equal,
                          self.evaluate(self.tf_policy.variables()),
                          self.evaluate(eager_py_policy.variables()),
                          check_types=False)

  def testInferenceFromCheckpoint(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x.')

    path = os.path.join(self.get_temp_dir(), 'saved_policy')
    saver = policy_saver.PolicySaver(self.tf_policy)
    saver.save(path)

    rng = np.random.RandomState()
    sample_time_step = array_spec.sample_spec_nest(self.time_step_spec, rng)
    batched_sample_time_step = nest_utils.batch_nested_array(sample_time_step)

    self.evaluate(
        tf.nest.map_structure(lambda v: v.assign(v * 0 + -1),
                              self.tf_policy.variables()))
    checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoint')
    saver.save_checkpoint(checkpoint_path)

    eager_py_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        path, self.time_step_spec, self.action_spec)

    # Use evaluate to force a copy.
    saved_model_variables = self.evaluate(eager_py_policy.variables())

    eager_py_policy.update_from_checkpoint(checkpoint_path)

    assert_np_not_equal = lambda a, b: self.assertFalse(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_not_equal, saved_model_variables,
                          self.evaluate(eager_py_policy.variables()))

    assert_np_all_equal = lambda a, b: self.assertTrue(np.equal(a, b).all())
    tf.nest.map_structure(assert_np_all_equal,
                          self.evaluate(self.tf_policy.variables()),
                          self.evaluate(eager_py_policy.variables()),
                          check_types=False)

    # Can't check if the action is different as in some cases depending on
    # variable initialization it will be the same. Checking that they are at
    # least always the same.
    checkpoint_action = eager_py_policy.action(sample_time_step)

    current_policy_action = self.tf_policy.action(batched_sample_time_step)
    current_policy_action = self.evaluate(
        nest_utils.unbatch_nested_tensors(current_policy_action))
    tf.nest.map_structure(assert_np_all_equal, current_policy_action,
                          checkpoint_action)


if __name__ == '__main__':
  test_utils.main()
