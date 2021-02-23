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

# Lint as: python3
"""Tests for tf_agents.policies.async_policy_saver."""

import os

from absl.testing.absltest import mock
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import q_network
from tf_agents.policies import async_policy_saver
from tf_agents.policies import policy_saver
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils


class AsyncPolicySaverTest(test_utils.TestCase):

  def testSave(self):
    saver = mock.create_autospec(policy_saver.PolicySaver, instance=True)
    async_saver = async_policy_saver.AsyncPolicySaver(saver)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    save_path = os.path.join(self.get_temp_dir(), 'policy')
    async_saver.save(save_path)
    async_saver.flush()

    saver.save.assert_called_once_with(save_path)
    # Have to close the saver to avoid hanging threads that will prevent OSS
    # tests from finishing.
    async_saver.close()

  def testCheckpointSave(self):
    saver = mock.create_autospec(policy_saver.PolicySaver, instance=True)
    async_saver = async_policy_saver.AsyncPolicySaver(saver)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    async_saver.save(path)
    async_saver.flush()
    checkpoint_path = os.path.join(self.get_temp_dir(), 'checkpoint')
    async_saver.save_checkpoint(checkpoint_path)
    async_saver.flush()

    saver.save_checkpoint.assert_called_once_with(checkpoint_path)
    # Have to close the saver to avoid hanging threads that will prevent OSS
    # tests from finishing.
    async_saver.close()

  def testBlockingSave(self):
    saver = mock.create_autospec(policy_saver.PolicySaver, instance=True)
    async_saver = async_policy_saver.AsyncPolicySaver(saver)
    path1 = os.path.join(self.get_temp_dir(), 'save_model')
    path2 = os.path.join(self.get_temp_dir(), 'save_model2')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    async_saver.save(path1)
    async_saver.save(path2, blocking=True)

    saver.save.assert_has_calls([mock.call(path1), mock.call(path2)])
    # Have to close the saver to avoid hanging threads that will prevent OSS
    # tests from finishing.
    async_saver.close()

  def testBlockingCheckpointSave(self):
    saver = mock.create_autospec(policy_saver.PolicySaver, instance=True)
    async_saver = async_policy_saver.AsyncPolicySaver(saver)
    path1 = os.path.join(self.get_temp_dir(), 'save_model')
    path2 = os.path.join(self.get_temp_dir(), 'save_model2')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    async_saver.save_checkpoint(path1)
    async_saver.save_checkpoint(path2, blocking=True)

    saver.save_checkpoint.assert_has_calls([mock.call(path1), mock.call(path2)])
    # Have to close the saver to avoid hanging threads that will prevent OSS
    # tests from finishing.
    async_saver.close()

  def testClose(self):
    saver = mock.create_autospec(policy_saver.PolicySaver, instance=True)
    async_saver = async_policy_saver.AsyncPolicySaver(saver)
    path = os.path.join(self.get_temp_dir(), 'save_model')

    self.evaluate(tf.compat.v1.global_variables_initializer())
    async_saver.save(path)
    self.assertTrue(async_saver._save_thread.is_alive())

    async_saver.close()
    saver.save.assert_called_once()

    self.assertFalse(async_saver._save_thread.is_alive())

    with self.assertRaises(ValueError):
      async_saver.save(path)

  def testRegisterFunction(self):
    if not common.has_eager_been_enabled():
      self.skipTest('Only supported in TF2.x. Step is required in TF1.x')

    time_step_spec = ts.TimeStep(
        step_type=tensor_spec.BoundedTensorSpec(
            dtype=tf.int32, shape=(), name='st', minimum=0, maximum=2),
        reward=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32, shape=(), name='reward', minimum=0.0,
            maximum=5.0),
        discount=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32,
            shape=(),
            name='discount',
            minimum=0.0,
            maximum=1.0),
        observation=tensor_spec.BoundedTensorSpec(
            dtype=tf.float32,
            shape=(4,),
            name='obs',
            minimum=-10.0,
            maximum=10.0))
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=10, name='act_0')

    network = q_network.QNetwork(
        input_tensor_spec=time_step_spec.observation,
        action_spec=action_spec)

    policy = q_policy.QPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        q_network=network)

    saver = policy_saver.PolicySaver(policy, batch_size=None)
    async_saver = async_policy_saver.AsyncPolicySaver(saver)
    async_saver.register_function('q_network', network,
                                  time_step_spec.observation)

    path = os.path.join(self.get_temp_dir(), 'save_model')
    async_saver.save(path)
    async_saver.flush()
    async_saver.close()
    self.assertFalse(async_saver._save_thread.is_alive())
    reloaded = tf.compat.v2.saved_model.load(path)

    sample_input = self.evaluate(
        tensor_spec.sample_spec_nest(
            time_step_spec.observation, outer_dims=(3,)))
    expected_output, _ = network(sample_input)
    reloaded_output, _ = reloaded.q_network(sample_input)

    self.assertAllClose(expected_output, reloaded_output)


if __name__ == '__main__':
  tf.test.main()
