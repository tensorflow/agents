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

from tf_agents.policies import async_policy_saver
from tf_agents.policies import policy_saver
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


if __name__ == '__main__':
  tf.test.main()
