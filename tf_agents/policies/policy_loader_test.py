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

# Lint as: python3
"""Tests for model_service_quick_saver."""
import os

import numpy as np

import tensorflow.compat.v2 as tf
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_loader
from tf_agents.policies import policy_saver
from tf_agents.policies import q_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils


class AddNet(network.Network):
  """Small model used for tests."""

  def __init__(self):
    super(AddNet,
          self).__init__(tensor_spec.TensorSpec((), tf.float32), (), 'add_net')
    self.var = tf.Variable(0.0, dtype=tf.float32)

  def call(self, observation, step_type=None, network_state=(), training=False):
    del step_type, network_state, training
    return observation + self.var, ()


class PolicyLoaderTest(test_utils.TestCase):
  """Tests for policy loader."""

  def setUp(self):
    super(PolicyLoaderTest, self).setUp()
    self.root_dir = self.get_temp_dir()
    tf_observation_spec = tensor_spec.TensorSpec((), np.float32)
    tf_time_step_spec = ts.time_step_spec(tf_observation_spec)
    tf_action_spec = tensor_spec.BoundedTensorSpec((), np.float32, 0.0, 3.0)
    self.net = AddNet()
    self.policy = greedy_policy.GreedyPolicy(
        q_policy.QPolicy(tf_time_step_spec, tf_action_spec, self.net))
    self.train_step = common.create_variable('train_step', initial_value=0)
    self.saver = policy_saver.PolicySaver(
        self.policy, train_step=self.train_step)

  def _createModelsOnDisk(self):
    saved_model_dir = os.path.join(self.root_dir, 'policy')
    ckpt_dir = os.path.join(self.root_dir, 'checkpoint')
    self.train_step.assign(0)
    self.net.var.assign(0)
    saved_at_0_path = os.path.join(saved_model_dir, '000')
    self.saver.save(saved_at_0_path)
    self.train_step.assign(1)
    self.net.var.assign(10)
    ckpt_at_1_path = os.path.join(ckpt_dir, '001')
    self.saver.save_checkpoint(ckpt_at_1_path)
    return saved_at_0_path, ckpt_at_1_path

  def testLoad(self):
    saved_path, ckpt_at_path_1 = self._createModelsOnDisk()
    policy_at_0 = policy_loader.load(saved_path)
    self.assertEqual(0, policy_at_0.get_train_step())
    self.assertEqual(0, policy_at_0.variables()[0].numpy())
    policy_at_1 = policy_loader.load(saved_path, ckpt_at_path_1)
    self.assertEqual(1, policy_at_1.get_train_step())
    self.assertEqual(10, policy_at_1.variables()[0].numpy())

  def testMaterialize(self):
    saved_path, ckpt_at_path_1 = self._createModelsOnDisk()
    materialized_path = os.path.join(self.root_dir, 'material/001')
    policy_loader.materialize_saved_model(saved_path, ckpt_at_path_1,
                                          materialized_path)
    policy_at_1 = policy_loader.load(materialized_path)
    self.assertEqual(1, policy_at_1.get_train_step())
    self.assertEqual(10, policy_at_1.variables()[0].numpy())


if __name__ == '__main__':
  tf.test.main()
