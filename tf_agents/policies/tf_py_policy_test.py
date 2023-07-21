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

"""Test for tf_agents.policies.tf_py_policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing.absltest import mock
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import tf_py_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils


class TFPyPolicyTest(test_utils.TestCase):

  def testRandomPyPolicyGeneratesActionTensors(self):
    array_action_spec = array_spec.BoundedArraySpec((7,), np.int32, -10, 10)
    observation = tf.ones([3], tf.float32)
    time_step = ts.restart(observation)

    observation_spec = tensor_spec.TensorSpec.from_tensor(observation)
    time_step_spec = ts.time_step_spec(observation_spec)

    tf_py_random_policy = tf_py_policy.TFPyPolicy(
        random_py_policy.RandomPyPolicy(time_step_spec=time_step_spec,
                                        action_spec=array_action_spec))

    batched_time_step = nest_utils.batch_nested_tensors(time_step)
    action_step = tf_py_random_policy.action(time_step=batched_time_step)
    action, new_policy_state = self.evaluate(
        [action_step.action, action_step.state])

    self.assertEqual((1,) + array_action_spec.shape, action.shape)
    self.assertTrue(np.all(action >= array_action_spec.minimum))
    self.assertTrue(np.all(action <= array_action_spec.maximum))
    self.assertEqual(new_policy_state, ())

  def testAction(self):
    py_observation_spec = array_spec.BoundedArraySpec((3,), np.int32, 1, 1)
    py_time_step_spec = ts.time_step_spec(py_observation_spec)
    py_action_spec = array_spec.BoundedArraySpec((7,), np.int32, 1, 1)
    py_policy_state_spec = array_spec.BoundedArraySpec((5,), np.int32, 0, 1)
    py_policy_info_spec = array_spec.BoundedArraySpec((3,), np.int32, 0, 1)

    mock_py_policy = mock.create_autospec(py_policy.PyPolicy)
    mock_py_policy.time_step_spec = py_time_step_spec
    mock_py_policy.action_spec = py_action_spec
    mock_py_policy.policy_state_spec = py_policy_state_spec
    mock_py_policy.info_spec = py_policy_info_spec

    expected_py_policy_state = np.ones(py_policy_state_spec.shape,
                                       py_policy_state_spec.dtype)
    expected_py_time_step = tf.nest.map_structure(
        lambda arr_spec: np.ones((1,) + arr_spec.shape, arr_spec.dtype),
        py_time_step_spec)
    expected_py_action = np.ones((1,) + py_action_spec.shape,
                                 py_action_spec.dtype)
    expected_new_py_policy_state = np.zeros(py_policy_state_spec.shape,
                                            py_policy_state_spec.dtype)
    expected_py_info = np.zeros(py_policy_info_spec.shape,
                                py_policy_info_spec.dtype)

    mock_py_policy.action.return_value = policy_step.PolicyStep(
        nest_utils.unbatch_nested_array(expected_py_action),
        expected_new_py_policy_state, expected_py_info)

    tf_mock_py_policy = tf_py_policy.TFPyPolicy(mock_py_policy)
    time_step = tf.nest.map_structure(
        lambda arr_spec: tf.ones((1,) + arr_spec.shape, arr_spec.dtype),
        py_time_step_spec)
    action_step = tf_mock_py_policy.action(
        time_step, tf.ones(py_policy_state_spec.shape, tf.int32))
    py_action_step = self.evaluate(action_step)

    self.assertEqual(1, mock_py_policy.action.call_count)
    np.testing.assert_equal(
        mock_py_policy.action.call_args[1]['time_step'],
        nest_utils.unbatch_nested_array(expected_py_time_step))
    np.testing.assert_equal(mock_py_policy.action.call_args[1]['policy_state'],
                            expected_py_policy_state)
    np.testing.assert_equal(py_action_step.action, expected_py_action)
    np.testing.assert_equal(py_action_step.state, expected_new_py_policy_state)
    np.testing.assert_equal(py_action_step.info, expected_py_info)

  def testZeroState(self):
    policy_state_length = 5
    batch_size = 3
    mock_py_policy = mock.create_autospec(py_policy.PyPolicy)
    observation_spec = array_spec.ArraySpec((3,), np.float32)
    mock_py_policy.time_step_spec = ts.time_step_spec(observation_spec)
    mock_py_policy.action_spec = array_spec.BoundedArraySpec(
        (7,), np.int32, 1, 1)
    py_policy_state_spec = array_spec.BoundedArraySpec((policy_state_length,),
                                                       np.int32, 1, 1)
    # Make the mock policy and reset return value.
    mock_py_policy.policy_state_spec = py_policy_state_spec
    mock_py_policy.info_spec = ()

    expected_py_policy_state = np.zeros(
        [batch_size] + list(py_policy_state_spec.shape),
        py_policy_state_spec.dtype)
    mock_py_policy.get_initial_state.return_value = expected_py_policy_state

    tf_mock_py_policy = tf_py_policy.TFPyPolicy(mock_py_policy)
    initial_state = tf_mock_py_policy.get_initial_state(batch_size=batch_size)
    initial_state_ = self.evaluate(initial_state)

    self.assertEqual(1, mock_py_policy.get_initial_state.call_count)
    np.testing.assert_equal(initial_state_, expected_py_policy_state)

  def testDistributionRaisesNotImplementedError(self):
    mock_tf_py_policy = tf_py_policy.TFPyPolicy(
        self._get_mock_py_policy())
    observation = tf.ones([5], tf.float32)
    time_step = ts.restart(observation)
    with self.assertRaises(NotImplementedError):
      mock_tf_py_policy.distribution(time_step=time_step)

  def testVariables(self):
    mock_tf_py_policy = tf_py_policy.TFPyPolicy(
        self._get_mock_py_policy())
    np.testing.assert_equal(mock_tf_py_policy.variables(), [])

  def testPyPolicyIsBatchedTrue(self):
    action_dims = 5
    observation_dims = 3
    batch_size = 2
    array_action_spec = array_spec.BoundedArraySpec((action_dims,), np.int32,
                                                    -10, 10)
    observation_spec = array_spec.ArraySpec((observation_dims,), np.float32)
    array_time_step_spec = ts.time_step_spec(observation_spec)

    observation = tf.ones([batch_size, observation_dims], tf.float32)
    time_step = ts.restart(observation, batch_size=batch_size)

    tf_py_random_policy = tf_py_policy.TFPyPolicy(
        random_py_policy.RandomPyPolicy(time_step_spec=array_time_step_spec,
                                        action_spec=array_action_spec),
        py_policy_is_batched=True)

    action_step = tf_py_random_policy.action(time_step=time_step)
    action = self.evaluate(action_step.action)

    self.assertEqual(action.shape, (batch_size, action_dims))

  def _get_mock_py_policy(self):
    mock_py_policy = mock.create_autospec(py_policy.PyPolicy)
    observation_spec = tensor_spec.TensorSpec([5], dtype=tf.float32)
    mock_py_policy.time_step_spec = ts.time_step_spec(observation_spec)
    mock_py_policy.action_spec = tensor_spec.BoundedTensorSpec(
        [3], tf.float32, -1.0, 1.0)
    mock_py_policy.policy_state_spec = ()
    mock_py_policy.info_spec = ()
    return mock_py_policy

if __name__ == '__main__':
  tf.test.main()
