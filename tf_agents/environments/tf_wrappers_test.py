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

"""Test for tf_agents.environments.tf_wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing.absltest import mock

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import random_tf_environment
from tf_agents.environments import tf_wrappers
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils


class TFEnvironmentBaseWrapperTest(tf.test.TestCase):

  def test_wrapped_method_propagation(self):
    mock_env = mock.MagicMock()
    env = tf_wrappers.TFEnvironmentBaseWrapper(mock_env)

    env.time_step_spec()
    self.assertEqual(1, mock_env.time_step_spec.call_count)

    env.action_spec()
    self.assertEqual(1, mock_env.action_spec.call_count)

    env.observation_spec()
    self.assertEqual(1, mock_env.observation_spec.call_count)

    env.batched()
    self.assertEqual(1, mock_env.batched.call_count)

    env.batch_size()
    self.assertEqual(1, mock_env.batch_size.call_count)

    env.current_time_step()
    self.assertEqual(1, mock_env.current_time_step.call_count)

    env.reset()
    self.assertEqual(1, mock_env.reset.call_count)

    env.step(0)
    self.assertEqual(1, mock_env.step.call_count)
    mock_env.step.assert_called_with(0)

    env.render()
    self.assertEqual(1, mock_env.render.call_count)


def _build_test_env(obs_spec=None, action_spec=None, batch_size=2):
  if obs_spec is None:
    obs_spec = tensor_spec.BoundedTensorSpec((2, 3), tf.int32, -10, 10)
  if action_spec is None:
    action_spec = tensor_spec.BoundedTensorSpec((1,), tf.int32, 0, 4)
  time_step_spec = ts.time_step_spec(obs_spec)
  return random_tf_environment.RandomTFEnvironment(
      time_step_spec, action_spec, batch_size=batch_size)


class OneHotActionWrapperTest(tf.test.TestCase):

  def test_action_spec(self):
    action_spec = tensor_spec.BoundedTensorSpec((1,), tf.int32, 0, 4)
    env = _build_test_env(action_spec=action_spec)
    wrapper = tf_wrappers.OneHotActionWrapper(env)
    expected_action_spec = tensor_spec.BoundedTensorSpec(
        shape=(1, 5),
        dtype=tf.int32,
        minimum=0,
        maximum=1,
        name='one_hot_action_spec')
    self.assertEqual(expected_action_spec, wrapper.action_spec())

  def test_action_spec_nested(self):
    action_spec = (tensor_spec.BoundedTensorSpec((1,), tf.int32, 0, 4),
                   tensor_spec.TensorSpec((2, 2), tf.float32))
    env = _build_test_env(action_spec=action_spec)
    wrapper = tf_wrappers.OneHotActionWrapper(env)
    expected_action_spec = (
        tensor_spec.BoundedTensorSpec(
            shape=(1, 5),
            dtype=tf.int32,
            minimum=0,
            maximum=1,
            name='one_hot_action_spec'),
        action_spec[1])
    self.assertEqual(expected_action_spec, wrapper.action_spec())

  def test_raises_invalid_action_spec(self):
    action_spec = tensor_spec.BoundedTensorSpec((1, 1), tf.int32, 0, 4)
    with self.assertRaisesRegexp(ValueError, 'at most one dimension'):
      tf_wrappers.OneHotActionWrapper(_build_test_env(action_spec=action_spec))

  def test_step(self):
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 4)
    env = _build_test_env(action_spec=action_spec)
    mock_env = mock.Mock(wraps=env)
    wrapper = tf_wrappers.OneHotActionWrapper(mock_env)
    wrapper.reset()

    wrapper.step(tf.constant([[0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0]], tf.int32))
    self.assertTrue(mock_env.step.called)
    self.assertAllEqual([1, 3], mock_env.step.call_args[0][0])


if __name__ == '__main__':
  test_utils.main()
