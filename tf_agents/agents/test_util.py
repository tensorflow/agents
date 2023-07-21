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

"""Test util for agents.tf_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils


def test_loss_and_train_output(test: test_utils.TestCase,
                               expect_equal_loss_values: bool,
                               agent: tf_agent.TFAgent,
                               experience: types.NestedTensor,
                               weights: Optional[types.Tensor] = None,
                               **kwargs):
  """Tests that loss() and train() outputs are equivalent.

  Checks that the outputs have the same structures and shapes, and compares
  loss values based on `expect_equal_loss_values`.

  Args:
    test: An instance of `test_utils.TestCase`.
    expect_equal_loss_values: Whether to expect `LossInfo.loss` to have the same
      values for loss() and train().
    agent: An instance of `TFAgent`.
    experience: A batch of experience data in the form of a `Trajectory`.
    weights: (optional).  A `Tensor` containing weights to be used when
      calculating the total loss.
    **kwargs: Any additional data as args to `train` and `loss`.
  """
  loss_info_from_loss = agent.loss(
      experience=experience, weights=weights, **kwargs)
  loss_info_from_train = agent.train(
      experience=experience, weights=weights, **kwargs)
  if not tf.executing_eagerly():
    test.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info_from_loss = test.evaluate(loss_info_from_loss)
    loss_info_from_train = test.evaluate(loss_info_from_train)

  test.assertIsInstance(loss_info_from_train, tf_agent.LossInfo)
  test.assertEqual(type(loss_info_from_train), type(loss_info_from_loss))

  # Compare loss values.
  if expect_equal_loss_values:
    test.assertEqual(
        loss_info_from_train.loss,
        loss_info_from_loss.loss,
        msg='Expected equal loss values, but train() has output '
        '{loss_from_train} vs loss() output {loss_from_loss}.'.format(
            loss_from_train=loss_info_from_train.loss,
            loss_from_loss=loss_info_from_loss.loss))
  else:
    test.assertNotEqual(
        loss_info_from_train.loss,
        loss_info_from_loss.loss,
        msg='Expected train() and loss() output to have different loss values, '
        'but both are {loss}.'.format(loss=loss_info_from_train.loss))

  # Check that both `LossInfo` outputs have matching dtypes and shapes.
  nest_utils.assert_tensors_matching_dtypes_and_shapes(
      loss_info_from_train, loss_info_from_loss, test,
      '`LossInfo` from train()', '`LossInfo` from loss()')
