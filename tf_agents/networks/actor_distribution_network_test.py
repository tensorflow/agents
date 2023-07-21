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

"""Tests for tf_agents.networks.actor_distribution_network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tensorflow.python.framework import test_util  # TF internal


class ActorDistributionNetworkTest(tf.test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testBuilds(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))

    action_spec = [
        tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3),
        tensor_spec.BoundedTensorSpec((3,), tf.int32, 0, 3)
    ]

    net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(4, 2, 2)],
        fc_layer_params=(5,))

    action_distributions, _ = net(time_step.observation, time_step.step_type,
                                  ())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([1, 2], action_distributions[0].mode().shape.as_list())
    self.assertEqual([1, 3], action_distributions[1].mode().shape.as_list())

  @test_util.run_in_graph_and_eager_modes()
  def testBuildsScalarContinuousActionSpace(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))

    action_spec = tensor_spec.BoundedTensorSpec((), tf.float32, 2, 3)

    net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec
    )

    action_distributions, _ = net(time_step.observation, time_step.step_type,
                                  ())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([1], action_distributions.mode().shape.as_list())

  @test_util.run_in_graph_and_eager_modes()
  def testHandlesExtraOuterDims(self):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(
        time_step_spec, outer_dims=(3, 2, 2))

    action_spec = [
        tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3),
        tensor_spec.BoundedTensorSpec((3,), tf.int32, 0, 3)
    ]

    net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(4, 2, 2)],
        fc_layer_params=(5,))

    action_distributions, _ = net(time_step.observation, time_step.step_type,
                                  ())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([3, 2, 2, 2],
                     action_distributions[0].mode().shape.as_list())
    self.assertEqual([3, 2, 2, 3],
                     action_distributions[1].mode().shape.as_list())

  @test_util.run_in_graph_and_eager_modes()
  def testHandlePreprocessingLayers(self):
    observation_spec = (tensor_spec.TensorSpec([1], tf.float32),
                        tensor_spec.TensorSpec([], tf.float32))
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(3,))

    action_spec = [
        tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3),
        tensor_spec.BoundedTensorSpec((3,), tf.int32, 0, 3)
    ]

    preprocessing_layers = (tf.keras.layers.Dense(4),
                            tf.keras.Sequential([
                                tf.keras.layers.Reshape((1,)),
                                tf.keras.layers.Dense(4)
                            ]))

    net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Add())

    action_distributions, _ = net(time_step.observation, time_step.step_type,
                                  ())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual([3, 2], action_distributions[0].mode().shape.as_list())
    self.assertEqual([3, 3], action_distributions[1].mode().shape.as_list())
    self.assertGreater(len(net.trainable_variables), 4)

  @parameterized.named_parameters(
      ('TrainingTrue', True,),
      ('TrainingFalse', False))
  def testDropoutFCLayersWithConv(self, training):
    observation_spec = tensor_spec.BoundedTensorSpec((8, 8, 3), tf.float32, 0,
                                                     1)
    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(time_step_spec, outer_dims=(1,))
    action_spec = tensor_spec.BoundedTensorSpec((2,), tf.float32, 2, 3)

    net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        conv_layer_params=[(4, 2, 2)],
        fc_layer_params=[5],
        dropout_layer_params=[0.5])

    modes = []
    num_modes = 10
    for _ in range(num_modes):
      action_distributions, _ = net(
          time_step.observation, time_step.step_type, (), training=training)
      modes.append(action_distributions.mode())

    self.evaluate(tf.compat.v1.global_variables_initializer())
    modes = self.evaluate(modes)

    # Verify that the modes from action_distributions are not all the same.
    any_modes_differ = False
    for i in range(num_modes):
      for j in range(i+1, num_modes):
        any_modes_differ = np.linalg.norm(modes[i] - modes[j]) > 1e-6
        if any_modes_differ:
          self.assertEqual(training, any_modes_differ)
          return

    self.assertEqual(training, any_modes_differ)

if __name__ == '__main__':
  tf.test.main()
