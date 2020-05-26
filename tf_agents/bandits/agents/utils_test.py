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

"""Tests for tf_agents.bandits.agents.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.bandits.agents import utils
from tf_agents.bandits.policies import policy_utilities
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.trajectories import trajectory

tfd = tfp.distributions
tf.compat.v1.enable_v2_behavior()


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10',
          'batch_size': 1,
          'context_dim': 10,
      }, {
          'testcase_name': '_batch4_contextdim5',
          'batch_size': 4,
          'context_dim': 5,
      })


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def testNumActionsFromTensorSpecGoodSpec(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=15)
    num_actions = utils.get_num_actions_from_tensor_spec(action_spec)
    self.assertEqual(num_actions, 16)

  def testNumActionsFromTensorSpecWrongRank(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(2, 3), minimum=0, maximum=15)

    with self.assertRaisesRegexp(ValueError, r'Action spec must be a scalar'):
      utils.get_num_actions_from_tensor_spec(action_spec)

  @test_cases()
  def testBUpdate(self, batch_size, context_dim):
    b_array = np.array(range(context_dim))
    r_array = np.array(range(batch_size)).reshape((batch_size, 1))
    x_array = np.array(range(batch_size * context_dim)).reshape(
        (batch_size, context_dim))
    rx = r_array * x_array
    expected_b_updated_array = b_array + np.sum(rx, axis=0)

    b = tf.constant(b_array, dtype=tf.float32, shape=[context_dim])
    r = tf.constant(r_array, dtype=tf.float32, shape=[batch_size])
    x = tf.constant(x_array, dtype=tf.float32, shape=[batch_size, context_dim])
    b_update = utils.sum_reward_weighted_observations(r, x)
    self.assertAllClose(expected_b_updated_array, self.evaluate(b + b_update))

  @test_cases()
  def testBUpdateEmptyObservations(self, batch_size, context_dim):
    r = tf.constant([], dtype=tf.float32, shape=[0, 1])
    x = tf.constant([], dtype=tf.float32, shape=[0, context_dim])
    b_update = utils.sum_reward_weighted_observations(r, x)
    expected_b_update_array = np.zeros([context_dim], dtype=np.float32)
    self.assertAllClose(expected_b_update_array, self.evaluate(b_update))

  def testLaplacian1D(self):
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=4)
    num_actions = utils.get_num_actions_from_tensor_spec(action_spec)
    laplacian_matrix = tf.convert_to_tensor(
        utils.build_laplacian_over_ordinal_integer_actions(action_spec),
        dtype=tf.float32)
    res = tf.matmul(
        laplacian_matrix, tf.ones([num_actions, 1], dtype=tf.float32))
    # The vector of ones is in the null space of the Laplacian matrix.
    self.assertAllClose(0.0, self.evaluate(tf.norm(res)))

    # The row sum is zero.
    row_sum = tf.reduce_sum(laplacian_matrix, 1)
    self.assertAllClose(0.0, self.evaluate(tf.norm(row_sum)))

    # The column sum is zero.
    column_sum = tf.reduce_sum(laplacian_matrix, 0)
    self.assertAllClose(0.0, self.evaluate(tf.norm(column_sum)))

    # The diagonal elements are 2.0.
    self.assertAllClose(2.0, laplacian_matrix[1, 1])

    laplacian_matrix_expected = np.array(
        [[1.0, -1.0, 0.0, 0.0, 0.0],
         [-1.0, 2.0, -1.0, 0.0, 0.0],
         [0.0, -1.0, 2.0, -1.0, 0.0],
         [0.0, 0.0, -1.0, 2.0, -1.0],
         [0.0, 0.0, 0.0, -1.0, 1.0]])
    self.assertAllClose(laplacian_matrix_expected,
                        self.evaluate(laplacian_matrix))

  def testComputePairwiseDistances(self):
    input_vects = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pdist_matrix = np.array(
        [[0.0, 27.0, 108.0,],
         [27.0, 0.0, 27.0],
         [108.0, 27.0, 0.0]])
    tf_dist_matrix = utils.compute_pairwise_distances(
        tf.constant(input_vects, dtype=tf.float32))
    self.assertAllClose(pdist_matrix, self.evaluate(tf_dist_matrix))

  def testBuildLaplacianNearestNeighborGraph(self):
    input_vects = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],
                            [10, 11, 12], [13, 14, 15]])
    num_actions = input_vects.shape[0]
    laplacian_matrix = utils.build_laplacian_nearest_neighbor_graph(
        tf.constant(input_vects, dtype=tf.float32), k=2)

    # The vector of ones is in the null space of the Laplacian matrix.
    res = tf.matmul(
        laplacian_matrix, tf.ones([num_actions, 1], dtype=tf.float32))
    self.assertAllClose(0.0, self.evaluate(tf.norm(res)))

    # The row sum is zero.
    row_sum = tf.reduce_sum(laplacian_matrix, 1)
    self.assertAllClose(0.0, self.evaluate(tf.norm(row_sum)))

    # The column sum is zero.
    column_sum = tf.reduce_sum(laplacian_matrix, 0)
    self.assertAllClose(0.0, self.evaluate(tf.norm(column_sum)))

    self.assertAllClose(2.0, laplacian_matrix[0, 0])
    self.assertAllClose(4.0, laplacian_matrix[2, 2])

  def testProcessExperienceGlobalFeatures(self):
    observation_spec = {
        'f1': tf.TensorSpec(shape=(5,), dtype=tf.string),
        'f2': tf.TensorSpec(shape=(5, 2), dtype=tf.int32)
    }
    time_step_spec = time_step.time_step_spec(observation_spec)
    training_data_spec = trajectory.Trajectory(
        step_type=time_step_spec.step_type,
        observation=time_step_spec.observation,
        action=tensor_spec.BoundedTensorSpec(
            shape=(), minimum=0, maximum=4, dtype=tf.int32),
        policy_info=(),
        next_step_type=time_step_spec.step_type,
        reward=tensor_spec.BoundedTensorSpec(
            shape=(), minimum=0, maximum=2, dtype=tf.float32),
        discount=time_step_spec.discount)
    experience = tensor_spec.sample_spec_nest(
        training_data_spec, outer_dims=(7, 2))
    observation, action, reward = utils.process_experience_for_neural_agents(
        experience, None, False, training_data_spec)
    self.assertAllEqual(
        observation['f1'][0], experience.observation['f1'][0, 0])
    self.assertEqual(action[0], experience.action[0, 0])
    self.assertEqual(reward[0], experience.reward[0, 0])

  def testProcessExperiencePerArmFeaturesWithMask(self):
    mask_spec = tensor_spec.BoundedTensorSpec(
        shape=(5,), minimum=0, maximum=1, dtype=tf.int32)
    observation_spec = ({
        'global': tf.TensorSpec(shape=(4,), dtype=tf.float32),
        'per_arm': {
            'f1': tf.TensorSpec(shape=(5,), dtype=tf.string),
            'f2': tf.TensorSpec(shape=(5, 2), dtype=tf.int32)
        }
    }, mask_spec)
    time_step_spec = time_step.time_step_spec(observation_spec)
    policy_info_spec = policy_utilities.PerArmPolicyInfo(
        chosen_arm_features={
            'f1': tf.TensorSpec(shape=(), dtype=tf.string),
            'f2': tf.TensorSpec(shape=(2,), dtype=tf.int32)
        })
    training_data_spec = trajectory.Trajectory(
        step_type=time_step_spec.step_type,
        observation=time_step_spec.observation,
        action=tensor_spec.BoundedTensorSpec(
            shape=(), minimum=0, maximum=4, dtype=tf.int32),
        policy_info=policy_info_spec,
        next_step_type=time_step_spec.step_type,
        reward=tensor_spec.BoundedTensorSpec(
            shape=(), minimum=0, maximum=2, dtype=tf.float32),
        discount=time_step_spec.discount)
    experience = tensor_spec.sample_spec_nest(
        training_data_spec, outer_dims=(7, 2))
    observation, action, reward = utils.process_experience_for_neural_agents(
        experience, lambda x: (x[0], x[1]), True, training_data_spec)
    self.assertEqual(observation['per_arm']['f1'][0],
                     experience.policy_info.chosen_arm_features['f1'][0, 0])
    self.assertAllEqual(action, tf.zeros(14, dtype=tf.int32))
    self.assertEqual(reward[0], experience.reward[0, 0])


if __name__ == '__main__':
  tf.test.main()
