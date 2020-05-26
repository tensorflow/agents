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

"""Common utility code and linear algebra functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


def sum_reward_weighted_observations(r, x):
  """Calculates an update used by some Bandit algorithms.

  Given an observation `x` and corresponding reward `r`, the weigthed
  observations vector (denoted `b` here) should be updated as `b = b + r * x`.
  This function calculates the sum of weighted rewards for batched
  observations `x`.

  Args:
    r: a `Tensor` of shape [`batch_size`]. This is the rewards of the batched
      observations.
    x: a `Tensor` of shape [`batch_size`, `context_dim`]. This is the matrix
      with the (batched) observations.

  Returns:
    The update that needs to be added to `b`. Has the same shape as `b`.
    If the observation matrix `x` is empty, a zero vector is returned.
  """
  batch_size = tf.shape(x)[0]

  return tf.reduce_sum(tf.reshape(r, [batch_size, 1]) * x, axis=0)


def get_num_actions_from_tensor_spec(action_spec):
  """Validates `action_spec` and returns number of actions.

  `action_spec` must specify a scalar int32 or int64 with minimum zero.

  Args:
    action_spec: a `TensorSpec`.

  Returns:
    The number of actions described by `action_spec`.

  Raises:
    ValueError: if `action_spec` is not an bounded scalar int32 or int64 spec
      with minimum 0.
  """
  if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
    raise ValueError('Action spec must be a `BoundedTensorSpec`; '
                     'got {}'.format(type(action_spec)))
  if action_spec.shape.rank != 0:
    raise ValueError('Action spec must be a scalar; '
                     'got shape{}'.format(action_spec.shape))
  if action_spec.dtype not in (tf.int32, tf.int64):
    raise ValueError('Action spec must be have dtype int32 or int64; '
                     'got {}'.format(action_spec.dtype))
  if action_spec.minimum != 0:
    raise ValueError('Action spec must have minimum 0; '
                     'got {}'.format(action_spec.minimum))
  return action_spec.maximum + 1


@gin.configurable
def build_laplacian_over_ordinal_integer_actions(action_spec):
  """Build the unnormalized Laplacian matrix over ordinal integer actions.

  Assuming integer actions, this functions builds the (unnormalized) Laplacian
  matrix of the graph implied over the action space. The graph vertices are the
  integers {0...action_spec.maximum - 1}. Two vertices are adjacent if they
  correspond to consecutive integer actions. The `action_spec` must specify
  a scalar int32 or int64 with minimum zero.

  Args:
    action_spec: a `BoundedTensorSpec`.

  Returns:
    The graph Laplacian matrix (float tensor) of size equal to the number of
    actions. The diagonal elements are equal to 2 and the off-diagonal elements
    are equal to -1.

  Raises:
    ValueError: if `action_spec` is not a bounded scalar int32 or int64 spec
      with minimum 0.
  """
  num_actions = get_num_actions_from_tensor_spec(action_spec)
  adjacency_matrix = np.zeros([num_actions, num_actions])
  for i in range(num_actions - 1):
    adjacency_matrix[i, i + 1] = 1.0
    adjacency_matrix[i + 1, i] = 1.0
  laplacian_matrix = np.diag(np.sum(adjacency_matrix,
                                    axis=0)) - adjacency_matrix
  return laplacian_matrix


def compute_pairwise_distances(input_vecs):
  """Compute the pairwise distances matrix.

  Given input embedding vectors, this utility computes the (squared) pairwise
  distances matrix.

  Args:
    input_vecs: a `Tensor`. Input embedding vectors (one per row).

  Returns:
    The (squared) pairwise distances matrix. A dense float `Tensor` of shape
    [`num_vectors`, `num_vectors`], where `num_vectors` is the number of input
    embedding vectors.
  """
  r = tf.reduce_sum(input_vecs * input_vecs, axis=1, keepdims=True)
  pdistance_matrix = (
      r - 2 * tf.matmul(input_vecs, input_vecs, transpose_b=True)
      + tf.transpose(r))
  return tf.cast(pdistance_matrix, dtype=tf.float32)


@gin.configurable
def build_laplacian_nearest_neighbor_graph(input_vecs, k=1):
  """Build the Laplacian matrix of a nearest neighbor graph.

  Given input embedding vectors, this utility returns the Laplacian matrix of
  the induced k-nearest-neighbor graph.

  Args:
    input_vecs: a `Tensor`. Input embedding vectors (one per row).  Shaped
      `[num_vectors, ...]`.
    k : an integer. Number of nearest neighbors to use.

  Returns:
    The graph Laplacian matrix. A dense float `Tensor` of shape
    `[num_vectors, num_vectors]`, where `num_vectors` is the number of input
    embedding vectors (`Tensor`).
  """
  num_actions = tf.shape(input_vecs)[0]
  pdistance_matrix = compute_pairwise_distances(input_vecs)
  sorted_indices = tf.argsort(values=pdistance_matrix)
  selected_indices = tf.reshape(sorted_indices[:, 1 : k + 1], [-1, 1])
  rng = tf.tile(
      tf.expand_dims(tf.range(num_actions), axis=-1), [1, k])
  rng = tf.reshape(rng, [-1, 1])
  full_indices = tf.concat([rng, selected_indices], axis=1)
  adjacency_matrix = tf.zeros([num_actions, num_actions], dtype=tf.float32)
  adjacency_matrix = tf.tensor_scatter_nd_update(
      tensor=adjacency_matrix,
      indices=full_indices,
      updates=tf.ones([k * num_actions], dtype=tf.float32))
  # Symmetrize it.
  adjacency_matrix = adjacency_matrix + tf.transpose(adjacency_matrix)
  adjacency_matrix = tf.minimum(
      adjacency_matrix, tf.ones_like(adjacency_matrix))
  degree_matrix = tf.linalg.tensor_diag(tf.reduce_sum(adjacency_matrix, axis=1))
  laplacian_matrix = degree_matrix - adjacency_matrix
  return laplacian_matrix


def process_experience_for_neural_agents(
    experience,
    observation_and_action_constraint_splitter,
    accepts_per_arm_features,
    training_data_spec):
  """Processes the experience and prepares it for the network of the agent.

  First the reward, the action, and the observation are flattened to have only
  one batch dimension. Then the action mask is removed if it is there. Finally,
  if the experience includes chosen action features in the policy info, it gets
  copied in place of the per-arm observation.

  Args:
    experience: The experience coming from the replay buffer.
    observation_and_action_constraint_splitter: If the agent accepts action
      masks, this function splits the mask from the observation.
    accepts_per_arm_features: Whether the agent accepts per-arm features.
    training_data_spec: The data spec describing what the agent expects.

  Returns:
    A tuple of (reward, action, observation) tensors to be consumed by the train
      function of the neural agent.
  """
  flattened_experience, _ = nest_utils.flatten_multi_batched_nested_tensors(
      experience, training_data_spec)

  observation = flattened_experience.observation
  action = flattened_experience.action
  reward = flattened_experience.reward

  if observation_and_action_constraint_splitter is not None:
    observation, _ = observation_and_action_constraint_splitter(
        observation)
  if accepts_per_arm_features:
    # The arm observation we train on needs to be copied from the respective
    # policy info field to the per arm observation field. Pretending there was
    # only one action, we fill the action field with zeros.
    chosen_arm_features = flattened_experience.policy_info.chosen_arm_features
    observation[bandit_spec_utils.PER_ARM_FEATURE_KEY] = tf.nest.map_structure(
        lambda t: tf.expand_dims(t, axis=1), chosen_arm_features)
    action = tf.zeros_like(action)

  return observation, action, reward
