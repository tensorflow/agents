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

"""Class implementation of the per-arm MovieLens Bandit environment."""
from __future__ import absolute_import
# Using Type Annotations.

import random
from typing import Optional, Text
import gin
import numpy as np

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import dataset_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


GLOBAL_KEY = bandit_spec_utils.GLOBAL_FEATURE_KEY
PER_ARM_KEY = bandit_spec_utils.PER_ARM_FEATURE_KEY


@gin.configurable
class MovieLensPerArmPyEnvironment(bandit_py_environment.BanditPyEnvironment):
  """Implements the per-arm version of the MovieLens Bandit environment.

  This environment implements the MovieLens 100K dataset, available at:
  https://www.kaggle.com/prajitdatta/movielens-100k-dataset

  This dataset contains 100K ratings from 943 users on 1682 items.
  This csv list of:
  user id | item id | rating | timestamp.
  This environment computes a low-rank matrix factorization (using SVD) of the
  data matrix `A`, such that: `A ~= U * Sigma * V^T`.

  The environment uses the rows of `U` as global (or user) features, and the
  rows of `V` as per-arm (or movie) features.

  The reward of recommending movie `v` to user `u` is `u * Sigma * v^T`.
  """

  def __init__(self,
               data_dir: Text,
               rank_k: int,
               batch_size: int = 1,
               num_actions: int = 50,
               csv_delimiter=',',
               name: Optional[Text] = 'movielens_per_arm'):
    """Initializes the Per-arm MovieLens Bandit environment.

    Args:
      data_dir: (string) Directory where the data lies (in text form).
      rank_k : (int) Which rank to use in the matrix factorization. This will
        also be the feature dimension of both the user and the movie features.
      batch_size: (int) Number of observations generated per call.
      num_actions: (int) How many movies to choose from per round.
      csv_delimiter: (string) The delimiter to use in loading the data csv file.
      name: (string) The name of this environment instance.
    """
    self._batch_size = batch_size
    self._context_dim = rank_k
    self._num_actions = num_actions

    # Compute the matrix factorization.
    self._data_matrix = dataset_utilities.load_movielens_data(
        data_dir, delimiter=csv_delimiter)
    self._num_users, self._num_movies = self._data_matrix.shape

    # Compute the SVD.
    u, s, vh = np.linalg.svd(self._data_matrix, full_matrices=False)

    # Keep only the largest singular values.
    self._u_hat = u[:, :rank_k].astype(np.float32)
    self._s_hat = s[:rank_k].astype(np.float32)
    self._v_hat = np.transpose(vh[:rank_k]).astype(np.float32)

    self._approx_ratings_matrix = np.matmul(self._u_hat * self._s_hat,
                                            np.transpose(self._v_hat))

    self._action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int32,
        minimum=0,
        maximum=num_actions - 1,
        name='action')
    observation_spec = {
        GLOBAL_KEY:
            array_spec.ArraySpec(shape=[rank_k], dtype=np.float32),
        PER_ARM_KEY:
            array_spec.ArraySpec(
                shape=[num_actions, rank_k], dtype=np.float32),
    }
    self._time_step_spec = ts.time_step_spec(observation_spec)

    self._current_user_indices = np.zeros(batch_size, dtype=np.int32)
    self._previous_user_indices = np.zeros(batch_size, dtype=np.int32)

    self._current_movie_indices = np.zeros([batch_size, num_actions],
                                           dtype=np.int32)
    self._previous_movie_indices = np.zeros([batch_size, num_actions],
                                            dtype=np.int32)

    self._observation = {
        GLOBAL_KEY:
            np.zeros([batch_size, rank_k]),
        PER_ARM_KEY:
            np.zeros([batch_size, num_actions, rank_k]),
    }

    super(MovieLensPerArmPyEnvironment, self).__init__(
        observation_spec, self._action_spec, name=name)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def batched(self):
    return True

  def _observe(self):
    sampled_user_indices = np.random.randint(
        self._num_users, size=self._batch_size)
    self._previous_user_indices = self._current_user_indices
    self._current_user_indices = sampled_user_indices

    sampled_movie_indices = np.array([
        random.sample(range(self._num_movies), self._num_actions)
        for _ in range(self._batch_size)
    ])
    movie_index_vector = sampled_movie_indices.reshape(-1)
    flat_movie_list = self._v_hat[movie_index_vector]
    current_movies = flat_movie_list.reshape(
        [self._batch_size, self._num_actions, self._context_dim])

    self._previous_movie_indices = self._current_movie_indices
    self._current_movie_indices = sampled_movie_indices

    batched_observations = {
        GLOBAL_KEY:
            self._u_hat[sampled_user_indices],
        PER_ARM_KEY:
            current_movies,
    }
    return batched_observations

  def _apply_action(self, action):
    chosen_arm_indices = self._current_movie_indices[range(self._batch_size),
                                                     action]
    return self._approx_ratings_matrix[self._current_user_indices,
                                       chosen_arm_indices]

  def _rewards_for_all_actions(self):
    rewards_matrix = self._approx_ratings_matrix[
        np.expand_dims(self._previous_user_indices, axis=-1),
        self._previous_movie_indices]
    return rewards_matrix

  def compute_optimal_action(self):
    return np.argmax(self._rewards_for_all_actions(), axis=-1)

  def compute_optimal_reward(self):
    return np.max(self._rewards_for_all_actions(), axis=-1)
