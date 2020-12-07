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

"""Helper functions for the environments that load datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gin
import numpy as np

from six.moves import range
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

tfd = tfp.distributions


MOVIELENS_NUM_USERS = 943
MOVIELENS_NUM_MOVIES = 1682


def _validate_mushroom_data(numpy_data):
  """Checks if the numpy array looks like the mushroom dataset.

  Args:
    numpy_data: numpy array of rank 2 consisting of single characters. It should
    contain the mushroom dataset with each column being a feature and each row
    being a sample.
  """
  assert numpy_data.shape[1] == 23, 'The dataset should have 23 columns.'
  assert set(numpy_data[:, 0]) == {
      'e', 'p'
  }, 'The first column should be the label with values `e` and `p`.'


def _one_hot(data):
  """Encodes columns of a numpy array as one-hot.

  Args:
    data: A numpy array of rank 2. Every column is a categorical feature and
      every row is a sample.

  Returns:
    A 0/1 numpy array of rank 2 containing the same number of rows as the input.
    The number of columns is equal to the sum of distinct elements per column of
    the input array.
  """
  num_rows, num_cols = np.shape(data)
  encoded = np.array([], dtype=np.int32).reshape((num_rows, 0))
  for i in range(num_cols):
    vocabulary = sorted(list(set(data[:, i])))
    lookup = dict(list(zip(vocabulary, list(range(len(vocabulary))))))
    int_encoded = np.array([lookup[x] for x in data[:, i]])
    new_cols = np.eye(len(vocabulary), dtype=np.int32)[int_encoded]
    encoded = np.append(encoded, new_cols, axis=1)
  return encoded


@gin.configurable
def convert_mushroom_csv_to_tf_dataset(file_path, buffer_size=40000):
  """Converts the mushroom CSV dataset into a `tf.Dataset`.

  The dataset CSV contains the label in the first column, then the features.
  Two example rows:
    `p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u`: poisonous;
    `e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g`: edible.

  Args:
    file_path: Path to the CSV file.
    buffer_size: The buffer to use for shuffling the data.

  Returns:
    A `tf.Dataset`, infinitely looped, shuffled, not batched.

  Raises:
    AssertionError: If the CSV file does not conform to the syntax of the
      mushroom environment.
  """
  with tf.io.gfile.GFile(file_path, 'r') as infile:
    nd = np.genfromtxt(infile, dtype=np.str, delimiter=',')
  _validate_mushroom_data(nd)
  encoded = _one_hot(nd)
  contexts = encoded[:, 2:]
  context_tensor = tf.cast(contexts, tf.float32)
  labels = encoded[:, 0]
  label_tensor = tf.cast(labels, tf.int32)
  dataset = tf.data.Dataset.from_tensor_slices((context_tensor, label_tensor))
  return dataset.repeat().shuffle(buffer_size=buffer_size)


@gin.configurable
def mushroom_reward_distribution(r_noeat, r_eat_safe, r_eat_poison_bad,
                                 r_eat_poison_good, prob_poison_bad):
  """Creates a distribution for rewards for the mushroom environment.

  Args:
    r_noeat: (float) Reward value for not eating the mushroom.
    r_eat_safe: (float) Reward value for eating an edible mushroom.
    r_eat_poison_bad: (float) Reward value for eating and getting poisoned from
      a poisonous mushroom.
    r_eat_poison_good: (float) Reward value for surviving after eating a
      poisonous mushroom.
    prob_poison_bad: Probability of getting poisoned by a poisonous mushroom.

  Returns:
    A reward distribution table, instance of `tfd.Distribution`.
  """

  # The function works by first creating a 2x2 Bernoulli with all but one having
  # parameter 0. The fourth one, that corresponds to eating a poisonous mushroom
  # has parameter `prob_poison_bad`. Then, the whole table is shifted and scaled
  # to the desired values.

  distr = tfd.Bernoulli(probs=[[0, prob_poison_bad], [0, 0]], dtype=tf.float32)
  reward_distr = (
      tfp.bijectors.Shift(
          [[r_noeat, r_eat_poison_bad],
           [r_noeat, r_eat_safe]])
      (tfp.bijectors.Scale(
          [[1, r_eat_poison_good - r_eat_poison_bad],
           [1, 1]])
       (distr)))
  return tfd.Independent(reward_distr, reinterpreted_batch_ndims=2)


def convert_covertype_dataset(file_path, buffer_size=40000):
  with tf.io.gfile.GFile(file_path, 'r') as infile:
    data_array = np.genfromtxt(infile, dtype=np.int, delimiter=',')
  contexts = data_array[:, :-1]
  context_tensor = tf.cast(contexts, tf.float32)
  labels = data_array[:, -1] - 1  # Classes are from [1, 7].
  label_tensor = tf.cast(labels, tf.int32)
  return tf.data.Dataset.from_tensor_slices(
      (context_tensor, label_tensor)).repeat().shuffle(buffer_size=buffer_size)


def load_movielens_data(data_file):
  """Loads the movielens data and returns the ratings matrix."""
  ratings_matrix = np.zeros([MOVIELENS_NUM_USERS, MOVIELENS_NUM_MOVIES])
  with tf.io.gfile.GFile(data_file, 'r') as infile:
    # The file is a csv with rows containing:
    # user id | item id | rating | timestamp
    reader = csv.reader(infile)
    for row in reader:
      user_id, item_id, rating, _ = row
      ratings_matrix[int(user_id) - 1, int(item_id) - 1] = float(rating)
  return ratings_matrix
