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

"""Tests for tf_agents.experimental.examples.cql_sac.kumar20.cql_sac_train_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import error

import rlds
import tensorflow as tf
import tensorflow_datasets as tfds

from tf_agents.examples.cql_sac.kumar20 import cql_sac_train_eval

_ENV_NAME = 'antmaze-medium-play-v0'
_DATASET_NAME = 'd4rl_antmaze/medium-play-v0'
_COUNT = 10


def _load_test_rlds(dataset_name: str) -> tf.data.Dataset:
  with tfds.testing.mock_data(num_examples=20):
    return rlds.load(dataset_name)


class CqlSacTrainEval(tf.test.TestCase):

  def testBasicTrainingSuccess(self):
    root_dir = self.get_temp_dir()
    cql_sac_train_eval.train_eval(
        root_dir,
        dataset_name=_DATASET_NAME,
        load_dataset_fn=_load_test_rlds,
        env_name=_ENV_NAME,
        num_gradient_updates=2,
        batch_size=4,
        eval_interval=2,
        eval_episodes=1,
        data_take=_COUNT,
        data_prefetch=1,
        pad_end_of_episodes=True)

  def testTransformationsTrainingSuccess(self):
    root_dir = self.get_temp_dir()
    cql_sac_train_eval.train_eval(
        root_dir,
        dataset_name=_DATASET_NAME,
        load_dataset_fn=_load_test_rlds,
        env_name=_ENV_NAME,
        num_gradient_updates=2,
        reward_shift=0.1,
        action_clipping=(0.1, 0.9),
        batch_size=4,
        eval_interval=2,
        eval_episodes=1,
        data_take=_COUNT,
        data_prefetch=1,
        pad_end_of_episodes=True)

  def testIncorrectDatasetTrainingFails(self):
    root_dir = self.get_temp_dir()
    with self.assertRaises(ValueError):
      cql_sac_train_eval.train_eval(
          root_dir,
          dataset_name='random_dataset_name',
          load_dataset_fn=_load_test_rlds,
          env_name=_ENV_NAME,
          num_gradient_updates=2,
          reward_shift=0.1,
          action_clipping=(0.1, 0.9),
          batch_size=4,
          eval_interval=2,
          eval_episodes=1,
          data_take=_COUNT,
          data_prefetch=1,
          pad_end_of_episodes=True)

  def testIncorrectEnvironmentTrainingFails(self):
    root_dir = self.get_temp_dir()
    with self.assertRaises(error.Error):
      cql_sac_train_eval.train_eval(
          root_dir,
          dataset_name=_DATASET_NAME,
          load_dataset_fn=_load_test_rlds,
          env_name='random_env_name',
          num_gradient_updates=2,
          reward_shift=0.1,
          action_clipping=(0.1, 0.9),
          batch_size=4,
          eval_interval=2,
          eval_episodes=1,
          data_take=_COUNT,
          data_prefetch=1,
          pad_end_of_episodes=True)


if __name__ == '__main__':
  tf.test.main()
