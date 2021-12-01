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

from absl import flags
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.examples.cql_sac.kumar20 import cql_sac_train_eval

FLAGS = flags.FLAGS
TEST_DATA = 'third_party/py/tf_agents/examples/cql_sac/kumar20/dataset/test_data/antmaze-medium-play-v0_0.tfrecord'
ENV_NAME = 'antmaze-medium-play-v0'


class CqlSacTrainEval(tf.test.TestCase):

  def testBasic(self):
    root_dir = self.get_temp_dir()
    cql_sac_train_eval.train_eval(
        root_dir,
        dataset_path=TEST_DATA,
        env_name=ENV_NAME,
        num_gradient_updates=2,
        batch_size=4,
        eval_interval=2,
        eval_episodes=1,
    )


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
