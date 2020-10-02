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

"""Unit tests for the PPOLearner in the TPU setup."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import tensorflow as tf

from tf_agents.experimental.examples.ppo import ppo_learner
from tf_agents.experimental.examples.ppo import ppo_learner_test_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class DistributedPpoLearnerTest(test_utils.TestCase):

  def _get_tpu_strategy(self):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.experimental.TPUStrategy(resolver)

  def testPPOLearnerRunTPU(self):
    if tf.config.list_logical_devices('TPU'):
      tpu_strategy = self._get_tpu_strategy()
    else:
      logging.info('TPU hardware is not available, TPU strategy test skipped.')
      return

    batch_size = 1
    minibatch_size = 5
    num_epochs = 3
    n_time_steps = 10
    num_replicas = tpu_strategy.num_replicas_in_sync

    # Create a dataset with 10 element of length 10. This simulates a Reverb
    # dataset.
    num_collected_episodes = 20
    traj = ppo_learner_test_utils.create_trajectories(
        n_time_steps=n_time_steps, batch_size=batch_size)
    info = ()

    def dataset_fn():
      return tf.data.Dataset.from_tensors(
          (traj, info),).repeat(num_collected_episodes)

    with tpu_strategy.scope():
      print('Number of devices for the strategy: {}'.format(
          tpu_strategy.num_replicas_in_sync))
      fake_agent = ppo_learner_test_utils.FakePPOAgent(tpu_strategy)

    learner = ppo_learner.PPOLearner(
        root_dir=FLAGS.test_tmpdir,
        train_step=tf.Variable(0, dtype=tf.int32),
        agent=fake_agent,
        experience_dataset_fn=dataset_fn,
        normalization_dataset_fn=dataset_fn,
        num_batches=num_collected_episodes,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        # Disable shuffling to have deterministic input into agent.train.
        shuffle_buffer_size=1,
        triggers=None,
        strategy=tpu_strategy)

    learner.run()

    # Check that fake agent was called the expected number of times.
    num_train_frames = (
        num_collected_episodes * batch_size * n_time_steps * num_epochs)
    num_minibatches = num_train_frames / minibatch_size
    num_minibatches_per_replica = int(num_minibatches / num_replicas)
    self.assertEqual(fake_agent.train_called_times.numpy(),
                     num_minibatches_per_replica)

    # Check that fake agent was called the expected number of times the second
    # time it is called.
    fake_agent.reset()
    learner.run()
    self.assertEqual(fake_agent.train_called_times.numpy(),
                     num_minibatches_per_replica)


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
