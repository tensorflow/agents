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

"""Generate D4RL TFRecord dataset that is compatible with TF-Agents."""
# Lint as: python3
import functools
import os

from absl import app
from absl import flags
from absl import logging

import d4rl  # pylint: disable=unused-import
import gym

from tf_agents.experimental.examples.cql_sac.kumar20.dataset import dataset_utils
from tf_agents.experimental.examples.cql_sac.kumar20.dataset import file_utils
from tf_agents.system import system_multiprocessing as multiprocessing
# Using XM.  # pylint: disable=unused-import

flags.DEFINE_string('root_dir', '/tmp/dataset/', 'Output dataset directory.')
flags.DEFINE_string(
    'env_name', 'hopper-medium-v0', 'Env name. '
    'Should match one of keys in d4rl.infos.DATASET_URLS')
flags.DEFINE_integer('replicas', None,
                     'Number of parallel replicas generating evaluations.')
flags.DEFINE_integer(
    'replica_id', None,
    'Replica id. If not None, only generate for this replica slice.')
flags.DEFINE_bool(
    'use_trajectories', False,
    'Whether to save samples as trajectories. If False, save as transitions.')
flags.DEFINE_bool(
    'exclude_timeouts', False, 'Whether to exclude the final episode step '
    'if it from a timeout instead of a terminal.')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  d4rl_env = gym.make(FLAGS.env_name)
  d4rl_dataset = d4rl_env.get_dataset()
  root_dir = os.path.join(FLAGS.root_dir, FLAGS.env_name)

  dataset_dict = dataset_utils.create_episode_dataset(
      d4rl_dataset,
      FLAGS.exclude_timeouts,
      observation_dtype=d4rl_env.observation_space.dtype)
  num_episodes = len(dataset_dict['episode_start_index'])
  logging.info('Found %d episodes, %s total steps.', num_episodes,
               len(dataset_dict['states']))

  collect_data_spec = dataset_utils.create_collect_data_spec(
      dataset_dict, use_trajectories=FLAGS.use_trajectories)
  logging.info('Collect data spec %s', collect_data_spec)

  num_replicas = FLAGS.replicas or 1
  interval_size = num_episodes // num_replicas + 1

  # If FLAGS.replica_id is set, only run that section of the dataset.
  # This is useful if distributing the replicas on Borg.
  if FLAGS.replica_id is not None:
    file_name = '%s_%d.tfrecord' % (FLAGS.env_name, FLAGS.replica_id)
    start_index = FLAGS.replica_id * interval_size
    end_index = min((FLAGS.replica_id + 1) * interval_size, num_episodes)
    file_utils.write_samples_to_tfrecord(
        dataset_dict=dataset_dict,
        collect_data_spec=collect_data_spec,
        dataset_path=os.path.join(root_dir, file_name),
        start_episode=start_index,
        end_episode=end_index,
        use_trajectories=FLAGS.use_trajectories)
  else:
    # Otherwise, parallelize with tf_agents.system.multiprocessing.
    jobs = []
    context = multiprocessing.get_context()

    for i in range(num_replicas):
      if num_replicas == 1:
        file_name = '%s.tfrecord' % FLAGS.env_name
      else:
        file_name = '%s_%d.tfrecord' % (FLAGS.env_name, i)
      dataset_path = os.path.join(root_dir, file_name)
      start_index = i * interval_size
      end_index = min((i + 1) * interval_size, num_episodes)
      kwargs = dict(
          dataset_dict=dataset_dict,
          collect_data_spec=collect_data_spec,
          dataset_path=dataset_path,
          start_episode=start_index,
          end_episode=end_index,
          use_trajectories=FLAGS.use_trajectories)
      job = context.Process(
          target=file_utils.write_samples_to_tfrecord, kwargs=kwargs)
      job.start()
      jobs.append(job)

    for job in jobs:
      job.join()


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
