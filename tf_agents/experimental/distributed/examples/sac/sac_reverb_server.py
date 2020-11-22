# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
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

# Lint as: python3
r"""Main binary to launch a stand alone Reverb RB server.

See README for launch instructions.
"""

import os

from absl import app
from absl import flags
from absl import logging

import reverb
import tensorflow.compat.v2 as tf

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.experimental.train import learner
from tf_agents.experimental.train.utils import train_utils
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.specs import tensor_spec

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('min_table_size_before_sampling', 1,
                     'Minimum number of elements in table before sampling.')
flags.DEFINE_integer('replay_buffer_capacity', 1000000,
                     'Capacity of the replay buffer table.')
flags.DEFINE_integer('port', None, 'Port to start the server on.')
flags.DEFINE_integer(
    'samples_per_insert', None,
    'Samples per insert limit. To use this option, ensure that '
    'min_table_size_before_sampling >= 2 * max(1.0, samples_per_insert)')

FLAGS = flags.FLAGS

# Ratio for samples per insert rate limiting tolerance
_SAMPLES_PER_INSERT_TOLERANCE_RATIO = 0.1


def main(_):
  logging.set_verbosity(logging.INFO)

  # Create the path for the serialized collect policy.
  collect_policy_saved_model_path = os.path.join(
      FLAGS.root_dir, learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  saved_model_pb_path = os.path.join(collect_policy_saved_model_path,
                                     'saved_model.pb')

  samples_per_insert = FLAGS.samples_per_insert
  min_table_size_before_sampling = FLAGS.min_table_size_before_sampling

  try:
    # Wait for the collect policy to be outputed by learner (timeout after 2
    # days), then load it.
    train_utils.wait_for_file(
        saved_model_pb_path, sleep_time_secs=2, num_retries=86400)
    collect_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        collect_policy_saved_model_path, load_specs_from_pbtxt=True)
  except TimeoutError as e:
    # If the collect policy does not become available during the wait time of
    # the call `wait_for_file`, that probably means the learner is not running.
    logging.error('Could not get the file %s. Exiting.', saved_model_pb_path)
    raise e

  # Create the signature for the variable container holding the policy weights.
  train_step = train_utils.create_train_step()
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
      variables)
  logging.info('Signature of variables: \n%s', variable_container_signature)

  # Create the signature for the replay buffer holding observed experience.
  replay_buffer_signature = tensor_spec.from_spec(
      collect_policy.collect_data_spec)
  logging.info('Signature of experience: \n%s', replay_buffer_signature)

  if samples_per_insert is not None:
    # Use SamplesPerInsertRatio limiter
    samples_per_insert_tolerance = _SAMPLES_PER_INSERT_TOLERANCE_RATIO * samples_per_insert
    error_buffer = min_table_size_before_sampling * samples_per_insert_tolerance

    experience_rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_table_size_before_sampling,
        samples_per_insert=samples_per_insert,
        error_buffer=error_buffer)
  else:
    # Use MinSize limiter
    experience_rate_limiter = reverb.rate_limiters.MinSize(
        min_table_size_before_sampling)

  # Crete and start the replay buffer and variable container server.
  server = reverb.Server(
      tables=[
          reverb.Table(  # Replay buffer storing experience.
              name=reverb_replay_buffer.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=experience_rate_limiter,
              max_size=FLAGS.replay_buffer_capacity,
              max_times_sampled=0,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=1,
              max_times_sampled=0,
              signature=variable_container_signature,
          ),
      ],
      port=FLAGS.port)
  server.wait()


if __name__ == '__main__':
  flags.mark_flags_as_required(['root_dir', 'port'])
  app.run(main)
