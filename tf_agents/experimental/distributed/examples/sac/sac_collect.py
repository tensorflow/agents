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

# Lint as: python3
r"""Sample collection Job using a variable container for policy updates.

See README for launch instructions.
"""

import os

from absl import app
from absl import flags
from absl import logging

import reverb
import tensorflow.compat.v2 as tf

from tf_agents.environments import suite_mujoco
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('replay_buffer_server_address', None,
                    'Replay buffer server address.')
flags.DEFINE_string('variable_container_server_address', None,
                    'Variable container server address.')
flags.DEFINE_integer(
    'task', 0, 'Identifier of the collect task. Must be unique in a job.')

FLAGS = flags.FLAGS


def collect(task,
            root_dir,
            replay_buffer_server_address,
            variable_container_server_address,
            create_env_fn,
            initial_collect_steps=10000,
            num_iterations=10000000):
  """Collects experience using a policy updated after every episode."""
  # Create the environment. For now support only single environment collection.
  collect_env = create_env_fn()

  # Create the path for the serialized collect policy.
  collect_policy_saved_model_path = os.path.join(
      root_dir, learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  saved_model_pb_path = os.path.join(collect_policy_saved_model_path,
                                     'saved_model.pb')
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

  # Create the variable container.
  train_step = train_utils.create_train_step()
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.update(variables)

  # Create the replay buffer observer.
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb.Client(replay_buffer_server_address),
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      sequence_length=2,
      stride_length=1)

  random_policy = random_py_policy.RandomPyPolicy(
      collect_env.time_step_spec(), collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  env_step_metric = py_metrics.EnvironmentSteps()
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR, str(task)),
      observers=[rb_observer, env_step_metric])

  # Run the experience collection loop.
  for _ in range(num_iterations):
    logging.info('Collecting with policy at step: %d', train_step.numpy())
    collect_actor.run()
    variable_container.update(variables)


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  collect(
      FLAGS.task,
      FLAGS.root_dir,
      replay_buffer_server_address=FLAGS.variable_container_server_address,
      variable_container_server_address=FLAGS.variable_container_server_address,
      create_env_fn=lambda: suite_mujoco.load('HalfCheetah-v2'))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir', 'replay_buffer_server_address',
      'variable_container_server_address'
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
