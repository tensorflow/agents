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

r"""Sample collection Job using a variable container for policy updates.

See README for launch instructions.
"""

import os
from typing import Callable, Text

from absl import app
from absl import flags
from absl import logging

import gin
import reverb
import tensorflow.compat.v2 as tf

from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', None, 'Name of the environment')
flags.DEFINE_string('replay_buffer_server_address', None,
                    'Replay buffer server address.')
flags.DEFINE_string('variable_container_server_address', None,
                    'Variable container server address.')
flags.DEFINE_integer(
    'task', 0, 'Identifier of the collect task. Must be unique in a job.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def collect(summary_dir: Text,
            environment_name: Text,
            collect_policy: py_tf_eager_policy.PyTFEagerPolicyBase,
            replay_buffer_server_address: Text,
            variable_container_server_address: Text,
            suite_load_fn: Callable[
                [Text], py_environment.PyEnvironment] = suite_mujoco.load,
            initial_collect_steps: int = 10000,
            max_train_steps: int = 2000000) -> None:
  """Collects experience using a policy updated after every episode."""
  # Create the environment. For now support only single environment collection.
  collect_env = suite_load_fn(environment_name)

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

  random_policy = random_py_policy.RandomPyPolicy(collect_env.time_step_spec(),
                                                  collect_env.action_spec())
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
      summary_dir=summary_dir,
      observers=[rb_observer, env_step_metric])

  # Run the experience collection loop.
  while train_step.numpy() < max_train_steps:
    logging.info('Collecting with policy at step: %d', train_step.numpy())
    collect_actor.run()
    variable_container.update(variables)


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  # Wait for the collect policy to become available, then load it.
  collect_policy_dir = os.path.join(FLAGS.root_dir,
                                    learner.POLICY_SAVED_MODEL_DIR,
                                    learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True)

  # Prepare summary directory.
  summary_dir = os.path.join(FLAGS.root_dir, learner.TRAIN_DIR, str(FLAGS.task))

  # Perform collection.
  collect(
      summary_dir=summary_dir,
      environment_name=FLAGS.env_name,
      collect_policy=collect_policy,
      replay_buffer_server_address=FLAGS.replay_buffer_server_address,
      variable_container_server_address=FLAGS.variable_container_server_address)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir', 'env_name', 'replay_buffer_server_address',
      'variable_container_server_address'
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
