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
"""Eval job using a variable container to fetch the weights of the policy."""

import functools
import os
from typing import Callable, Iterable, Optional, Sequence, Text

from absl import app
from absl import flags
from absl import logging

import gin

import tensorflow.compat.v2 as tf

from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metric
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('variable_container_server_address', None,
                    'Variable container server address.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def evaluate(
    summary_dir: Text,
    environment_name: Text,
    policy: py_tf_eager_policy.PyTFEagerPolicyBase,
    variable_container: reverb_variable_container.ReverbVariableContainer,
    suite_load_fn: Callable[[Text],
                            py_environment.PyEnvironment] = suite_mujoco.load,
    additional_metrics: Optional[Iterable[py_metric.PyStepMetric]] = None,
    is_running: Optional[Callable[[], bool]] = None) -> None:
  """Evaluates a policy iteratively fetching weights from variable container.

  Args:
    summary_dir: Directory which is used to store the summaries.
    environment_name: Name of the environment used to evaluate the policy.
    policy: The policy being evaluated. The weights of this policy are fetched
      from the variable container periodically.
    variable_container: Provides weights for the policy.
    suite_load_fn: Function that loads the environment (by calling it with the
      name of the environment) from a particular suite.
    additional_metrics: Optional collection of metrics that are computed as well
      during the evaluation. By default (`None`) it is empty.
    is_running: Optional callable which controls the running of the main
      evaluation loop (including fetching weights from the variable container
      and running the eval actor periodically). By default (`None`) this is a
      callable always returning `True` resulting in an infinite evaluation loop.
  """
  additional_metrics = additional_metrics or []
  is_running = is_running or (lambda: True)
  environment = suite_load_fn(environment_name)

  # Create the variable container.
  train_step = train_utils.create_train_step()
  variables = {
      reverb_variable_container.POLICY_KEY: policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step
  }
  variable_container.update(variables)
  prev_train_step_value = train_step.numpy()

  # Create the evaluator actor.
  eval_actor = actor.Actor(
      environment,
      policy,
      train_step,
      episodes_per_run=1,
      summary_dir=summary_dir,
      metrics=actor.collect_metrics(buffer_size=1) + additional_metrics,
      name='eval_actor')

  # Run the experience evaluation loop.
  while is_running():
    eval_actor.run()
    logging.info('Evaluating using greedy policy at step: %d',
                 train_step.numpy())

    def is_train_step_the_same_or_behind():
      # Checks if the `train_step` received from variable conainer is the same
      # (or behind) the latest evaluated train step (`prev_train_step_value`).
      variable_container.update(variables)
      return train_step.numpy() <= prev_train_step_value

    train_utils.wait_for_predicate(
        wait_predicate_fn=is_train_step_the_same_or_behind)
    prev_train_step_value = train_step.numpy()


def main(unused_argv: Sequence[Text]) -> None:
  logging.set_verbosity(logging.INFO)
  tf.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  # Wait for the greedy policy to become available, then load it.
  greedy_policy_dir = os.path.join(FLAGS.root_dir,
                                   learner.POLICY_SAVED_MODEL_DIR,
                                   learner.GREEDY_POLICY_SAVED_MODEL_DIR)
  policy = train_utils.wait_for_policy(
      greedy_policy_dir, load_specs_from_pbtxt=True)

  # Create the variable container. The weights of the greedy policy is updated
  # from it periodically.
  variable_container = reverb_variable_container.ReverbVariableContainer(
      FLAGS.variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])

  # Run the evaluation.
  evaluate(
      summary_dir=os.path.join(FLAGS.root_dir, learner.TRAIN_DIR, 'eval'),
      environment_name=gin.REQUIRED,
      policy=policy,
      variable_container=variable_container)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['root_dir', 'variable_container_server_address'])
  multiprocessing.handle_main(functools.partial(app.run, main))
