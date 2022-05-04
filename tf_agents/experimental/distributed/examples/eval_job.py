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

"""Eval job using a variable container to fetch the weights of the policy."""

import functools
import os
from typing import Callable, Iterable, Optional, Sequence, Text

from absl import app
from absl import flags
from absl import logging

import gin

from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
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
flags.DEFINE_integer(
    'task', 0, 'Identifier of the collect task. Must be unique in a job.')
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
    is_running: Optional[Callable[[], bool]] = None,
    eval_interval: int = 1000,
    eval_episodes: int = 1,
    # TODO(b/178225158): Deprecate in favor of the reporting libray when ready.
    return_reporting_fn: Optional[Callable[[int, float], None]] = None
) -> None:
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
    eval_interval: If set, eval is done at the given step interval or as close
      as possible based on polling.
    eval_episodes: Number of episodes to eval.
    return_reporting_fn: Optional callback function of the form `fn(train_step,
      average_return)` which reports the average return to a custom destination.
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
  metrics = actor.collect_metrics(buffer_size=eval_episodes)

  if return_reporting_fn:
    for m in metrics:
      if isinstance(m, py_metrics.AverageReturnMetric):
        average_return_metric = m
        break

  eval_actor = actor.Actor(
      environment,
      policy,
      train_step,
      episodes_per_run=eval_episodes,
      summary_dir=summary_dir,
      summary_interval=eval_interval,
      metrics=metrics + additional_metrics,
      name='eval_actor')

  # Run the experience evaluation loop.
  last_eval_step = 0
  while is_running():

    # Eval every step if no `eval_interval` is set, or if on the first step, or
    # if the step is equal or greater than `last_eval_step` + `eval_interval`.
    # It is very possible when logging a specific interval that the steps evaled
    # will not be exact, e.g. 1001 and then 2003 vs. 1000 and then 2000.
    if (train_step.numpy() == 0 or
        train_step.numpy() >= eval_interval + last_eval_step):
      logging.info('Evaluating using greedy policy at step: %d',
                   train_step.numpy())
      eval_actor.run()
      last_eval_step = train_step.numpy()

    def is_train_step_the_same_or_behind():
      # Checks if the `train_step` received from variable conainer is the same
      # (or behind) the latest evaluated train step (`prev_train_step_value`).
      variable_container.update(variables)
      return train_step.numpy() <= prev_train_step_value

    train_utils.wait_for_predicate(
        wait_predicate_fn=is_train_step_the_same_or_behind)
    prev_train_step_value = train_step.numpy()

    # Optionally report the average return metric via a callback.
    if return_reporting_fn:
      return_reporting_fn(train_step.numpy(), average_return_metric.result())


def run_eval(
    root_dir: Text,
    # TODO(b/178225158): Deprecate in favor of the reporting libray when ready.
    return_reporting_fn: Optional[Callable[[int, float], None]] = None
) -> None:
  """Load the policy and evaluate it.

  Args:
    root_dir: the root directory for this experiment.
    return_reporting_fn: Optional callback function of the form `fn(train_step,
      average_return)` which reports the average return to a custom destination.
  """
  # Wait for the greedy policy to become available, then load it.
  greedy_policy_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR,
                                   learner.GREEDY_POLICY_SAVED_MODEL_DIR)
  policy = train_utils.wait_for_policy(
      greedy_policy_dir, load_specs_from_pbtxt=True)

  # Create the variable container. The weights of the greedy policy is updated
  # from it periodically.
  variable_container = reverb_variable_container.ReverbVariableContainer(
      FLAGS.variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])

  # Prepare summary directory.
  summary_dir = os.path.join(FLAGS.root_dir, learner.TRAIN_DIR, 'eval',
                             str(FLAGS.task))

  # Run the evaluation.
  evaluate(
      summary_dir=summary_dir,
      environment_name=gin.REQUIRED,
      policy=policy,
      variable_container=variable_container,
      return_reporting_fn=return_reporting_fn)


def main(unused_argv: Sequence[Text]) -> None:
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  run_eval(FLAGS.root_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['root_dir', 'variable_container_server_address'])
  multiprocessing.handle_main(functools.partial(app.run, main))
