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

"""Evaluates policy saved_model checkpoints during training."""

import functools
import os
import time
from typing import Callable, List, Optional, Sequence, Text

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.google.experimental.examples.launchpad import metric_writers
from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.typing import types

# File name for tracking the last evaluated checkpoint. This is useful if the
# evaluator gets preempted.
LAST_EVALUATED_CHECKPOINT_FILE = 'last_evaluated.txt'
# Time to wait before checking if new checkpoints become available.
CHECKPOINT_RETRY_SLEEP = 5  # seconds

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'node_id', 0, 'Identifier of the task. Must be unique in a job.')
flags.DEFINE_string('env_name', None, 'Name of the environment.')


FLAGS = flags.FLAGS


@gin.configurable
class EvaluatorWorker():
  """Launchpad worker for Actors."""

  def __init__(
      self,
      summary_dir: str,
      checkpoint_dir: str,
      policy: py_tf_eager_policy.SavedModelPyTFEagerPolicy,
      node_id: int,
      env_name: Text,
      env_load_fn: Callable[[Text],
                            py_environment.PyEnvironment] = suite_mujoco.load,
      num_eval_episodes: int = 1,
      max_train_step: int = 3_000_000,
      metrics: Optional[List[types.Observer]] = None,
      num_retries: int = 5,
  ):
    self._env = env_load_fn(env_name)
    self._summary_dir = summary_dir
    self._checkpoint_dir = checkpoint_dir
    self._policy = policy
    self._max_train_step = max_train_step

    self._metrics = [
        py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]
    self._metrics.extend(metrics or [])

    self._num_eval_epsiodes = num_eval_episodes
    self._actor = actor.Actor(
        self._env,
        self._policy,
        tf.Variable(self._policy.get_train_step()),
        episodes_per_run=1,
        summary_dir=self._summary_dir,
        metrics=self._metrics,
        observers=None)
    self._restore_evaluated_checkpoints()

    # TODO(b/195434183): Create metric aggregator client and add to writer.
    self._metric_writer = metric_writers.create_default_writer(
        worker='eval',
        node_id=node_id,
        logdir=summary_dir,
    )
    self._num_retries = num_retries

  def _restore_evaluated_checkpoints(self):
    """Restore evaluated checkpoints.

    Mostly useful if this job was pre-empted in the past. Assumes sorting of
    checkpoint paths is linear w.r.t. train_step.
    """

    self._evaluated_checkpoints = set()
    available_checkpoints = self._get_checkpoints_to_evaluate()

    self._last_evaluated_checkpoints_file_path = os.path.join(
        self._summary_dir, LAST_EVALUATED_CHECKPOINT_FILE)

    if available_checkpoints and tf.io.gfile.exists(
        self._last_evaluated_checkpoints_file_path):

      with tf.io.gfile.GFile(self._last_evaluated_checkpoints_file_path,
                             'r') as f:
        last_evaluated_checkpoint = f.read().strip()

      for chkpt in available_checkpoints:
        if chkpt != last_evaluated_checkpoint:
          self._evaluated_checkpoints.add(chkpt)
        else:
          break
      logging.info('Restored evaluated checkpoints set.')

  def _get_checkpoints_to_evaluate(self):
    checkpoints = tf.io.gfile.glob(os.path.join(self._checkpoint_dir, '*'))
    return sorted(
        list(set(checkpoints) - self._evaluated_checkpoints), reverse=True)

  def run(self):
    """Main logic for running the evaluator till max_train_step is reached."""
    last_eval_step = self._policy.get_train_step()

    checkpoints_list = self._get_checkpoints_to_evaluate()
    while last_eval_step < self._max_train_step:
      if not checkpoints_list:
        logging.info('Waiting on new checkpoints to become available at: %s',
                     self._checkpoint_dir)
        time.sleep(CHECKPOINT_RETRY_SLEEP)
        checkpoints_list = self._get_checkpoints_to_evaluate()
        continue

      checkpoint = checkpoints_list.pop()

      for _ in range(self._num_retries):
        try:
          self._policy.update_from_checkpoint(checkpoint)
          break
        except (tf.errors.OpError, IndexError):
          logging.warning(
              'Encountered an error while evaluating a checkpoint. This can '
              'happen when reading a checkpoint before it is fully written. '
              'Retrying...')
          time.sleep(CHECKPOINT_RETRY_SLEEP)
          pass

      logging.info('Evaluating:\n\tStep:%d\tcheckpoint: %s',
                   self._policy.get_train_step(), checkpoint)

      self._actor.train_step.assign(self._policy.get_train_step())

      for _ in range(self._num_eval_epsiodes):
        self._actor.run()

        # Use data.last because we want to raw data, rather than aggergated by
        # the streaming metric.
        # TODO(b/195434183): Use non streaming metric here instead.
        results = {m.name: m.data.last for m in self._metrics}
        results['train_step'] = self._policy.get_train_step()
        self._metric_writer.write_scalars(self._policy.get_train_step(),
                                          results)

      self._actor.log_metrics()

      self._evaluated_checkpoints.add(checkpoint)
      with tf.io.gfile.GFile(self._last_evaluated_checkpoints_file_path,
                             'w') as f:
        f.write(checkpoint)
      last_eval_step = self._policy.get_train_step()

    logging.info(
        'Finished evaluations sleeping for a bit before killing the experiment.'
    )


def main(unused_argv: Sequence[Text]) -> None:
  logging.set_verbosity(logging.INFO)

  summary_dir = os.path.join(FLAGS.root_dir, learner.TRAIN_DIR, 'eval',
                             str(FLAGS.node_id))
  policy_dir = os.path.join(FLAGS.root_dir, learner.POLICY_SAVED_MODEL_DIR,
                            learner.GREEDY_POLICY_SAVED_MODEL_DIR)
  checkpoint_dir = os.path.join(
      FLAGS.root_dir, learner.TRAIN_DIR, learner.POLICY_CHECKPOINT_DIR)
  policy = train_utils.wait_for_policy(policy_dir, load_specs_from_pbtxt=True)

  eval_worker = EvaluatorWorker(
      summary_dir,
      checkpoint_dir,
      policy,
      node_id=FLAGS.node_id,
      env_name=FLAGS.env_name,
      num_eval_episodes=5,
      max_train_step=1000,)
  eval_worker.run()


if __name__ == '__main__':
  flags.mark_flags_as_required(['root_dir', 'env_name'])
  multiprocessing.handle_main(functools.partial(app.run, main))
