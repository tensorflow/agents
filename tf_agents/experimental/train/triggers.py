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

"""Common triggers useful during training.

See `interval_trigger.IntervalTrigger`.
"""

import os
from typing import Mapping, Optional, Text, Union

from absl import logging
import tensorflow.compat.v2 as tf

from tf_agents.agents import tf_agent
from tf_agents.experimental.train import interval_trigger
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import step_per_second_tracker
from tf_agents.metrics import py_metric
from tf_agents.policies import async_policy_saver
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import tf_policy

ENV_STEP_METADATA_KEY = 'env_step'


class PolicySavedModelTrigger(interval_trigger.IntervalTrigger):
  """Triggers saves policy checkpoints an agent's policy.

  On construction this trigger will generate a saved_model for a:
  `greedy_policy`, a `collect_policy`, and a `raw_policy`. When triggered a
  checkpoint will be saved which can be used to updated any of the saved_model
  policies.
  """

  def __init__(self,
               saved_model_dir: Text,
               agent: tf_agent.TFAgent,
               train_step: tf.Variable,
               interval: int,
               async_saving: bool = False,
               metadata_metrics: Optional[Mapping[Text,
                                                  py_metric.PyMetric]] = None,
               start: int = 0):
    """Initializes a PolicySavedModelTrigger.

    Args:
      saved_model_dir: Base dir where checkpoints will be saved.
      agent: Agent to extract policies from.
      train_step: `tf.Variable` which keeps track of the number of train steps.
      interval: How often, in train_steps, the trigger will save. Note that as
        long as the >= `interval` number of steps have passed since the
        last trigger, the event gets triggered. The current value is not
        necessarily `interval` steps away from the last triggered value.
      async_saving: If True saving will be done asynchronously in a separate
        thread. Note if this is on the variable values in the saved
        checkpoints/models are not deterministic.
      metadata_metrics: A dictionary of metrics, whose `result()` method returns
        a scalar to be saved along with the policy. Currently only supported
        when async_saving is False.
      start: Initial value for the trigger passed directly to the base class. It
        helps control from which train step the weigts of the model are saved.
    """
    if async_saving and metadata_metrics:
      raise NotImplementedError('Support for metadata_metrics is not '
                                'implemented for async policy saver.')

    self._train_step = train_step
    self._async_saving = async_saving
    self._metadata_metrics = metadata_metrics or {}
    self._metadata = {
        k: tf.Variable(0, dtype=v.result().dtype, shape=v.result().shape)
        for k, v in self._metadata_metrics.items()
    }

    collect_policy_saver = self._build_saver(agent.collect_policy)

    # TODO(b/145754641): Fix how greedy/raw policies are built in agents.
    if isinstance(agent.policy, greedy_policy.GreedyPolicy):
      greedy = agent.policy
    else:
      greedy = greedy_policy.GreedyPolicy(agent.policy)

    collect_policy_saver = self._build_saver(agent.collect_policy)
    greedy_policy_saver = self._build_saver(greedy)
    self._raw_policy_saver = self._build_saver(greedy.wrapped_policy)

    # Save initial saved_model. These can be updated from the
    # policy_checkpoints.
    collect_policy_saver.save(
        os.path.join(saved_model_dir, learner.COLLECT_POLICY_SAVED_MODEL_DIR))
    greedy_policy_saver.save(
        os.path.join(saved_model_dir, learner.GREEDY_POLICY_SAVED_MODEL_DIR))
    self._raw_policy_saver.save(
        os.path.join(saved_model_dir, learner.RAW_POLICY_SAVED_MODEL_DIR))

    self._checkpoint_dir = os.path.join(saved_model_dir,
                                        learner.POLICY_CHECKPOINT_DIR)
    super(PolicySavedModelTrigger, self).__init__(
        interval, self._save_fn, start=start)

  def _build_saver(
      self, policy: tf_policy.TFPolicy
  ) -> Union[policy_saver.PolicySaver, async_policy_saver.AsyncPolicySaver]:
    saver = policy_saver.PolicySaver(
        policy, train_step=self._train_step, metadata=self._metadata)
    if self._async_saving:
      saver = async_policy_saver.AsyncPolicySaver(saver)
    return saver

  def _save_fn(self) -> None:
    for k, v in self._metadata_metrics.items():
      self._metadata[k].assign(v.result())
    self._raw_policy_saver.save_checkpoint(
        os.path.join(self._checkpoint_dir,
                     'policy_checkpoint_%010d' % self._train_step.numpy()))


class StepPerSecondLogTrigger(interval_trigger.IntervalTrigger):
  """Logs train_steps_per_second."""

  def __init__(self, train_step: tf.Variable, interval: int):
    """Initializes a StepPerSecondLogTrigger.

    Args:
      train_step: `tf.Variable` which keeps track of the number of train steps.
      interval: How often, in train_steps, the trigger will save. Note that as
        long as the >= `interval` number of steps have passed since the
        last trigger, the event gets triggered. The current value is not
        necessarily `interval` steps away from the last triggered value.
    """
    self._train_step = train_step
    self._step_timer = step_per_second_tracker.StepPerSecondTracker(train_step)

    super(StepPerSecondLogTrigger, self).__init__(interval,
                                                  self._log_steps_per_sec)

  def _log_steps_per_sec(self) -> None:
    steps_per_sec = self._step_timer.steps_per_second()
    self._step_timer.restart()
    step = self._train_step.numpy()
    logging.info('Step: %d, %.3f steps/sec', step,
                 steps_per_sec)
    tf.summary.scalar(
        name='train_steps_per_sec',
        data=steps_per_sec,
        step=step)
