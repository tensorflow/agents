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

"""Common triggers useful during training.

See `interval_trigger.IntervalTrigger`.
"""

import os
from typing import Mapping, Optional, Sequence, Text, Tuple, Union

from absl import logging
import tensorflow.compat.v2 as tf

from tf_agents.agents import tf_agent
from tf_agents.metrics import py_metric
from tf_agents.policies import async_policy_saver
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import tf_policy
from tf_agents.train import interval_trigger
from tf_agents.train import learner
from tf_agents.train import step_per_second_tracker
from tf_agents.typing import types

ENV_STEP_METADATA_KEY = 'env_step'


class PolicySavedModelTrigger(interval_trigger.IntervalTrigger):
  """Triggers saves policy checkpoints an agent's policy.

  On construction this trigger will generate a saved_model for a:
  `greedy_policy`, a `collect_policy`, and a `raw_policy`. When triggered a
  checkpoint will be saved which can be used to updated any of the saved_model
  policies.
  """

  def __init__(
      self,
      saved_model_dir: Text,
      agent: tf_agent.TFAgent,
      train_step: tf.Variable,
      interval: int,
      async_saving: bool = False,
      metadata_metrics: Optional[Mapping[Text, py_metric.PyMetric]] = None,
      start: int = 0,
      extra_concrete_functions: Optional[Sequence[
          Tuple[str, policy_saver.def_function.Function]]] = None,
      batch_size: Optional[int] = None,
      use_nest_path_signatures: bool = True,
      save_greedy_policy=True,
      save_collect_policy=True,
      input_fn_and_spec: Optional[policy_saver.InputFnAndSpecType] = None,
  ):
    """Initializes a PolicySavedModelTrigger.

    Args:
      saved_model_dir: Base dir where checkpoints will be saved.
      agent: Agent to extract policies from.
      train_step: `tf.Variable` which keeps track of the number of train steps.
      interval: How often, in train_steps, the trigger will save. Note that as
        long as the >= `interval` number of steps have passed since the last
        trigger, the event gets triggered. The current value is not necessarily
        `interval` steps away from the last triggered value.
      async_saving: If True saving will be done asynchronously in a separate
        thread. Note if this is on the variable values in the saved
        checkpoints/models are not deterministic.
      metadata_metrics: A dictionary of metrics, whose `result()` method returns
        a scalar to be saved along with the policy. Currently only supported
        when async_saving is False.
      start: Initial value for the trigger passed directly to the base class. It
        helps control from which train step the weigts of the model are saved.
      extra_concrete_functions: Optional sequence of extra concrete functions to
        register in the policy savers. The sequence should consist of tuples
        with string name for the function and the tf.function to register. Note
        this does not support adding extra assets.
      batch_size: The number of batch entries the policy will process at a time.
        This must be either `None` (unknown batch size) or a python integer.
      use_nest_path_signatures: SavedModel spec signatures will be created based
        on the sructure of the specs. Otherwise all specs must have unique
        names.
      save_greedy_policy: Disable when an agent's policy distribution method
        does not support mode.
      save_collect_policy: Disable when not saving collect policy.
      input_fn_and_spec: A `(input_fn, tensor_spec)` tuple where input_fn is a
        function that takes inputs according to tensor_spec and converts them to
        the `(time_step, policy_state)` tuple that is used as the input to the
        action_fn. When `input_fn_and_spec` is set, `tensor_spec` is the input
        for the action signature. When `input_fn_and_spec is None`, the action
        signature takes as input `(time_step, policy_state)`.
    """
    if async_saving and metadata_metrics:
      raise NotImplementedError('Support for metadata_metrics is not '
                                'implemented for async policy saver.')

    self._agent = agent
    self._train_step = train_step
    self._async_saving = async_saving
    self._metadata_metrics = metadata_metrics or {}
    self._metadata = {
        k: tf.Variable(0, dtype=v.result().dtype, shape=v.result().shape)
        for k, v in self._metadata_metrics.items()
    }
    self._input_fn_and_spec = input_fn_and_spec

    greedy = None
    if isinstance(agent.policy, greedy_policy.GreedyPolicy):
      raw_policy = agent.policy.wrapped_policy
      greedy = agent.policy
    else:
      raw_policy = agent.policy
      if save_greedy_policy:
        greedy = greedy_policy.GreedyPolicy(agent.policy)

    self._raw_policy_saver = self._build_saver(raw_policy, batch_size,
                                               use_nest_path_signatures)
    savers = [(self._raw_policy_saver, learner.RAW_POLICY_SAVED_MODEL_DIR)]

    if save_collect_policy:
      collect_policy_saver = self._build_saver(agent.collect_policy, batch_size,
                                               use_nest_path_signatures)

      savers.append(
          (collect_policy_saver, learner.COLLECT_POLICY_SAVED_MODEL_DIR))

    if save_greedy_policy:
      greedy_policy_saver = self._build_saver(greedy, batch_size,
                                              use_nest_path_signatures)
      savers.append(
          (greedy_policy_saver, learner.GREEDY_POLICY_SAVED_MODEL_DIR))

    # Save initial saved_model if they do not exist yet. These can be updated
    # from the policy_checkpoints.
    raw_policy_specs_path = os.path.join(saved_model_dir,
                                         learner.RAW_POLICY_SAVED_MODEL_DIR,
                                         'policy_specs.pbtxt')

    extra_concrete_functions = extra_concrete_functions or []
    for saver, _ in savers:
      for name, fn in extra_concrete_functions:
        saver.register_concrete_function(name, fn)

    self._checkpoint_dir = os.path.join(saved_model_dir,
                                        learner.POLICY_CHECKPOINT_DIR)

    # TODO(b/173815037): Use a TF-Agents util to check for whether a saved
    # policy already exists.
    if not tf.io.gfile.exists(raw_policy_specs_path):
      for saver, path in savers:
        saver.save(os.path.join(saved_model_dir, path))

    super(PolicySavedModelTrigger, self).__init__(
        interval, self._save_fn, start=start)

  def _build_saver(
      self,
      policy: tf_policy.TFPolicy,
      batch_size: Optional[int] = None,
      use_nest_path_signatures: bool = True,
  ) -> Union[policy_saver.PolicySaver, async_policy_saver.AsyncPolicySaver]:
    saver = policy_saver.PolicySaver(
        policy,
        batch_size=batch_size,
        train_step=self._train_step,
        metadata=self._metadata,
        use_nest_path_signatures=use_nest_path_signatures,
        input_fn_and_spec=self._input_fn_and_spec,
    )
    if self._async_saving:
      saver = async_policy_saver.AsyncPolicySaver(saver)
    return saver

  def _save_fn(self) -> None:
    self._agent.post_process_policy()
    for k, v in self._metadata_metrics.items():
      self._metadata[k].assign(v.result())
    self._raw_policy_saver.save_checkpoint(
        os.path.join(self._checkpoint_dir,
                     'policy_checkpoint_%010d' % self._train_step.numpy()))


class StepPerSecondLogTrigger(interval_trigger.IntervalTrigger):
  """Logs train_steps_per_second."""

  def __init__(self,
               train_step: tf.Variable,
               interval: int,
               log_to_terminal: bool = True):
    """Initializes a StepPerSecondLogTrigger.

    Args:
      train_step: `tf.Variable` which keeps track of the number of train steps.
      interval: How often, in train_steps, the trigger will save. Note that as
        long as the >= `interval` number of steps have passed since the last
        trigger, the event gets triggered. The current value is not necessarily
        `interval` steps away from the last triggered value.
      log_to_terminal: If True ouputs steps/sec to logging.info.
    """
    self._train_step = train_step
    self._log_to_terminal = log_to_terminal
    self._step_timer = step_per_second_tracker.StepPerSecondTracker(train_step)

    super(StepPerSecondLogTrigger, self).__init__(interval,
                                                  self._log_steps_per_sec)

  def _log_steps_per_sec(self) -> None:
    steps_per_sec = self._step_timer.steps_per_second()
    self._step_timer.restart()
    step = self._train_step.numpy()
    if self._log_to_terminal:
      logging.info('Step: %d, %.3f steps/sec', step, steps_per_sec)
    with tf.compat.v2.summary.record_if(True):
      tf.summary.scalar(name='train_steps_per_sec',
                        data=steps_per_sec,
                        step=step)


class ReverbCheckpointTrigger(interval_trigger.IntervalTrigger):
  """Checkpoints data from Reverb replay buffer."""

  def __init__(self, train_step: tf.Variable, interval: int,
               reverb_client: types.ReverbClient):
    """Initializes a StepPerSecondLogTrigger.

    Args:
      train_step: `tf.Variable` which keeps track of the number of train steps.
      interval: How often, in train_steps, the trigger will save. Note that as
        long as the >= `interval` number of steps have passed since the last
        trigger, the event gets triggered. The current value is not necessarily
        `interval` steps away from the last triggered value.
      reverb_client: the Reverb client required for checkpointing.
    """
    self._train_step = train_step
    self._reverb_client = reverb_client

    super(ReverbCheckpointTrigger, self).__init__(interval, self._save_fn)

  def _save_fn(self) -> None:
    checkpoint_path = self._reverb_client.checkpoint()
    logging.info('Checkpointing Reverb data to %s', checkpoint_path)
