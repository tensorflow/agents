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

"""An agent that mixes a list of agents with a constant mixture distribution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import List, Optional, Sequence, Text
import gin
import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.bandits.policies import mixture_policy
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


def _dynamic_partition_of_nested_tensors(
    nested_tensor: types.NestedTensor, partitions: types.Int,
    num_partitions: int) -> List[types.NestedTensor]:
  """This function takes a nested structure and partitions every element of it.

  Specifically it outputs a list of nest that all have the same structure as the
  original, and every element of the list is a nest that contains a dynamic
  partition of the corresponding original tensors.

  Note that this function uses tf.dynamic_partition, and thus
  'MixtureAgent' is not compatible with XLA.

  Args:
    nested_tensor: The input nested structure to partition.
    partitions: int32 tensor based on which the partitioning happens.
    num_partitions: The number of expected partitions.

  Returns:
    A list of nested tensors with the same structure as `nested_tensor`.
  """
  flattened_tensors = tf.nest.flatten(nested_tensor)
  if not flattened_tensors:
    return [nested_tensor] * num_partitions
  partitioned_flat_tensors = [
      tf.dynamic_partition(
          data=t, partitions=partitions, num_partitions=num_partitions)
      for t in flattened_tensors
  ]
  list_of_partitions = list(map(list, zip(*partitioned_flat_tensors)))
  return [
      tf.nest.pack_sequence_as(nested_tensor, i) for i in list_of_partitions
  ]


@gin.configurable
class MixtureAgent(tf_agent.TFAgent):
  """An agent that mixes a set of agents with a given mixture.

  For every data sample, the agent updates the sub-agent that was used to make
  the action choice in that sample. For this update to happen, the mixture agent
  needs to have the information on which sub-agent is "responsible" for the
  action. This information is in a policy info field `mixture_agent_id`.

  Note that this agent makes use of `tf.dynamic_partition`, and thus it is not
  compatible with XLA.
  """

  def __init__(self,
               mixture_distribution: types.Distribution,
               agents: Sequence[tf_agent.TFAgent],
               name: Optional[Text] = None):
    """Initializes an instance of `MixtureAgent`.

    Args:
      mixture_distribution: An instance of `tfd.Categorical` distribution. This
        distribution is used to draw sub-policies by the mixture policy. The
        parameters of the distribution is trained by the mixture agent.
      agents: List of instances of TF-Agents bandit agents. These agents will be
        trained and used to select actions. The length of this list should match
        that of `mixture_weights`.
      name: The name of this instance of `MixtureAgent`.
    """
    tf.Module.__init__(self, name=name)
    time_step_spec = agents[0].time_step_spec
    action_spec = agents[0].action_spec
    self._original_info_spec = agents[0].policy.info_spec
    error_message = None
    for agent in agents[1:]:
      if action_spec != agent.action_spec:
        error_message = 'Inconsistent action specs.'
      if time_step_spec != agent.time_step_spec:
        error_message = 'Inconsistent time step specs.'
      if self._original_info_spec != agent.policy.info_spec:
        error_message = 'Inconsistent info specs.'
    if error_message is not None:
      raise ValueError(error_message)
    self._agents = agents
    self._num_agents = len(agents)
    self._mixture_distribution = mixture_distribution
    policies = [agent.collect_policy for agent in agents]
    policy = mixture_policy.MixturePolicy(mixture_distribution, policies)
    super(MixtureAgent, self).__init__(
        time_step_spec, action_spec, policy, policy, train_sequence_length=None)
    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)
    for agent in self._agents:
      agent.initialize()

  # Subclasses must implement this method.
  @abc.abstractmethod
  def _update_mixture_distribution(self, experience):
    """This function updates the mixture weights given training experience."""
    raise NotImplementedError('`_update_mixture_distribution` should be '
                              'implemented by subclasses of `MixtureAgent`.')

  def _train(self, experience, weights=None):
    del weights  # unused
    experience = self._as_trajectory(experience)

    reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._time_step_spec.reward)
    action, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.action, self._action_spec)
    observation, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.observation, self._time_step_spec.observation)
    policy_choice, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.policy_info[mixture_policy.MIXTURE_AGENT_ID],
        self._time_step_spec.reward)
    original_infos, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.policy_info[mixture_policy.SUBPOLICY_INFO],
        self._original_info_spec)

    partitioned_nested_infos = nest_utils.batch_nested_tensors(
        _dynamic_partition_of_nested_tensors(original_infos, policy_choice,
                                             self._num_agents))

    partitioned_nested_rewards = [
        nest_utils.batch_nested_tensors(t)
        for t in _dynamic_partition_of_nested_tensors(reward, policy_choice,
                                                      self._num_agents)
    ]
    partitioned_nested_actions = [
        nest_utils.batch_nested_tensors(t)
        for t in _dynamic_partition_of_nested_tensors(action, policy_choice,
                                                      self._num_agents)
    ]
    partitioned_nested_observations = [
        nest_utils.batch_nested_tensors(t)
        for t in _dynamic_partition_of_nested_tensors(
            observation, policy_choice, self._num_agents)
    ]
    loss = 0
    for k in range(self._num_agents):
      per_policy_experience = trajectory.single_step(
          observation=partitioned_nested_observations[k],
          action=partitioned_nested_actions[k],
          policy_info=partitioned_nested_infos[k],
          reward=partitioned_nested_rewards[k],
          discount=tf.zeros_like(partitioned_nested_rewards[k]))
      loss_info = self._agents[k].train(per_policy_experience)
      loss += loss_info.loss
    common.function_in_tf1()(self._update_mixture_distribution)(experience)
    return tf_agent.LossInfo(loss=(loss), extra=())
