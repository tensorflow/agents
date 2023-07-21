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

"""An Thompson sampling agent for Bernoulli bandit problems."""

from typing import Optional, Text, Sequence

import gin
import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.bandits.policies import bernoulli_thompson_sampling_policy as bernoulli_policy
from tf_agents.policies import utils as policy_utilities
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class BernoulliBanditVariableCollection(tf.Module):
  """A collection of variables used by `BernoulliThompsonSamplingAgent`."""

  def __init__(self,
               num_actions: int,
               dtype: tf.DType = tf.float32,
               name: Optional[Text] = None):
    """Initializes an instance of `BernoulliBanditVariableCollection`.

    It creates all the variables needed for `BernoulliThompsonSamplingAgent`.
    For each action, the agent maintains the `alpha` and `beta` parameters of
    the beta distribution.

    Args:
      num_actions: (int) The number of actions.
      dtype: The type of the variables. Should be one of `tf.float32` or
        `tf.float64`.
      name: (string) the name of this instance.
    """
    tf.Module.__init__(self, name=name)
    # It holds the `alpha` parameter of the beta distribution of each arm.
    self.alpha = [
        tf.compat.v2.Variable(tf.ones([], dtype=dtype),
                              name='alpha_{}'.format(k)) for k in range(
                                  num_actions)]
    # It holds the `beta` parameter of the beta distribution of each arm.
    self.beta = [
        tf.compat.v2.Variable(tf.ones([], dtype=dtype),
                              name='beta_{}'.format(k)) for k in range(
                                  num_actions)]


@gin.configurable
class BernoulliThompsonSamplingAgent(tf_agent.TFAgent):
  """A Thompson Sampling agent for non-contextual Bernoulli bandit problems.

  In Bernoulli bandit problems, the reward for each arm is a Bernoulli
  distribution with parameter p. The agent assumes that the prior distribution
  is beta distributed, which is conjugate to the bernoulli distribution and
  the posterior distribution of each arm admits simple incremental updates.
  For a reference, see e.g., Chapter 3 in "A Tutorial on Thompson Sampling" by
  Russo et al. (https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.BoundedTensorSpec,
      variable_collection: Optional[BernoulliBanditVariableCollection] = None,
      dtype: tf.DType = tf.float32,
      batch_size: Optional[int] = 1,
      observation_and_action_constraint_splitter: Optional[
          types.Splitter] = None,
      emit_policy_info: Sequence[Text] = (),
      name: Optional[Text] = None):
    """Creates a Bernoulli Thompson Sampling Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      variable_collection: Instance of `BernoulliBanditVariableCollection`.
        Collection of variables to be updated by the agent. If `None`, a new
        instance of `BernoulliBanditVariableCollection` will be created.
      dtype: The type of the variables. Should be one of `tf.float32` or
        `tf.float64`.
      batch_size: optional int with the batch size. It defaults to 1.
      observation_and_action_constraint_splitter: A function used for masking
        valid/invalid actions with each state of the environment. The function
        takes in a full observation and returns a tuple consisting of 1) the
        part of the observation intended as input to the bandit agent and
        policy, and 2) the boolean mask. This function should also work with a
        `TensorSpec` as input, and should output `TensorSpec` objects for the
        observation and mask.
      emit_policy_info: (tuple of strings) what side information we want to get
        as part of the policy info. Allowed values can be found in
        `policy_utilities.PolicyInfo`.
      name: Python str name of this agent. All variables in this module will
        fall under that name. Defaults to the class name.

    Raises:
      ValueError: If the action spec contains more than one action or it is
        not a bounded scalar int32 spec with minimum 0.
      TypeError: if variable_collection is not an instance of
        `BernoulliBanditVariableCollection`.
    """
    tf.Module.__init__(self, name=name)
    common.tf_agents_gauge.get_cell('TFABandit').set(True)
    self._observation_and_action_constraint_splitter = (
        observation_and_action_constraint_splitter)
    self._num_actions = policy_utilities.get_num_actions_from_tensor_spec(
        action_spec)

    self._dtype = dtype
    if variable_collection is None:
      variable_collection = BernoulliBanditVariableCollection(
          num_actions=self._num_actions,
          dtype=dtype)
    elif not isinstance(variable_collection, BernoulliBanditVariableCollection):
      raise TypeError('Parameter `variable_collection` should be '
                      'of type `BernoulliBanditVariableCollection`.')
    self._variable_collection = variable_collection
    self._alpha = variable_collection.alpha
    self._beta = variable_collection.beta
    self._batch_size = batch_size
    policy = bernoulli_policy.BernoulliThompsonSamplingPolicy(
        time_step_spec,
        action_spec,
        self._alpha,
        self._beta,
        observation_and_action_constraint_splitter,
        emit_policy_info=emit_policy_info)

    super(BernoulliThompsonSamplingAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy=policy,
        train_sequence_length=None)
    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)

  def _initialize(self):
    tf.compat.v1.variables_initializer(self.variables)

  @property
  def alpha(self):
    return self._alpha

  @property
  def beta(self):
    return self._beta

  def _train(self, experience, weights):
    experience = self._as_trajectory(experience)
    reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._time_step_spec.reward)
    reward = tf.clip_by_value(reward, clip_value_min=0.0, clip_value_max=1.0)
    action, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.action, self._action_spec)

    partitioned_rewards = tf.dynamic_partition(
        reward, action, self._num_actions)
    for k in range(self._num_actions):
      tf.compat.v1.assign_add(
          self._alpha[k], tf.cast(
              tf.reduce_sum(partitioned_rewards[k]), dtype=self._dtype))
      tf.compat.v1.assign_add(
          self._beta[k], tf.cast(
              tf.reduce_sum(1.0 - partitioned_rewards[k]), dtype=self._dtype))

    self.train_step_counter.assign_add(self._batch_size)
    loss = -1. * tf.reduce_sum(reward)
    return tf_agent.LossInfo(loss=(loss), extra=())
