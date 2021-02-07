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

"""A mixture agent that updates the mixture distribution based on EXP3.

For a reference on EXP3, see `Bandit Algorithms` by Tor Lattimore and Csaba
Szepesvari (https://tor-lattimore.com/downloads/book/book.pdf).
"""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import List, Optional, Text

import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents import tf_agent
from tf_agents.bandits.agents import mixture_agent
from tf_agents.bandits.policies import mixture_policy
from tf_agents.utils import nest_utils

tfd = tfp.distributions


@gin.configurable
class Exp3MixtureVariableCollection(tf.Module):
  """A collection of variables used by subclasses of `MixtureAgent`.

  Note that this variable collection only contains the mixture weights. The
  variables of the sub-agents that the mixture agent mixes are in variable
  collections of the respective sub-agents.
  """

  def __init__(self,
               num_agents: int,
               reward_aggregates: Optional[List[float]] = None,
               inverse_temperature: float = 0.0):
    """Initializes an instace of 'Exp3MixtureVariableCollection'.

    Args:
      num_agents: (int) the number of agents mixed by the mixture agent.
      reward_aggregates: A list of floats containing the reward aggregates for
        each agent. If not set, the initial values will be 0.
      inverse_temperature: The initial value for the inverse temperature
        variable used by the mixture agent.
    """
    if reward_aggregates is None:
      reward_aggregates = [0.0] * num_agents
    else:
      if num_agents != len(reward_aggregates):
        raise ValueError('`reward_aggregates` must have `num_agents` elements.')
    self._reward_aggregates = tf.Variable(
        reward_aggregates, name='reward_aggregates', dtype=tf.float32)
    self._inverse_temperature = tf.Variable(
        inverse_temperature, dtype=tf.float32)

  @property
  def reward_aggregates(self):
    return self._reward_aggregates

  @property
  def inverse_temperature(self):
    return self._inverse_temperature


@gin.configurable
class Exp3MixtureAgent(mixture_agent.MixtureAgent):
  """An agent that mixes a set of agents and updates the weights with Exp3.

  For a reference on EXP3, see `Bandit Algorithms` by Tor Lattimore and Csaba
  Szepesvari (https://tor-lattimore.com/downloads/book/book.pdf).

  The update uses a slighlty modified version of EXP3 to make sure that the
  weights do not go to one seemingly good agent in the very beginning. To smooth
  the weights, two extra measures are taken:

  1. A forgetting factor makes sure that the aggregated reward estimates do not
  grow indefinitely.
  2. The `inverse temperature` has a maximum parameter that prevents it from
  growing indefinitely.

  It is generally a good idea to set

  ```
  forgetting_factor = 1 - (1 / max_inverse_temperature)
  ```

  so that the two smoothing factors work together nicely.

  For every data sample, the agent updates the sub-agent that was used to make
  the action choice in that sample. For this update to happen, the mixture agent
  needs to have the information on which sub-agent is "responsible" for the
  action. This information is in a policy info field `mixture_choice_info`.
  """

  def __init__(
      self,
      agents: List[tf_agent.TFAgent],
      variable_collection: Optional[Exp3MixtureVariableCollection] = None,
      forgetting: float = 0.999,
      max_inverse_temperature: float = 1000.0,
      name: Optional[Text] = None):
    """Initializes an instance of `Exp3MixtureAgent`.

    Args:
      agents: List of TF-Agents agents that this mixture agent trains.
      variable_collection: An instance of `Exp3VariableCollection`. If not set,
        A default one will be created. It contains all the variables that are
        needed to restore the mixture agent, excluding the variables of the
        subagents.
      forgetting: A float value in (0, 1]. This is how much the estimated
        reward aggregates are shrinked in every training step.
      max_inverse_temperature: This value caps the inverse temperature that
       would otherwise grow as the square root of the number of samples seen.
      name: Name fo this instance of `Exp3MixtureAgent`.
    """
    self._num_agents = len(agents)
    self._forgetting = forgetting
    self._max_inverse_temperature = max_inverse_temperature
    if variable_collection is None:
      variable_collection = Exp3MixtureVariableCollection(
          self._num_agents)
    elif not isinstance(variable_collection,
                        Exp3MixtureVariableCollection):
      raise TypeError('Parameter `variable_collection` should be '
                      'of type `MixtureVariableCollection`.')
    elif variable_collection.reward_aggregates.shape != self._num_agents:
      raise ValueError('`variable_collection.reward_aggregates` should have '
                       'shape `[len(agents)]`.')
    self._variable_collection = variable_collection

    # The `_mixture_weights` value is reassigned in every training step and only
    # depends on reward aggregates and inverse temperature. This variable is not
    # part of the variable collection because it is not needed to restore an
    # agent. The only reason why this value is a tf.Variable is because this way
    # the categorical distribution is dynamically parameterized.

    self._mixture_weights = tf.Variable(
        tf.zeros_like(variable_collection.reward_aggregates))
    mixture_distribution = tfd.Categorical(
        logits=self._mixture_weights)
    super(Exp3MixtureAgent, self).__init__(
        mixture_distribution, agents, name=name)

  def _update_mixture_distribution(self, experience):

    reward, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.reward, self._time_step_spec.reward)
    policy_choice, _ = nest_utils.flatten_multi_batched_nested_tensors(
        experience.policy_info[mixture_policy.MIXTURE_AGENT_ID],
        self._time_step_spec.reward)
    batch_size = tf.compat.dimension_value(
        reward.shape[0]) or tf.shape(reward)[0]
    unnormalized_probabilities = tf.exp(self._mixture_weights)
    probabilities = unnormalized_probabilities / tf.norm(
        unnormalized_probabilities, 1)

    normalizer = tf.reduce_sum(unnormalized_probabilities)
    probabilities = unnormalized_probabilities / normalizer
    self._summarize_probabilities(probabilities)
    repeated_probs = tf.tile(
        tf.expand_dims(probabilities, axis=0), [batch_size, 1])
    probs_per_step = tf.gather(
        repeated_probs, policy_choice, batch_dims=1)
    per_step_update_term = tf.expand_dims((1 - reward) / probs_per_step, axis=0)
    one_hot_policy_choice = tf.one_hot(
        policy_choice, depth=self._num_agents)
    update_term = 1 - tf.squeeze(
        tf.matmul(per_step_update_term, one_hot_policy_choice))
    self._update_aggregates(update_term)
    self._update_inverse_temperature(batch_size)
    return self._mixture_weights.assign(
        self._variable_collection.reward_aggregates /
        self._variable_collection.inverse_temperature)

  def _summarize_probabilities(self, probabilities):
    for k in range(self._num_agents):
      tf.compat.v2.summary.scalar(
          name='policy_{}_prob'.format(k),
          data=probabilities[k],
          step=self.train_step_counter)

  def _update_aggregates(self, update_term):
    self._variable_collection.reward_aggregates.assign(
        self._forgetting *
        (self._variable_collection.reward_aggregates + update_term))

  def _update_inverse_temperature(self, batch_size):
    self._variable_collection.inverse_temperature.assign(
        tf.maximum(
            self._max_inverse_temperature,
            tf.sqrt(
                tf.square(self._variable_collection.inverse_temperature) +
                tf.cast(batch_size, dtype=tf.float32))))
