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

"""Tests for tf_agents.bandits.agents.examples.v2.trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import exp3_mixture_agent
from tf_agents.bandits.agents import lin_ucb_agent
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import stationary_stochastic_py_environment
from tf_agents.bandits.environments import wheel_py_environment
from tf_agents.bandits.policies import policy_utilities
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network

tfd = tfp.distributions

tf.compat.v1.enable_v2_behavior()


def get_environment_and_optimal_functions_by_name(environment_name, batch_size):
  """Helper function that outputs an environment and related functions.

  Args:
    environment_name: The (string) name of the desired environment.
    batch_size: The batch_size

  Returns:
    A tuple of (environment, optimal_reward_fn, optimal_action_fn), where the
    latter two functions are for calculating regret and the suboptimal actions
    metrics.
  """
  if environment_name == 'stationary_stochastic':
    context_dim = 7
    num_actions = 5
    action_reward_fns = (
        environment_utilities.sliding_linear_reward_fn_generator(
            context_dim, num_actions, 0.1))
    py_env = (
        stationary_stochastic_py_environment
        .StationaryStochasticPyEnvironment(
            functools.partial(
                environment_utilities.context_sampling_fn,
                batch_size=batch_size,
                context_dim=context_dim),
            action_reward_fns,
            batch_size=batch_size))
    optimal_reward_fn = functools.partial(
        environment_utilities.tf_compute_optimal_reward,
        per_action_reward_fns=action_reward_fns)

    optimal_action_fn = functools.partial(
        environment_utilities.tf_compute_optimal_action,
        per_action_reward_fns=action_reward_fns)
    environment = tf_py_environment.TFPyEnvironment(py_env)
  elif environment_name == 'wheel':
    delta = 0.5
    mu_base = [0.05, 0.01, 0.011, 0.009, 0.012]
    std_base = [0.001] * 5
    mu_high = 0.5
    std_high = 0.001
    py_env = wheel_py_environment.WheelPyEnvironment(delta, mu_base, std_base,
                                                     mu_high, std_high,
                                                     batch_size)
    environment = tf_py_environment.TFPyEnvironment(py_env)
    optimal_reward_fn = functools.partial(
        environment_utilities.tf_wheel_bandit_compute_optimal_reward,
        delta=delta,
        mu_inside=mu_base[0],
        mu_high=mu_high)
    optimal_action_fn = functools.partial(
        environment_utilities.tf_wheel_bandit_compute_optimal_action,
        delta=delta)
  return (environment, optimal_reward_fn, optimal_action_fn)


def get_agent_by_name(agent_name, time_step_spec, action_spec):
  """Helper function that outputs an agent.

  Args:
    agent_name: The name (string) of the desired agent.
    time_step_spec: The time step spec of the environment on which the agent
      acts.
    action_spec: The action spec on which the agent acts.

  Returns:
    The desired agent.
  """
  if agent_name == 'LinUCB':
    return lin_ucb_agent.LinearUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=tf.float32)
  elif agent_name == 'LinTS':
    return lin_ts_agent.LinearThompsonSamplingAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        dtype=tf.float32)
  elif agent_name == 'epsGreedy':
    network = q_network.QNetwork(
        input_tensor_spec=time_step_spec.observation,
        action_spec=action_spec,
        fc_layer_params=(50, 50, 50))
    return neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        reward_network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        epsilon=0.1)
  elif agent_name == 'mix':
    emit_policy_info = (policy_utilities.InfoFields.PREDICTED_REWARDS_MEAN,)
    network = q_network.QNetwork(
        input_tensor_spec=time_step_spec.observation,
        action_spec=action_spec,
        fc_layer_params=(50, 50, 50))
    agent_epsgreedy = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        reward_network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        emit_policy_info=emit_policy_info,
        epsilon=0.1)
    agent_linucb = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        emit_policy_info=emit_policy_info,
        dtype=tf.float32)
    agent_random = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        reward_network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        emit_policy_info=emit_policy_info,
        epsilon=1.)
    agent_halfrandom = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        reward_network=network,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.05),
        emit_policy_info=emit_policy_info,
        epsilon=0.5)
    return exp3_mixture_agent.Exp3MixtureAgent(
        (agent_epsgreedy, agent_linucb, agent_random, agent_halfrandom))


