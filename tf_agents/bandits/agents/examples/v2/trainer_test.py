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

"""Tests for tf_agents.bandits.agents.examples.v2.trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tempfile

from absl.testing import parameterized
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import exp3_agent
from tf_agents.bandits.agents.examples.v2 import trainer
from tf_agents.bandits.agents.examples.v2 import trainer_test_utils
from tf_agents.bandits.environments import environment_utilities
from tf_agents.bandits.environments import random_bandit_environment
from tf_agents.bandits.environments import stationary_stochastic_py_environment
from tf_agents.bandits.environments import wheel_py_environment
from tf_agents.bandits.metrics import tf_metrics as tf_bandit_metrics
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec

tfd = tfp.distributions

tf.compat.v1.enable_v2_behavior()


def get_bounded_reward_random_environment(
    observation_shape, action_shape, batch_size, num_actions):
  """Returns a RandomBanditEnvironment with U(0, 1) observation and reward."""
  overall_shape = [batch_size] + observation_shape
  observation_distribution = tfd.Independent(
      tfd.Uniform(low=tf.zeros(overall_shape), high=tf.ones(overall_shape)))
  reward_distribution = tfd.Uniform(
      low=tf.zeros(batch_size), high=tf.ones(batch_size))
  action_spec = tensor_spec.BoundedTensorSpec(
      shape=action_shape, dtype=tf.int32, minimum=0, maximum=num_actions - 1)
  return random_bandit_environment.RandomBanditEnvironment(
      observation_distribution, reward_distribution, action_spec)


def get_environment_and_optimal_functions_by_name(environment_name, batch_size):
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


class TrainerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='_0',
           num_actions=11,
           observation_shape=[8],
           action_shape=[],
           batch_size=32,
           training_loops=10,
           steps_per_loop=10,
           learning_rate=.1),
      dict(testcase_name='_1',
           num_actions=73,
           observation_shape=[5, 4, 3, 2],
           action_shape=[],
           batch_size=121,
           training_loops=7,
           steps_per_loop=8,
           learning_rate=.5),
      )
  def testTrainerExportsCheckpoints(self,
                                    num_actions,
                                    observation_shape,
                                    action_shape,
                                    batch_size,
                                    training_loops,
                                    steps_per_loop,
                                    learning_rate):
    """Exercises trainer code, checks that expected checkpoints are exported."""
    root_dir = tempfile.mkdtemp(dir=os.getenv('TEST_TMPDIR'))
    environment = get_bounded_reward_random_environment(
        observation_shape, action_shape, batch_size, num_actions)
    agent = exp3_agent.Exp3Agent(
        learning_rate=learning_rate,
        time_step_spec=environment.time_step_spec(),
        action_spec=environment.action_spec())
    for i in range(1, 4):
      trainer.train(
          root_dir=root_dir,
          agent=agent,
          environment=environment,
          training_loops=training_loops,
          steps_per_loop=steps_per_loop)
      latest_checkpoint = tf.train.latest_checkpoint(root_dir)
      expected_checkpoint_regex = '.*-{}'.format(i * training_loops)
      self.assertRegex(latest_checkpoint, expected_checkpoint_regex)

  @parameterized.named_parameters(
      dict(testcase_name='_stat_stoch__linucb',
           environment_name='stationary_stochastic',
           agent_name='LinUCB'),
      dict(testcase_name='_stat_stoch__lints',
           environment_name='stationary_stochastic',
           agent_name='LinTS'),
      dict(testcase_name='_stat_stoch__epsgreedy',
           environment_name='stationary_stochastic',
           agent_name='epsGreedy'),
      dict(testcase_name='_wheel__linucb',
           environment_name='wheel',
           agent_name='LinUCB'),
      dict(testcase_name='_wheel__lints',
           environment_name='wheel',
           agent_name='LinTS'),
      dict(testcase_name='_wheel__epsgreedy',
           environment_name='wheel',
           agent_name='epsGreedy'),
      dict(testcase_name='_wheel__mix',
           environment_name='wheel',
           agent_name='mix'),
      )
  def testAgentAndEnvironmentRuns(self, environment_name, agent_name):
    batch_size = 8
    training_loops = 3
    steps_per_loop = 2
    (environment, optimal_reward_fn, optimal_action_fn
    ) = trainer_test_utils.get_environment_and_optimal_functions_by_name(
        environment_name, batch_size)

    agent = trainer_test_utils.get_agent_by_name(agent_name,
                                                 environment.time_step_spec(),
                                                 environment.action_spec())

    regret_metric = tf_bandit_metrics.RegretMetric(optimal_reward_fn)
    suboptimal_arms_metric = tf_bandit_metrics.SuboptimalArmsMetric(
        optimal_action_fn)
    trainer.train(
        root_dir=tempfile.mkdtemp(dir=os.getenv('TEST_TMPDIR')),
        agent=agent,
        environment=environment,
        training_loops=training_loops,
        steps_per_loop=steps_per_loop,
        additional_metrics=[regret_metric, suboptimal_arms_metric])


if __name__ == '__main__':
  tf.test.main()
