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

# Lint as: python2, python3
"""Benchmarks for DqnAgent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.agents.dqn import dqn_agent
from tf_agents.benchmark import distribution_strategy_utils
from tf_agents.benchmark import utils
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.utils import common

from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import  # TF internal


class DqnCartPoleAgentBenchmark(tf.test.Benchmark):
  """Short benchmarks (~110 steps) for DQN CartPole environment."""

  def _run(self,
           strategy,
           batch_size=64,
           tf_function=True,
           replay_buffer_max_length=1000,
           train_steps=110,
           log_steps=10):
    """Runs Dqn CartPole environment.

    Args:
      strategy: Strategy to use, None is a valid value.
      batch_size: Total batch size to use for the run.
      tf_function: If True tf.function is used.
      replay_buffer_max_length: Max length of the replay buffer.
      train_steps: Number of steps to run.
      log_steps: How often to log step statistics, e.g. step time.
    """
    obs_spec = array_spec.BoundedArraySpec([
        4,
    ], np.float32, -4., 4.)
    action_spec = array_spec.BoundedArraySpec((), np.int64, 0, 1)

    py_env = random_py_environment.RandomPyEnvironment(
        obs_spec,
        action_spec,
        batch_size=1,
        reward_fn=lambda *_: np.random.randint(1, 10, 1))
    env = tf_py_environment.TFPyEnvironment(py_env)

    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                             env.action_spec())

    with distribution_strategy_utils.strategy_scope_context(strategy):
      q_net = q_network.QNetwork(
          env.time_step_spec().observation,
          env.action_spec(),
          fc_layer_params=(100,))

      tf_agent = dqn_agent.DqnAgent(
          env.time_step_spec(),
          env.action_spec(),
          q_network=q_net,
          optimizer=tf.keras.optimizers.Adam(),
          td_errors_loss_fn=common.element_wise_squared_loss)
      tf_agent.initialize()
      print(q_net.summary())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_max_length)

    driver = dynamic_step_driver.DynamicStepDriver(env, policy,
                                                   [replay_buffer.add_batch])
    if tf_function:
      driver.run = common.function(driver.run)

    for _ in range(replay_buffer_max_length):
      driver.run()

    check_values = ['QNetwork/EncodingNetwork/dense/bias:0']
    initial_values = utils.get_initial_values(tf_agent, check_values)

    with distribution_strategy_utils.strategy_scope_context(strategy):
      dataset = replay_buffer.as_dataset(
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          sample_batch_size=batch_size,
          num_steps=2)
      if strategy:
        iterator = iter(strategy.experimental_distribute_dataset(dataset))
      else:
        iterator = iter(dataset)

      def train_step():
        experience, _ = next(iterator)
        return tf_agent.train(experience)

      if tf_function:
        train_step = common.function(train_step)
      self.run_and_report(
          train_step,
          strategy,
          batch_size,
          train_steps=train_steps,
          log_steps=log_steps)

    utils.check_values_changed(tf_agent, initial_values, check_values)

  def run_and_report(self,
                     train_step,
                     strategy,
                     batch_size,
                     train_steps=110,
                     log_steps=10):
    """Run function provided and report results per `tf.test.Benchmark`.

    Args:
      train_step: Function to execute on each step.
      strategy: Strategy to use, None is a valid value.
      batch_size: Total batch_size.
      train_steps: Number of steps to run.
      log_steps: How often to log step statistics, e.g. step time.

    Returns:
      `TimeHistory` object with statistics about the throughput perforamnce.
    """
    history = utils.run_test(
        train_step,
        train_steps,
        strategy,
        batch_size=batch_size,
        log_steps=log_steps)
    print('Avg step time:{}'.format(history.get_average_step_time()))
    print('Avg exp/sec:{}'.format(history.get_average_examples_per_second()))
    metrics = []
    metrics.append({
        'name': 'exp_per_second',
        'value': history.get_average_examples_per_second()
    })
    metrics.append({
        'name': 'steps_per_second',
        'value': 1 / history.get_average_step_time()
    })
    metrics.append({
        'name': 'step_time',
        'value': history.get_average_step_time()
    })
    self.report_benchmark(
        iters=-1, wall_time=history.get_average_step_time(), metrics=metrics)
    return history

  def benchmark_dqn_cpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=0)
    self._run(strategy)

  def benchmark_dqn_mirrored_cpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='mirrored', num_gpus=0)
    self._run(strategy)

  def benchmark_dqn_eagerly_cpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=0)
    self._run(strategy, tf_function=False)

  def benchmark_dqn_no_dist_strat_cpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=0)
    self._run(strategy)

  def benchmark_dqn_no_dist_strat_eagerly_cpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=0)
    self._run(strategy, tf_function=False)

  def benchmark_dqn_no_dist_strat_1_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=1)
    self._run(strategy)

  def benchmark_dqn_no_dist_strat_eagerly_1_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=1)
    self._run(strategy, tf_function=False)

  def benchmark_dqn_no_dist_strat_1_gpu_xla(self):
    utils.set_session_config(enable_xla=True)
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=1)
    self._run(strategy)

  def benchmark_dqn_1_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=1)
    self._run(strategy)

  def benchmark_dqn_2_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=2)
    self._run(strategy, batch_size=64 * 2)

  def benchmark_dqn_8_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=8)
    self._run(strategy, batch_size=64 * 8)

  def benchmark_dqn_mirrored_1_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='mirrored', num_gpus=1)
    self._run(strategy)

  def benchmark_dqn_eagerly_1_gpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=1)
    self._run(strategy, tf_function=False)

  def benchmark_dqn_1_gpu_xla(self):
    utils.set_session_config(enable_xla=True)
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=1)
    self._run(strategy)

  def benchmark_dqn_2_gpu_xla(self):
    utils.set_session_config(enable_xla=True)
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=2)
    self._run(strategy, batch_size=64 * 2)

  def benchmark_dqn_8_gpu_xla(self):
    utils.set_session_config(enable_xla=True)
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=8)
    self._run(strategy, batch_size=64 * 8)


class DqnCartPoleAgentBenchmarkTest(tf.test.TestCase):
  """Tests for DqnCartPoleAgentBenchmark."""

  def _run(self, strategy, tf_function=True):

    benchmark = DqnCartPoleAgentBenchmark()
    benchmark._run(
        strategy,
        tf_function=tf_function,
        replay_buffer_max_length=5,
        train_steps=2,
        log_steps=1)

  @unittest.skipUnless(tf2.enabled(), 'TF 2.x only test.')
  def testCpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=0)
    self._run(strategy)

  @unittest.skipUnless(tf2.enabled(), 'TF 2.x only test.')
  def testEagerCpu(self):
    print('TF 2.0 enable:{}'.format(tf2.enabled()))
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='default', num_gpus=0)
    self._run(strategy, tf_function=False)

  @unittest.skipUnless(tf2.enabled(), 'TF 2.x only test.')
  def testNoStrategyCpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=0)
    self._run(strategy)

  @unittest.skipUnless(tf2.enabled(), 'TF 2.x only test.')
  def testNoStrategyEagerCpu(self):
    strategy = distribution_strategy_utils.get_distribution_strategy(
        distribution_strategy='off', num_gpus=0)
    self._run(strategy, tf_function=False)


if __name__ == '__main__':
  tf.test.main()
