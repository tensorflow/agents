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

# Lint as: python2, python3
"""Unit tests for the Learner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
from absl.testing import parameterized

import numpy as np
from six.moves import range
import tensorflow.compat.v2 as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.experimental.train import learner
from tf_agents.experimental.train.utils import test_utils as dist_test_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import test_utils


class LearnerTest(test_utils.TestCase, parameterized.TestCase):
  """A set of tests for the Learner."""

  def setUp(self):
    super(LearnerTest, self).setUp()

    # Define 4 logical CPUs on the first physical one.
    dist_test_utils.configure_logical_cpus()

    devices_gpu = tf.config.list_physical_devices('GPU')
    # If there are GPU devices:
    if devices_gpu:
      tf.config.experimental.set_virtual_device_configuration(
          devices_gpu[0], [
              tf.config.LogicalDeviceConfiguration(memory_limit=10),
              tf.config.LogicalDeviceConfiguration(memory_limit=10),
              tf.config.LogicalDeviceConfiguration(memory_limit=10),
              tf.config.LogicalDeviceConfiguration(memory_limit=10)
          ])

  def _build_learner_with_strategy(self,
                                   create_agent_and_dataset_fn,
                                   strategy,
                                   sample_batch_size=2):
    if strategy is None:
      # Get default strategy if None provided.
      strategy = tf.distribute.get_strategy()

    with strategy.scope():
      tf_env = tf_py_environment.TFPyEnvironment(
          suite_gym.load('CartPole-v0'))

      train_step = train_utils.create_train_step()
      agent, dataset, dataset_fn, _ = create_agent_and_dataset_fn(
          tf_env.action_spec(), tf_env.time_step_spec(), train_step,
          sample_batch_size)

      root_dir = os.path.join(self.create_tempdir().full_path, 'learner')

      test_learner = learner.Learner(
          root_dir=root_dir,
          train_step=train_step,
          agent=agent,
          experience_dataset_fn=dataset_fn)
      variables = agent.collect_policy.variables()
    return test_learner, dataset, variables, train_step

  def _compare_losses(self, loss_1, loss_2, delta=1.e-2):
    (digits_1, exponent_1) = np.frexp(loss_1)
    (digits_2, exponent_2) = np.frexp(loss_2)
    self.assertEqual(exponent_1, exponent_2)
    self.assertAlmostEqual(digits_1, digits_2, delta=delta)

  def testLearnerRun(self):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    test_learner, _, variables, _ = (
        self._build_learner_with_strategy(
            dist_test_utils.create_dqn_agent_and_dataset_fn,
            strategy,
            sample_batch_size=4))
    old_vars = self.evaluate(variables)
    loss = test_learner.run().loss
    new_vars = self.evaluate(variables)

    dist_test_utils.check_variables_different(
        self, old_vars, new_vars)
    self.assertAllInRange(loss, tf.float32.min, tf.float32.max)

  def testLearnerAssertInvalidIterations(self):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    test_learner, _, _, _ = (
        self._build_learner_with_strategy(
            dist_test_utils.create_dqn_agent_and_dataset_fn,
            strategy,
            sample_batch_size=4))
    with self.assertRaisesRegex(
        AssertionError, 'Iterations must be greater or equal to 1'):
      test_learner.run(iterations=0)

  def _get_tpu_strategy(self):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.experimental.TPUStrategy(resolver)

  @parameterized.named_parameters(
      ('DQN', dist_test_utils.create_dqn_agent_and_dataset_fn),
      ('PPO', dist_test_utils.create_ppo_agent_and_dataset_fn),
  )
  def testLossLearnerDifferentDistStrat(self, create_agent_fn):
    # Create the strategies used in the test. The second value is the per-core
    # batch size.
    bs_multiplier = 4
    strategies = {
        'default': (tf.distribute.get_strategy(), 4 * bs_multiplier),
        'one_device':
            (tf.distribute.OneDeviceStrategy('/cpu:0'), 4 * bs_multiplier),
        'mirrored': (tf.distribute.MirroredStrategy(), 1 * bs_multiplier),
    }
    if tf.config.list_logical_devices('TPU'):
      strategies['TPU'] = (self._get_tpu_strategy(), 2 * bs_multiplier)
    else:
      logging.info('TPU hardware is not available, TPU strategy test skipped.')

    learners = {
        name: self._build_learner_with_strategy(create_agent_fn, strategy,
                                                per_core_batch_size)
        for name, (strategy, per_core_batch_size) in strategies.items()
    }

    # Verify that the initial variable values in the learners are the same.
    default_strat_trainer, _, default_vars, _ = learners['default']
    for name, (trainer, _, variables, _) in learners.items():
      if name != 'default':
        self._assign_variables(default_strat_trainer, trainer)
        self.assertLen(variables, len(default_vars))
        for default_variable, variable in zip(default_vars, variables):
          self.assertAllEqual(default_variable, variable)

    # Calculate losses.
    losses = {}
    iterations = 1
    for name, (trainer, _, variables, train_step) in learners.items():
      old_vars = self.evaluate(variables)

      loss = trainer.run(iterations=iterations).loss
      logging.info('Using strategy: %s, the loss is: %s at train step: %s',
                   name, loss, train_step)

      new_vars = self.evaluate(variables)
      losses[name] = old_vars, loss, new_vars

    # Verify same dataset across learner calls.
    for item in tf.data.Dataset.zip(tuple([v[1] for v in learners.values()])):
      for i in range(1, len(item)):
        # Compare default strategy obervation to the other datasets, second
        # index is getting the trajectory from (trajectory, sample_info) tuple.
        self.assertAllEqual(item[0][0].observation, item[i][0].observation)

    # Check that the losses are close to each other.
    _, default_loss, _ = losses['default']
    for name, (_, loss, _) in losses.items():
      self._compare_losses(loss, default_loss, delta=1.e-2)

    # Check that the variables changed after calling `learner.run`.
    for old_vars, _, new_vars in losses.values():
      dist_test_utils.check_variables_different(self, old_vars, new_vars)

  def _assign_variables(self, src_learner, dst_learner):
    src_vars = src_learner._agent.collect_policy.variables()
    dst_vars = dst_learner._agent.collect_policy.variables()
    self.assertEqual(len(src_vars), len(dst_vars))
    for src_var, dst_var in zip(src_vars, dst_vars):
      dst_var.assign(src_var)


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
