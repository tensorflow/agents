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
"""Unit tests for the Learner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from absl import logging
from absl.testing import parameterized

import numpy as np
from six.moves import range
import tensorflow as tf

from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner
from tf_agents.train.utils import test_utils as dist_test_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import test_utils

_DQN_AGENT_FN = dist_test_utils.create_dqn_agent_and_dataset_fn
_PPO_AGENT_FN = dist_test_utils.create_ppo_agent_and_dataset_fn
_DEFAULT_STRATEGY_FN = tf.distribute.get_strategy
_ONE_DEVICE_STRATEGY_FN = functools.partial(tf.distribute.OneDeviceStrategy,
                                            '/cpu:0')
_MIRRORED_STRATEGY_FN = tf.distribute.MirroredStrategy


def _get_tpu_strategy():
  if not tf.config.list_logical_devices('TPU'):
    logging.info('TPU hardware is not available, TPU strategy test skipped.')
    return None

  resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return tf.distribute.experimental.TPUStrategy(resolver)


_TPU_STRATEGY_FN = _get_tpu_strategy


def _copy_recursive(src_dir, dst_dir):
  for subdir, _, filenames in tf.io.gfile.walk(src_dir):
    out_dir = os.path.join(dst_dir, os.path.relpath(subdir, start=src_dir))
    tf.io.gfile.makedirs(out_dir)
    for filename in filenames:
      tf.io.gfile.copy(
          os.path.join(subdir, filename), os.path.join(out_dir, filename))


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
                                   sample_batch_size=2,
                                   root_dir=None):
    if strategy is None:
      # Get default strategy if None provided.
      strategy = tf.distribute.get_strategy()

    with strategy.scope():
      tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))

      train_step = train_utils.create_train_step()
      agent, dataset, dataset_fn, _ = create_agent_and_dataset_fn(
          tf_env.action_spec(), tf_env.time_step_spec(), train_step,
          sample_batch_size)

      if root_dir is None:
        root_dir = os.path.join(self.create_tempdir().full_path, 'learner')

      test_learner = learner.Learner(
          root_dir=root_dir,
          train_step=train_step,
          agent=agent,
          experience_dataset_fn=dataset_fn,
          checkpoint_interval=1)
      variables = agent.collect_policy.variables()
    return test_learner, dataset, variables, train_step, dataset_fn

  def _compare_losses(self, loss_1, loss_2, delta=1.e-2):
    (digits_1, exponent_1) = np.frexp(loss_1)
    (digits_2, exponent_2) = np.frexp(loss_2)
    self.assertEqual(exponent_1, exponent_2)
    self.assertAlmostEqual(digits_1, digits_2, delta=delta)

  def testLearnerRun(self):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    test_learner, _, variables, _, _ = (
        self._build_learner_with_strategy(
            dist_test_utils.create_dqn_agent_and_dataset_fn,
            strategy,
            sample_batch_size=4))
    old_vars = self.evaluate(variables)
    loss = test_learner.run().loss
    new_vars = self.evaluate(variables)

    dist_test_utils.check_variables_different(self, old_vars, new_vars)
    self.assertAllInRange(loss, tf.float32.min, tf.float32.max)

  def testLearnerLoss(self):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    test_learner, _, variables, _, _ = (
        self._build_learner_with_strategy(
            dist_test_utils.create_dqn_agent_and_dataset_fn,
            strategy,
            sample_batch_size=4))
    old_vars = self.evaluate(variables)

    # Compute loss using the default sum reduce op.
    loss_sum = test_learner.loss().loss
    new_vars = self.evaluate(variables)

    dist_test_utils.check_variables_same(self, old_vars, new_vars)
    self.assertAllInRange(loss_sum, tf.float32.min, tf.float32.max)

    # Compute loss using a mean reduce op.
    loss_mean = test_learner.loss(reduce_op=tf.distribute.ReduceOp.MEAN).loss
    new_vars = self.evaluate(variables)

    dist_test_utils.check_variables_same(self, old_vars, new_vars)
    self.assertAllInRange(loss_mean, tf.float32.min, tf.float32.max)

  def testLearnerLossPassExperience(self):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    test_learner, _, variables, _, dataset_fn = (
        self._build_learner_with_strategy(
            dist_test_utils.create_dqn_agent_and_dataset_fn,
            strategy,
            sample_batch_size=4))
    old_vars = self.evaluate(variables)

    dataset_iter = iter(dataset_fn())
    loss_sum = test_learner.loss(
        experience_and_sample_info=next(dataset_iter)).loss
    new_vars = self.evaluate(variables)

    dist_test_utils.check_variables_same(self, old_vars, new_vars)
    self.assertAllInRange(loss_sum, tf.float32.min, tf.float32.max)

  def testLearnerAssertInvalidIterations(self):
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    test_learner, _, _, _, _ = (
        self._build_learner_with_strategy(
            dist_test_utils.create_dqn_agent_and_dataset_fn,
            strategy,
            sample_batch_size=4))
    with self.assertRaisesRegex(AssertionError,
                                'Iterations must be greater or equal to 1'):
      test_learner.run(iterations=0)

  def testLearnerRaiseExceptionOnMismatchingBatchSetup(self):
    obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    time_step_spec = ts.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec([], tf.int32, 0, 1)
    flat_action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = flat_action_spec.maximum - flat_action_spec.minimum + 1

    network = sequential.Sequential([
        tf.keras.layers.Dense(num_actions, dtype=tf.float32),
        inner_reshape.InnerReshape([None], [num_actions])
    ])

    agent = behavioral_cloning_agent.BehavioralCloningAgent(
        time_step_spec, action_spec, cloning_network=network, optimizer=None)

    with self.assertRaisesRegex(
        ValueError,
        'All of the Tensors in `value` must have one outer dimension.'):
      learner.Learner(
          root_dir=os.path.join(self.create_tempdir().full_path, 'learner'),
          train_step=train_utils.create_train_step(),
          agent=agent)

  @parameterized.named_parameters(
      ('DQN_default', _DQN_AGENT_FN, _DEFAULT_STRATEGY_FN, 16),
      ('DQN_one_device', _DQN_AGENT_FN, _ONE_DEVICE_STRATEGY_FN, 16),
      ('DQN_mirrored', _DQN_AGENT_FN, _MIRRORED_STRATEGY_FN, 4),
      ('DQN_TPU', _DQN_AGENT_FN, _TPU_STRATEGY_FN, 8),
      ('PPO_default', _PPO_AGENT_FN, _DEFAULT_STRATEGY_FN, 16),
      ('PPO_one_device', _PPO_AGENT_FN, _ONE_DEVICE_STRATEGY_FN, 16),
      ('PPO_mirrored', _PPO_AGENT_FN, _MIRRORED_STRATEGY_FN, 4),
      ('PPO_TPU', _PPO_AGENT_FN, _TPU_STRATEGY_FN, 8),
  )
  def testInitialCheckpointStoresOptimizerVariables(self, create_agent_fn,
                                                    strategy_fn,
                                                    per_core_batch_size):
    # Create strategy. Skip the test if the strategy is unsupported.
    strategy = strategy_fn()
    if strategy is None:
      logging.info(('The strategy creted using the function %s is not '
                    'supported. Skipping the test.'), strategy_fn)

    # Create learner.
    base_dir = self.create_tempdir().full_path
    root_dir = os.path.join(base_dir, 'root_dir')
    trained_learner = self._build_learner_with_strategy(
        create_agent_fn, strategy, per_core_batch_size, root_dir=root_dir)[0]

    # Force saving initial checkpoint.
    trained_learner._checkpointer.save(trained_learner.train_step)
    self.assertLen(trained_learner._checkpointer.manager.checkpoints, 1)

    # Make a copy of the initial root directory.
    initial_root_dir = os.path.join(base_dir, 'initial_root_dir')
    _copy_recursive(root_dir, initial_root_dir)

    # Run one step of training which creates another checkpoint.
    trained_learner.run(iterations=1)
    self.assertLen(trained_learner._checkpointer.manager.checkpoints, 2)
    trained_learner = self._build_learner_with_strategy(
        create_agent_fn, strategy, per_core_batch_size, root_dir=root_dir)[0]
    self.assertLen(trained_learner._checkpointer.manager.checkpoints, 2)

    # Create another learner with the initial checkpoint.
    initial_learner_from_checkpoint = self._build_learner_with_strategy(
        create_agent_fn,
        strategy,
        per_core_batch_size,
        root_dir=initial_root_dir)[0]
    self.assertLen(
        initial_learner_from_checkpoint._checkpointer.manager.checkpoints, 1)

    # Both of the learners should contain the optimizer variable values which
    # should be different.
    initial_optimizer_vars = (
        initial_learner_from_checkpoint._agent._optimizer.variables())
    self.assertNotEmpty(initial_optimizer_vars)
    self.assertEqual(initial_optimizer_vars[0].numpy(), 0)  # Adam/iter:0

    trained_optimizer_vars = trained_learner._agent._optimizer.variables()
    self.assertNotEmpty(trained_optimizer_vars)
    self.assertEqual(trained_optimizer_vars[0].numpy(), 1)  # Adam/iter:0

    # Not all of the values should be close to each other.
    num_asserts = 0
    for initial, after_train_step in zip(initial_optimizer_vars,
                                         trained_optimizer_vars):
      name_parts = initial.name.split('/')
      if ('bias' in name_parts or 'EncodingNetwork' in name_parts or
          'NormalProjectionNetwork' in name_parts):
        # These always seem to be zero. Let's skip them.
        continue
      self.assertNotAllClose(
          initial.numpy(),
          after_train_step.numpy(),
          atol=1.e-12,
          rtol=1.e-12,
          msg='The {} and {} are close at each element: {}.'.format(
              initial.name, after_train_step.name, initial))
      num_asserts += 1

    # The value of `Adam/iter:0` is already compared above. There always should
    # be more assertions associated with this test.
    self.assertGreater(num_asserts, 1)

  @parameterized.named_parameters(
      ('DQN_default', _DQN_AGENT_FN, _DEFAULT_STRATEGY_FN, 16),
      ('DQN_one_device', _DQN_AGENT_FN, _ONE_DEVICE_STRATEGY_FN, 16),
      ('DQN_mirrored', _DQN_AGENT_FN, _MIRRORED_STRATEGY_FN, 4),
      ('DQN_TPU', _DQN_AGENT_FN, _TPU_STRATEGY_FN, 8),
      ('PPO_default', _PPO_AGENT_FN, _DEFAULT_STRATEGY_FN, 16),
      ('PPO_one_device', _PPO_AGENT_FN, _ONE_DEVICE_STRATEGY_FN, 16),
      ('PPO_mirrored', _PPO_AGENT_FN, _MIRRORED_STRATEGY_FN, 4),
      ('PPO_TPU', _PPO_AGENT_FN, _TPU_STRATEGY_FN, 8),
  )
  def testCheckpointStoresOptimizerVariables(self, create_agent_fn, strategy_fn,
                                             per_core_batch_size):
    # Create strategy. Skip the test if the strategy is unsupported.
    strategy = strategy_fn()
    if strategy is None:
      logging.info(('The strategy creted using the function %s is not '
                    'supported. Skipping the test.'), strategy_fn)

    # Create learner.
    base_dir = self.create_tempdir().full_path
    root_dir = os.path.join(base_dir, 'root_dir')
    current_step_learner = self._build_learner_with_strategy(
        create_agent_fn, strategy, per_core_batch_size, root_dir=root_dir)[0]

    # Run one step of training to create a checkpoint.
    current_step_learner.run(iterations=1)
    self.assertLen(current_step_learner._checkpointer.manager.checkpoints, 1)

    # Make a copy of the root directory.
    prev_step_root_dir = os.path.join(base_dir, 'prev_step_root_dir')
    _copy_recursive(root_dir, prev_step_root_dir)

    # Run one more step of training which creates another checkpoint.
    current_step_learner.run(iterations=1)
    self.assertLen(current_step_learner._checkpointer.manager.checkpoints, 2)
    current_step_learner = self._build_learner_with_strategy(
        create_agent_fn, strategy, per_core_batch_size, root_dir=root_dir)[0]
    self.assertLen(current_step_learner._checkpointer.manager.checkpoints, 2)

    # Create another learner with the initial checkpoint.
    prev_step_learner_from_checkpoint = self._build_learner_with_strategy(
        create_agent_fn,
        strategy,
        per_core_batch_size,
        root_dir=prev_step_root_dir)[0]
    self.assertLen(
        prev_step_learner_from_checkpoint._checkpointer.manager.checkpoints, 1)

    # Both of the learners should contain the optimizer variable values which
    # should be different.
    prev_step_optimizer_vars = (
        prev_step_learner_from_checkpoint._agent._optimizer.variables())
    self.assertNotEmpty(prev_step_optimizer_vars)
    self.assertEqual(prev_step_optimizer_vars[0].numpy(), 1)  # Adam/iter:0

    current_step_optimizer_vars = (
        current_step_learner._agent._optimizer.variables())
    self.assertNotEmpty(current_step_optimizer_vars)
    self.assertEqual(current_step_optimizer_vars[0].numpy(), 2)  # Adam/iter:0

    # Not all of the values should be close to each other.
    num_asserts = 0
    for prev_step_var, current_step_var in zip(prev_step_optimizer_vars,
                                               current_step_optimizer_vars):
      name_parts = prev_step_var.name.split('/')
      if ('bias' in name_parts or 'EncodingNetwork' in name_parts or
          'NormalProjectionNetwork' in name_parts):
        # These always seem to be zero. Let's skip them.
        continue
      self.assertNotAllClose(
          prev_step_var.numpy(),
          current_step_var.numpy(),
          atol=1.e-12,
          rtol=1.e-12,
          msg='The {} and {} are close at each element: {}.'.format(
              prev_step_var.name, current_step_var.name, prev_step_var))
      num_asserts += 1

    # The value of `Adam/iter:0` is already compared above. There always should
    # be more assertions associated with this test.
    self.assertGreater(num_asserts, 1)

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
      strategies['TPU'] = (_get_tpu_strategy(), 2 * bs_multiplier)
    else:
      logging.info('TPU hardware is not available, TPU strategy test skipped.')

    learners = {
        name: self._build_learner_with_strategy(create_agent_fn, strategy,
                                                per_core_batch_size)
        for name, (strategy, per_core_batch_size) in strategies.items()
    }

    # Verify that the initial variable values in the learners are the same.
    default_strat_trainer, _, default_vars, _, _ = learners['default']
    for name, (trainer, _, variables, _, _) in learners.items():
      if name != 'default':
        self._assign_variables(default_strat_trainer, trainer)
        self.assertLen(variables, len(default_vars))
        for default_variable, variable in zip(default_vars, variables):
          self.assertAllEqual(default_variable, variable)

    # Calculate losses.
    losses = {}
    checkpoint_path = {}
    iterations = 1
    optimizer_variables = {}
    for name, (trainer, _, variables, train_step, _) in learners.items():
      old_vars = self.evaluate(variables)

      loss = trainer.run(iterations=iterations).loss
      logging.info('Using strategy: %s, the loss is: %s at train step: %s',
                   name, loss, train_step)

      new_vars = self.evaluate(variables)
      losses[name] = old_vars, loss, new_vars
      self.assertNotEmpty(trainer._agent._optimizer.variables())
      optimizer_variables[name] = trainer._agent._optimizer.variables()
      checkpoint_path[name] = trainer._checkpointer.manager.directory

    for name, path in checkpoint_path.items():
      logging.info('Checkpoint dir for learner %s: %s. Content: %s', name, path,
                   tf.io.gfile.listdir(path))
      checkpointer = common.Checkpointer(path)

      # Make sure that the checkpoint file exists, so the learner initialized
      # using the corresponding root directory will pick up the values in the
      # checkpoint file.
      self.assertTrue(checkpointer.checkpoint_exists)

      # Create a learner using an existing root directory containing the
      # checkpoint files.
      strategy, per_core_batch_size = strategies[name]
      learner_from_checkpoint = self._build_learner_with_strategy(
          create_agent_fn,
          strategy,
          per_core_batch_size,
          root_dir=os.path.join(path, '..', '..'))[0]

      # Check if the learner was in fact created based on the an existing
      # checkpoint.
      self.assertTrue(learner_from_checkpoint._checkpointer.checkpoint_exists)

      # Check if the values of the variables of the learner initialized from
      # checkpoint that are the same as the values were used to write the
      # checkpoint.
      original_learner = learners[name][0]
      self.assertAllClose(
          learner_from_checkpoint._agent.collect_policy.variables(),
          original_learner._agent.collect_policy.variables())
      self.assertAllClose(learner_from_checkpoint._agent._optimizer.variables(),
                          original_learner._agent._optimizer.variables())

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

    # Check that the optimizer variables are close to each other.
    default_optimizer_vars = optimizer_variables['default']
    for name, optimizer_vars in optimizer_variables.items():
      self.assertAllClose(
          optimizer_vars,
          default_optimizer_vars,
          atol=1.e-2,
          rtol=1.e-2,
          msg=('The initial values of the optimizer variables for the strategy '
               '{} are significantly different from the initial values of the '
               'default strategy.').format(name))

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
