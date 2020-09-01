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

"""Tests for tf_agents.bandits.agents.neural_linucb_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp
from tf_agents.bandits.agents import neural_linucb_agent
from tf_agents.bandits.agents import utils as bandit_utils
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step
from tf_agents.utils import common


tfd = tfp.distributions


class DummyNet(network.Network):

  def __init__(self, observation_spec, encoding_dim=10):
    super(DummyNet, self).__init__(
        observation_spec, state_spec=(), name='DummyNet')
    context_dim = observation_spec.shape[0]

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            encoding_dim,
            kernel_initializer=tf.compat.v1.initializers.constant(
                np.ones([context_dim, encoding_dim])),
            bias_initializer=tf.compat.v1.initializers.constant(
                np.zeros([encoding_dim])))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_batch1_contextdim10',
          'batch_size': 1,
          'context_dim': 10,
      }, {
          'testcase_name': '_batch4_contextdim5',
          'batch_size': 4,
          'context_dim': 5,
      })


def _get_initial_and_final_steps(batch_size, context_dim):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  initial_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.FIRST, dtype=tf.int32, shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(observation, dtype=tf.float32,
                  shape=[batch_size, context_dim], name='observation'))
  final_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.LAST, dtype=tf.int32, shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      tf.constant(observation + 100.0, dtype=tf.float32,
                  shape=[batch_size, context_dim], name='observation'))
  return initial_step, final_step


def _get_initial_and_final_steps_with_action_mask(batch_size,
                                                  context_dim,
                                                  num_actions=None):
  observation = np.array(range(batch_size * context_dim)).reshape(
      [batch_size, context_dim])
  observation = tf.constant(observation, dtype=tf.float32)
  mask = 1 - tf.eye(batch_size, num_columns=num_actions, dtype=tf.int32)
  reward = np.random.uniform(0.0, 1.0, [batch_size])
  initial_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observation, mask))
  final_step = time_step.TimeStep(
      tf.constant(
          time_step.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(reward, dtype=tf.float32, shape=[batch_size], name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observation + 100.0, mask))
  return initial_step, final_step


def _get_action_step(action):
  return policy_step.PolicyStep(
      action=tf.convert_to_tensor(action),
      info=policy_utilities.PolicyInfo())


def _get_experience(initial_step, action_step, final_step):
  single_experience = driver_utils.trajectory_for_bandit(
      initial_step, action_step, final_step)
  # Adds a 'time' dimension.
  return tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.convert_to_tensor(x), 1),
      single_experience)


class NeuralLinUCBAgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NeuralLinUCBAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()

  @test_cases()
  def testInitializeAgentNumTrainSteps0(self, batch_size, context_dim):
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)

    encoder = DummyNet(observation_spec)
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=0,
        encoding_dim=10,
        optimizer=None)
    self.evaluate(agent.initialize())

  @test_cases()
  def testInitializeAgentNumTrainSteps10(self, batch_size, context_dim):
    num_actions = 5
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)

    encoder = DummyNet(observation_spec)
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=10,
        encoding_dim=10,
        optimizer=None)
    self.evaluate(agent.initialize())

  @test_cases()
  def testNeuralLinUCBUpdateNumTrainSteps0(self, batch_size=1, context_dim=10):
    """Check NeuralLinUCBAgent updates when behaving like LinUCB."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    encoder = DummyNet(observation_spec)
    encoding_dim = 10
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=0,
        encoding_dim=encoding_dim,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2))

    loss_info = agent.train(experience)
    self.evaluate(agent.initialize())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    for _, (observations_for_arm, rewards_for_arm) in enumerate(zip(
        observations_list, rewards_list)):

      encoded_observations_for_arm, _ = encoder(observations_for_arm)

      num_samples_for_arm_current = tf.shape(rewards_for_arm)[0]
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            encoded_observations_for_arm,
            encoded_observations_for_arm,
            transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, encoded_observations_for_arm)
        return a_new, b_new
      def false_fn():
        return (tf.zeros([encoding_dim, encoding_dim], dtype=tf.float32),
                tf.zeros([encoding_dim], dtype=tf.float32))

      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0,
          true_fn,
          false_fn)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)

  @test_cases()
  def testNeuralLinUCBUpdateDistributed(self, batch_size=1, context_dim=10):
    """Same as above but with distributed LinUCB updates."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    encoder = DummyNet(observation_spec)
    encoding_dim = 10
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=0,
        encoding_dim=encoding_dim,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2))

    self.evaluate(agent.initialize())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    # Call the distributed LinUCB training instead of agent.train().
    train_fn = common.function_in_tf1()(
        agent.compute_loss_using_linucb_distributed)
    reward = tf.cast(experience.reward, agent._dtype)
    loss_info = train_fn(
        experience.observation, action, reward, weights=None)
    self.evaluate(loss_info)
    final_a = self.evaluate(agent.cov_matrix)
    final_b = self.evaluate(agent.data_vector)

    # Compute the expected updated estimates.
    observations_list = tf.dynamic_partition(
        data=tf.reshape(experience.observation, [batch_size, context_dim]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    rewards_list = tf.dynamic_partition(
        data=tf.reshape(experience.reward, [batch_size]),
        partitions=tf.convert_to_tensor(action),
        num_partitions=num_actions)
    expected_a_updated_list = []
    expected_b_updated_list = []
    for _, (observations_for_arm, rewards_for_arm) in enumerate(zip(
        observations_list, rewards_list)):

      encoded_observations_for_arm, _ = encoder(observations_for_arm)

      num_samples_for_arm_current = tf.cast(
          tf.shape(rewards_for_arm)[0], tf.float32)
      num_samples_for_arm_total = num_samples_for_arm_current

      # pylint: disable=cell-var-from-loop
      def true_fn():
        a_new = tf.matmul(
            encoded_observations_for_arm,
            encoded_observations_for_arm,
            transpose_a=True)
        b_new = bandit_utils.sum_reward_weighted_observations(
            rewards_for_arm, encoded_observations_for_arm)
        return a_new, b_new
      def false_fn():
        return (tf.zeros([encoding_dim, encoding_dim], dtype=tf.float32),
                tf.zeros([encoding_dim], dtype=tf.float32))

      a_new, b_new = tf.cond(
          tf.squeeze(num_samples_for_arm_total) > 0,
          true_fn,
          false_fn)

      expected_a_updated_list.append(self.evaluate(a_new))
      expected_b_updated_list.append(self.evaluate(b_new))

    # Check that the actual updated estimates match the expectations.
    self.assertAllClose(expected_a_updated_list, final_a)
    self.assertAllClose(expected_b_updated_list, final_b)

  @test_cases()
  def testNeuralLinUCBUpdateNumTrainSteps10(self, batch_size=1, context_dim=10):
    """Check NeuralLinUCBAgent updates when behaving like eps-greedy."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps(
        batch_size, context_dim)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = tensor_spec.TensorSpec([context_dim], tf.float32)
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    encoder = DummyNet(observation_spec)
    encoding_dim = 10
    variable_collection = neural_linucb_agent.NeuralLinUCBVariableCollection(
        num_actions, encoding_dim)
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=10,
        encoding_dim=encoding_dim,
        variable_collection=variable_collection,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001))

    loss_info, _ = agent.train(experience)
    self.evaluate(agent.initialize())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_value = self.evaluate(loss_info)
    self.assertGreater(loss_value, 0.0)

  @test_cases()
  def testNeuralLinUCBUpdateNumTrainSteps10MaskedActions(
      self, batch_size=1, context_dim=10):
    """Check updates when behaving like eps-greedy and using masked actions."""

    # Construct a `Trajectory` for the given action, observation, reward.
    num_actions = 5
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        batch_size, context_dim, num_actions)
    action = np.random.randint(num_actions, size=batch_size, dtype=np.int32)
    action_step = _get_action_step(action)
    experience = _get_experience(initial_step, action_step, final_step)

    # Construct an agent and perform the update.
    observation_spec = (tensor_spec.TensorSpec([context_dim], tf.float32),
                        tensor_spec.TensorSpec([num_actions], tf.int32))
    time_step_spec = time_step.time_step_spec(observation_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    encoder = DummyNet(observation_spec[0])
    encoding_dim = 10
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=10,
        encoding_dim=encoding_dim,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]))

    loss_info, _ = agent.train(experience)
    self.evaluate(agent.initialize())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_value = self.evaluate(loss_info)
    self.assertGreater(loss_value, 0.0)

  def testInitializeRestoreVariableCollection(self):
    if not tf.executing_eagerly():
      self.skipTest('Test only works in eager mode.')
    num_actions = 5
    encoding_dim = 7
    variable_collection = neural_linucb_agent.NeuralLinUCBVariableCollection(
        num_actions=num_actions, encoding_dim=encoding_dim)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(variable_collection.num_samples_list)
    checkpoint = tf.train.Checkpoint(variable_collection=variable_collection)
    checkpoint_dir = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint')
    checkpoint.save(file_prefix=checkpoint_prefix)

    variable_collection.actions_from_reward_layer.assign(False)

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)
    self.evaluate(checkpoint_load_status.initialize_or_restore())
    self.assertEqual(
        self.evaluate(variable_collection.actions_from_reward_layer), True)

  def testTrainPerArmAgentVariableActions(self):
    num_actions = 5
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        2, 3, num_actions, add_num_actions_feature=True)
    time_step_spec = time_step.time_step_spec(obs_spec)
    action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=num_actions - 1)
    encoding_dim = 10
    encoder = (
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2), encoding_dim))
    agent = neural_linucb_agent.NeuralLinUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        encoding_network=encoder,
        encoding_network_num_train_steps=10,
        encoding_dim=encoding_dim,
        accepts_per_arm_features=True,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001))
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tf.cast(
                tf.reshape(tf.range(30), shape=[2, 5, 3]), dtype=tf.float32),
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
            tf.constant([3, 4], dtype=tf.int32)
    }
    actions = np.array([0, 3], dtype=np.int32)
    rewards = np.array([0.5, 3.0], dtype=np.float32)
    initial_step = time_step.TimeStep(
        tf.constant(
            time_step.StepType.FIRST,
            dtype=tf.int32,
            shape=[2],
            name='step_type'),
        tf.constant(0.0, dtype=tf.float32, shape=[2], name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[2], name='discount'),
        observations)
    final_step = time_step.TimeStep(
        tf.constant(
            time_step.StepType.LAST,
            dtype=tf.int32,
            shape=[2],
            name='step_type'),
        tf.constant(rewards, dtype=tf.float32, name='reward'),
        tf.constant(1.0, dtype=tf.float32, shape=[2], name='discount'),
        observations)
    action_step = policy_step.PolicyStep(
        action=tf.convert_to_tensor(actions),
        info=policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=np.array([[1, 2, 3], [3, 2, 1]],
                                         dtype=np.float32)))
    experience = _get_experience(initial_step, action_step, final_step)
    loss_info, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    loss_value = self.evaluate(loss_info)
    self.assertGreater(loss_value, 0.0)


if __name__ == '__main__':
  tf.test.main()
