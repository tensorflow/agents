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

"""Tests for neural_falcon_agent."""

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import neural_falcon_agent
from tf_agents.networks import network
from tf_agents.policies import utils as policy_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


class DummyNet(network.Network):

  def __init__(self, observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer([[1, 1.5, 2],
                                                        [1, 1.5, 4]]),
            bias_initializer=tf.constant_initializer([[1], [1], [-10]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class NeuralFalconAgentTest(tf.test.TestCase):

  def setUp(self):
    super(NeuralFalconAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation

  def _check_uniform_actions(self, actions: np.ndarray,
                             batch_size: int) -> None:
    self.assertAllInSet(actions, [0, 1, 2])
    # Set tolerance in the chosen count to be 4 std.
    tol = 4.0 * np.sqrt(batch_size * 1.0 / 3 * 2.0 / 3)
    expected_count = batch_size / 3
    for action in range(3):
      action_chosen_count = np.sum(actions == action)
      self.assertNear(
          action_chosen_count,
          expected_count,
          tol,
          msg=f'action: {action} is expected to be chosen between '
          f'{expected_count - tol} and {expected_count + tol} times, but was '
          f'actually chosen {action_chosen_count} times.')

  def testUntrainedPolicy(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = neural_falcon_agent.NeuralFalconAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        num_samples_list=[
            tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples_0'),
            tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples_1'),
            tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples_2')
        ],
        exploitation_coefficient=10000.0,
        emit_policy_info=(policy_utils.InfoFields.LOG_PROBABILITY,),
        optimizer=optimizer)

    # An untrained policy is expected to sample actions uniformly at random.
    batch_size = 3000
    observations = tf.constant([[1, 2]] * batch_size, dtype=tf.float32)
    time_step = ts.restart(observations, batch_size=batch_size)
    # Untrained policy samples actions uniformly at random.
    action_step = agent.policy.action(time_step, seed=1)
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are uniform.
    self.assertAllEqual(p_info.log_probability,
                        tf.math.log([1.0 / 3] * batch_size))
    # Check the empirical distribution of the chosen arms is uniform.
    actions = self.evaluate(action_step.action)
    self._check_uniform_actions(actions, batch_size)

  def testTrainedPolicy(self):
    reward_net = DummyNet(self._observation_spec, self._action_spec)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    agent = neural_falcon_agent.NeuralFalconAgent(
        self._time_step_spec,
        self._action_spec,
        reward_network=reward_net,
        num_samples_list=[
            tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples_0'),
            tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples_1'),
            tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples_2')
        ],
        exploitation_coefficient=10000.0,
        emit_policy_info=(policy_utils.InfoFields.LOG_PROBABILITY,
                          policy_utils.InfoFields.PREDICTED_REWARDS_MEAN),
        optimizer=optimizer)

    # Train the policy.
    train_batch_size = 240
    dummy_log_prob = tf.ones([train_batch_size], dtype=tf.float32)
    dummy_predicted_rewards = tf.ones([train_batch_size, 3], dtype=tf.float32)
    policy_info = policy_utils.PolicyInfo(
        log_probability=dummy_log_prob,
        predicted_rewards_mean=dummy_predicted_rewards,
        predicted_rewards_sampled=(),
        bandit_policy_type=(),
        multiobjective_scalarized_predicted_rewards_mean=())
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(10):
      observations = np.array(
          np.random.normal(size=[train_batch_size, 2]), dtype=np.float32)
      actions = np.tile(
          np.array([0, 1, 2], dtype=np.int32), int(train_batch_size / 3))
      rewards = np.array(
          np.random.uniform(size=[train_batch_size]), dtype=np.float32)
      experience = trajectory.single_step(
          observation=observations,
          action=actions,
          policy_info=policy_info,
          reward=rewards,
          discount=tf.zeros([train_batch_size]))
      experience = tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1),
                                         experience)
      self.evaluate(agent.train(experience, None).loss)

    # Due to the large `exploitation_coefficient`, the trained policy is
    # expected to choose greedily.
    batch_size = 100
    observations = tf.constant(
        np.array(np.random.normal(size=[batch_size, 2]), dtype=np.float32),
        dtype=tf.float32)
    time_step = ts.restart(observations, batch_size)
    action_step = agent.policy.action(time_step, seed=1)
    actions = self.evaluate(action_step.action)
    p_info = self.evaluate(action_step.info)
    # Check the log probabilities in the policy info are near greedy.
    self.assertAllClose(p_info.log_probability, [0.0] * batch_size, atol=5e-3)
    # Check the chosen arms are greedy.
    self.assertAllEqual(actions,
                        np.argmax(p_info.predicted_rewards_mean, axis=1))

if __name__ == '__main__':
  tf.test.main()
