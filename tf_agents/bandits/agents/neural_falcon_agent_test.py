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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.agents import neural_falcon_agent
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.policies import utils as policy_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types


class DummyNet(network.Network):

  def __init__(self, observation_spec, action_spec, name=None):
    super(DummyNet, self).__init__(observation_spec, state_spec=(), name=name)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer([[1, 1.5,
                                                         2], [1, 1.5, 4],
                                                        [2, 1.5, -1]]),
            bias_initializer=tf.constant_initializer([[1], [1], [-10]]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class NeuralFalconAgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NeuralFalconAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([3], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._per_arm_action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=9)
    self._observation_spec = self._time_step_spec.observation

  def _num_actions(self, accepts_per_arm_features: bool):
    action_spec = (
        self._per_arm_action_spec
        if accepts_per_arm_features else self._action_spec)
    return action_spec.maximum - action_spec.minimum + 1

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

  def _create_agent(
      self,
      accepts_per_arm_features: bool) -> neural_falcon_agent.NeuralFalconAgent:
    if accepts_per_arm_features:
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=1e-2)
      obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
          global_dim=3,
          per_arm_dim=3,
          max_num_actions=self._num_actions(accepts_per_arm_features),
          add_num_actions_feature=True)
      time_step_spec = ts.time_step_spec(obs_spec)
      reward_net = (
          global_and_arm_feature_network
          .create_feed_forward_dot_product_network(
              obs_spec,
              global_layers=[3],
              arm_layers=[3],
              activation_fn=tf.keras.activations.linear))
      agent = neural_falcon_agent.NeuralFalconAgent(
          time_step_spec,
          action_spec=self._per_arm_action_spec,
          reward_network=reward_net,
          accepts_per_arm_features=True,
          num_samples_list=[
              tf.compat.v2.Variable(0, dtype=tf.int64, name='num_samples')
          ],
          exploitation_coefficient=10000.0,
          emit_policy_info=(policy_utils.InfoFields.LOG_PROBABILITY,
                            policy_utils.InfoFields.PREDICTED_REWARDS_MEAN),
          optimizer=optimizer)
    else:
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      reward_net = DummyNet(self._observation_spec, self._action_spec)
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
    return agent

  def _generate_observations(
      self, batch_size: int,
      accepts_per_arm_features: bool) -> types.NestedTensor:
    if accepts_per_arm_features:
      num_actions = self._num_actions(accepts_per_arm_features)
      observations = {
          bandit_spec_utils.GLOBAL_FEATURE_KEY:
              np.array(
                  np.random.normal(size=[batch_size, 3]), dtype=np.float32),
          bandit_spec_utils.PER_ARM_FEATURE_KEY:
              np.array(
                  np.random.normal(size=[batch_size, num_actions, 3]),
                  dtype=np.float32),
          bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
              num_actions * tf.ones([batch_size], dtype=tf.int32)
      }
    else:
      observations = np.array(
          np.random.normal(size=[batch_size, 3]), dtype=np.float32)
    return observations

  def _generate_reward(self, observations: types.NestedTensor,
                       actions: types.Tensor,
                       accepts_per_arm_features: bool) -> types.Tensor:
    if accepts_per_arm_features:
      global_obs = observations[bandit_spec_utils.GLOBAL_FEATURE_KEY]
      arm_obs = observations[bandit_spec_utils.PER_ARM_FEATURE_KEY]
      return tf.reduce_sum(
          global_obs * tf.gather(params=arm_obs, indices=actions, batch_dims=1),
          axis=1)
    else:
      return tf.gather(params=observations, indices=actions, batch_dims=1)

  def _get_policy_info(
      self, batch_size: int,
      accepts_per_arm_features: bool) -> types.NestedSpecTensorOrArray:
    dummy_log_prob = tf.ones([batch_size], dtype=tf.float32)
    num_actions = self._num_actions(accepts_per_arm_features)
    dummy_predicted_rewards = tf.ones([batch_size, num_actions],
                                      dtype=tf.float32)
    if accepts_per_arm_features:
      dummy_chosen_arm_features = tf.ones([batch_size, 3], dtype=tf.float32)
      return policy_utils.PerArmPolicyInfo(
          log_probability=dummy_log_prob,
          predicted_rewards_mean=dummy_predicted_rewards,
          predicted_rewards_sampled=(),
          bandit_policy_type=(),
          multiobjective_scalarized_predicted_rewards_mean=(),
          chosen_arm_features=dummy_chosen_arm_features)
    else:
      return policy_utils.PolicyInfo(
          log_probability=dummy_log_prob,
          predicted_rewards_mean=dummy_predicted_rewards,
          predicted_rewards_sampled=(),
          bandit_policy_type=(),
          multiobjective_scalarized_predicted_rewards_mean=())

  def _generate_training_experience(
      self, accepts_per_arm_features: bool) -> types.NestedTensor:
    batch_size = 240
    observations = self._generate_observations(batch_size,
                                               accepts_per_arm_features)
    num_actions = self._num_actions(accepts_per_arm_features)
    actions = np.tile(
        np.array(range(num_actions), dtype=np.int32),
        int(batch_size / num_actions))
    rewards = self._generate_reward(observations, actions,
                                    accepts_per_arm_features)
    experience = trajectory.single_step(
        observation=observations,
        action=actions,
        policy_info=self._get_policy_info(batch_size, accepts_per_arm_features),
        reward=rewards,
        discount=tf.zeros([batch_size]))
    experience = tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1),
                                       experience)
    return experience

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
    observations = tf.constant([[1, 2, 3]] * batch_size, dtype=tf.float32)
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

  @parameterized.named_parameters(
      {
          'testcase_name': 'accepts_per_arm_features',
          'accepts_per_arm_features': True
      }, {
          'testcase_name': 'simple_action',
          'accepts_per_arm_features': False
      })
  def testTrainedPolicy(self, accepts_per_arm_features):
    agent = self._create_agent(accepts_per_arm_features)
    # Train the policy.
    # Initialize all variables
    self.evaluate(tf.compat.v1.global_variables_initializer())
    for _ in range(20):
      self.evaluate(
          agent.train(
              self._generate_training_experience(accepts_per_arm_features),
              None).loss)

    # Due to the large `exploitation_coefficient`, the trained policy is
    # expected to choose greedily.
    batch_size = 100
    observations = self._generate_observations(batch_size,
                                               accepts_per_arm_features)
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
