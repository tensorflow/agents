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

"""Tests for tf_agents.agents.sac.sac_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import mock
import tensorflow as tf

from tf_agents.agents import test_util
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import nest_map
from tf_agents.networks import network
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import test_utils


class _MockDistribution(object):

  def __init__(self, action):
    self._action = action

  def sample(self):
    return self._action

  def log_prob(self, unused_sample):
    return tf.constant(10., shape=[1])


class DummyActorPolicy(object):

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               training=False):
    del time_step_spec
    del actor_network
    del training
    single_action_spec = tf.nest.flatten(action_spec)[0]
    # Action is maximum of action range.
    self._action = single_action_spec.maximum
    self._action_spec = action_spec
    self.info_spec = ()

  def action(self, time_step):
    observation = time_step.observation
    batch_size = observation.shape[0]
    action = tf.constant(self._action, dtype=tf.float32, shape=[batch_size, 1])
    return policy_step.PolicyStep(action=action)

  def distribution(self, time_step, policy_state=()):
    del policy_state
    action = self.action(time_step).action
    return policy_step.PolicyStep(action=_MockDistribution(action))

  def get_initial_state(self, batch_size):
    del batch_size
    return ()


class DummyCriticNet(network.Network):

  def __init__(self, l2_regularization_weight=0.0, shared_layer=None):
    super(DummyCriticNet, self).__init__(
        input_tensor_spec=(tensor_spec.TensorSpec([2], tf.float32),
                           tensor_spec.TensorSpec([1], tf.float32)),
        state_spec=(),
        name=None)
    self._l2_regularization_weight = l2_regularization_weight
    self._value_layer = tf.keras.layers.Dense(
        1,
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_weight),
        kernel_initializer=tf.constant_initializer([[0], [1]]),
        bias_initializer=tf.constant_initializer([[0]]))
    self._shared_layer = shared_layer
    self._action_layer = tf.keras.layers.Dense(
        1,
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_weight),
        kernel_initializer=tf.constant_initializer([[1]]),
        bias_initializer=tf.constant_initializer([[0]]))

  def copy(self, name=''):
    del name
    return DummyCriticNet(
        l2_regularization_weight=self._l2_regularization_weight,
        shared_layer=self._shared_layer)

  def call(self, inputs, step_type, network_state=()):
    del step_type
    observation, actions = inputs
    actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)

    states = tf.cast(tf.nest.flatten(observation)[0], tf.float32)

    s_value = self._value_layer(states)
    if self._shared_layer:
      s_value = self._shared_layer(s_value)
    a_value = self._action_layer(actions)
    # Biggest state is best state.
    q_value = tf.reshape(s_value + a_value, [-1])
    return q_value, network_state


def create_sequential_critic_net(l2_regularization_weight=0.0,
                                 shared_layer=None):
  value_layer = tf.keras.layers.Dense(
      1,
      kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_weight),
      kernel_initializer=tf.initializers.constant([[0], [1]]),
      bias_initializer=tf.initializers.constant([[0]]))
  if shared_layer:
    value_layer = sequential.Sequential([value_layer, shared_layer])

  action_layer = tf.keras.layers.Dense(
      1,
      kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_weight),
      kernel_initializer=tf.initializers.constant([[1]]),
      bias_initializer=tf.initializers.constant([[0]]))

  def sum_value_and_action_out(value_and_action_out):
    value_out, action_out = value_and_action_out
    return tf.reshape(value_out + action_out, [-1])

  return sequential.Sequential([
      nest_map.NestMap((value_layer, action_layer)),
      tf.keras.layers.Lambda(sum_value_and_action_out)
  ])


class SacAgentTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(SacAgentTest, self).setUp()
    self._obs_spec = tensor_spec.BoundedTensorSpec([2],
                                                   tf.float32,
                                                   minimum=0,
                                                   maximum=1)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

  @parameterized.named_parameters(('Network', DummyCriticNet, False),
                                  ('Keras', create_sequential_critic_net, True))
  def testCreateAgent(self, create_critic_net_fn, skip_in_tf1):
    if skip_in_tf1 and not common.has_eager_been_enabled():
      self.skipTest('Skipping test: sequential networks not supported in TF1')

    critic_network = create_critic_net_fn()

    sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=critic_network,
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        actor_policy_ctor=DummyActorPolicy)

  def testAgentTrajectoryTrain(self):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._obs_spec,
        self._action_spec,
        fc_layer_params=(10,),
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=actor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(0.001))

    trajectory_spec = trajectory.Trajectory(
        step_type=self._time_step_spec.step_type,
        observation=self._time_step_spec.observation,
        action=self._action_spec,
        policy_info=(),
        next_step_type=self._time_step_spec.step_type,
        reward=tensor_spec.BoundedTensorSpec(
            [], tf.float32, minimum=0.0, maximum=1.0, name='reward'),
        discount=self._time_step_spec.discount)

    sample_trajectory_experience = tensor_spec.sample_spec_nest(
        trajectory_spec, outer_dims=(3, 2))
    agent.train(sample_trajectory_experience)

  def testAgentTransitionTrain(self):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._obs_spec,
        self._action_spec,
        fc_layer_params=(10,),
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=actor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(0.001))

    time_step_spec = self._time_step_spec._replace(
        reward=tensor_spec.BoundedTensorSpec(
            [], tf.float32, minimum=0.0, maximum=1.0, name='reward'))

    transition_spec = trajectory.Transition(
        time_step=time_step_spec,
        action_step=policy_step.PolicyStep(action=self._action_spec,
                                           state=(),
                                           info=()),
        next_time_step=time_step_spec)

    sample_trajectory_experience = tensor_spec.sample_spec_nest(
        transition_spec, outer_dims=(3,))
    agent.train(sample_trajectory_experience)

  @parameterized.named_parameters(('Network', DummyCriticNet, False),
                                  ('Keras', create_sequential_critic_net, True))
  def testCriticLoss(self, create_critic_net_fn, skip_in_tf1):
    if skip_in_tf1 and not common.has_eager_been_enabled():
      self.skipTest('Skipping test: sequential networks not supported in TF1')

    critic_network = create_critic_net_fn()
    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=critic_network,
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[5], [6]], dtype=tf.float32)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    td_targets = [7.3, 19.1]
    pred_td_targets = [7., 10.]

    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Expected critic loss has factor of 2, for the two TD3 critics.
    expected_loss = self.evaluate(2 * tf.compat.v1.losses.mean_squared_error(
        tf.constant(td_targets), tf.constant(pred_td_targets)))

    loss = agent.critic_loss(
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn=tf.math.squared_difference)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testCriticRegLoss(self):
    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(0.5),
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.zeros((2, 2), dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.zeros((2, 1), dtype=tf.float32)

    rewards = tf.zeros((2,), dtype=tf.float32)
    discounts = tf.zeros((2,), dtype=tf.float32)
    next_observations = tf.zeros((2, 2), dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    # Expected loss only regularization loss.
    expected_loss = 2.0

    loss = agent.critic_loss(
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn=tf.math.squared_difference)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testActorLoss(self):
    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    expected_loss = (2 * 10 - (2 + 1) - (4 + 1)) / 2
    loss = agent.actor_loss(time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testAlphaLoss(self):
    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        target_entropy=3.0,
        initial_log_alpha=4.0,
        actor_policy_ctor=DummyActorPolicy)
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)

    expected_loss = 4.0 * (-10 - 3)
    loss = agent.alpha_loss(time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @mock.patch.object(sac_agent.SacAgent, '_apply_gradients')
  @mock.patch.object(sac_agent.SacAgent, '_actions_and_log_probs')
  def testLoss(self, mock_actions_and_log_probs, mock_apply_gradients):
    # Mock _actions_and_log_probs so that _train() and _loss() run on the same
    # sampled values.
    actions = tf.constant([[0.2], [0.5], [-0.8]])
    log_pi = tf.constant([-1.1, -0.8, -0.5])
    mock_actions_and_log_probs.return_value = (actions, log_pi)

    # Skip applying gradients since mocking _actions_and_log_probs.
    del mock_apply_gradients

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._obs_spec,
        self._action_spec,
        fc_layer_params=(10,),
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=actor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(0.001))

    observations = tf.constant(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
        dtype=tf.float32)
    actions = tf.constant([[[0], [1]], [[2], [3]], [[4], [5]]],
                          dtype=tf.float32)
    time_steps = ts.TimeStep(
        step_type=tf.constant([[1, 1]] * 3, dtype=tf.int32),
        reward=tf.constant([[1, 1]] * 3, dtype=tf.float32),
        discount=tf.constant([[1, 1]] * 3, dtype=tf.float32),
        observation=observations)

    experience = trajectory.Trajectory(
        time_steps.step_type, observations, actions, (),
        time_steps.step_type, time_steps.reward, time_steps.discount)

    test_util.test_loss_and_train_output(
        test=self,
        expect_equal_loss_values=True,
        agent=agent,
        experience=experience)

  def testPolicy(self):
    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_steps = ts.restart(observations)
    action_step = agent.policy.action(time_steps)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    action_ = self.evaluate(action_step.action)
    self.assertLessEqual(action_, self._action_spec.maximum)
    self.assertGreaterEqual(action_, self._action_spec.minimum)

  def testTrainWithRnn(self):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        self._obs_spec,
        self._action_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        conv_layer_params=None,
        lstm_size=(40,),
    )

    critic_net = critic_rnn_network.CriticRnnNetwork(
        (self._obs_spec, self._action_spec),
        observation_fc_layer_params=(16,),
        action_fc_layer_params=(16,),
        joint_fc_layer_params=(16,),
        lstm_size=(16,),
        output_fc_layer_params=None,
    )

    counter = common.create_variable('test_train_counter')

    optimizer_fn = tf.compat.v1.train.AdamOptimizer

    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=critic_net,
        actor_network=actor_net,
        actor_optimizer=optimizer_fn(1e-3),
        critic_optimizer=optimizer_fn(1e-3),
        alpha_optimizer=optimizer_fn(1e-3),
        train_step_counter=counter,
    )

    batch_size = 5
    observations = tf.constant(
        [[[1, 2], [3, 4], [5, 6]]] * batch_size, dtype=tf.float32)
    actions = tf.constant([[[0], [1], [1]]] * batch_size, dtype=tf.float32)
    time_steps = ts.TimeStep(
        step_type=tf.constant([[1] * 3] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        observation=observations)

    experience = trajectory.Trajectory(
        time_steps.step_type, observations, actions, (),
        time_steps.step_type, time_steps.reward, time_steps.discount)

    # Force variable creation.
    agent.policy.variables()

    if not tf.executing_eagerly():
      # Get experience first to make sure optimizer variables are created and
      # can be initialized.
      experience = agent.train(experience)
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
      self.assertEqual(self.evaluate(counter), 0)
      self.evaluate(experience)
      self.assertEqual(self.evaluate(counter), 1)
    else:
      self.assertEqual(self.evaluate(counter), 0)
      self.evaluate(agent.train(experience))
      self.assertEqual(self.evaluate(counter), 1)

  def testSharedLayer(self):
    shared_layer = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.constant_initializer([0]),
        bias_initializer=tf.constant_initializer([0]),
        name='shared')

    critic_net_1 = DummyCriticNet(shared_layer=shared_layer)
    critic_net_2 = DummyCriticNet(shared_layer=shared_layer)

    target_shared_layer = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.constant_initializer([0]),
        bias_initializer=tf.constant_initializer([0]),
        name='shared_target')

    target_critic_net_1 = DummyCriticNet(shared_layer=target_shared_layer)
    target_critic_net_2 = DummyCriticNet(shared_layer=target_shared_layer)

    agent = sac_agent.SacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=critic_net_1,
        critic_network_2=critic_net_2,
        target_critic_network=target_critic_net_1,
        target_critic_network_2=target_critic_net_2,
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        target_entropy=3.0,
        initial_log_alpha=4.0,
        target_update_tau=0.5,
        actor_policy_ctor=DummyActorPolicy)

    self.evaluate([
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()
    ])

    self.evaluate(agent.initialize())

    for v in shared_layer.variables:
      self.evaluate(v.assign(v * 0 + 1))

    self.evaluate(agent._update_target())

    self.assertEqual(1.0, self.evaluate(shared_layer.variables[0][0][0]))
    self.assertEqual(0.5, self.evaluate(target_shared_layer.variables[0][0][0]))


if __name__ == '__main__':
  tf.test.main()
