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

"""Tests for TF Agents ppo_eager_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
from absl.testing.absltest import mock

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.networks import value_network
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class DummyActorNet(network.DistributionNetwork):

  def __init__(self, input_spec, action_spec, name=None):
    output_spec = self._get_normal_distribution_spec(action_spec)
    super(DummyActorNet, self).__init__(
        input_spec, (), output_spec=output_spec, name='DummyActorNet')
    self._action_spec = action_spec
    self._flat_action_spec = tf.nest.flatten(self._action_spec)[0]

    self._layers.append(
        tf.keras.layers.Dense(
            self._flat_action_spec.shape.num_elements() * 2,
            kernel_initializer=tf.compat.v1.initializers.constant([[2.0, 1.0],
                                                                   [1.0, 1.0]]),
            bias_initializer=tf.compat.v1.initializers.constant([5.0, 5.0]),
            activation=None,
        ))

  def _get_normal_distribution_spec(self, sample_spec):
    input_param_shapes = tfp.distributions.Normal.param_static_shapes(
        sample_spec.shape)
    input_param_spec = tf.nest.map_structure(
        lambda tensor_shape: tensor_spec.TensorSpec(  # pylint: disable=g-long-lambda
            shape=tensor_shape,
            dtype=sample_spec.dtype),
        input_param_shapes)

    return distribution_spec.DistributionSpec(
        tfp.distributions.Normal, input_param_spec, sample_spec=sample_spec)

  def call(self, inputs, unused_step_type=None, network_state=()):
    hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]

    # Calls coming from agent.train() has a time dimension. Direct loss calls
    # may not have a time dimension. It order to make BatchSquash work, we need
    # to specify the outer dimension properly.
    has_time_dim = nest_utils.get_outer_rank(inputs,
                                             self.input_tensor_spec) == 2
    outer_rank = 2 if has_time_dim else 1
    batch_squash = network_utils.BatchSquash(outer_rank)
    hidden_state = batch_squash.flatten(hidden_state)

    for layer in self.layers:
      hidden_state = layer(hidden_state)

    actions, stdevs = tf.split(hidden_state, 2, axis=1)
    actions = batch_squash.unflatten(actions)
    stdevs = batch_squash.unflatten(stdevs)
    actions = tf.nest.pack_sequence_as(self._action_spec, [actions])
    stdevs = tf.nest.pack_sequence_as(self._action_spec, [stdevs])

    return self.output_spec.build_distribution(
        loc=actions, scale=stdevs), network_state


class DummyValueNet(network.Network):

  def __init__(self, observation_spec, name=None, outer_rank=1):
    super(DummyValueNet, self).__init__(observation_spec, (), 'DummyValueNet')
    self._outer_rank = outer_rank
    self._layers.append(
        tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.compat.v1.initializers.constant([2, 1]),
            bias_initializer=tf.compat.v1.initializers.constant([5])))

  def call(self, inputs, unused_step_type=None, network_state=()):
    hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]
    batch_squash = network_utils.BatchSquash(self._outer_rank)
    hidden_state = batch_squash.flatten(hidden_state)
    for layer in self.layers:
      hidden_state = layer(hidden_state)
    value_pred = tf.squeeze(batch_squash.unflatten(hidden_state), axis=-1)
    return value_pred, network_state


def _compute_returns_fn(rewards, discounts, next_state_return=0.0):
  """Python implementation of computing discounted returns."""
  returns = np.zeros_like(rewards)
  for t in range(len(returns) - 1, -1, -1):
    returns[t] = rewards[t] + discounts[t] * next_state_return
    next_state_return = returns[t]
  return returns


class PPOAgentTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(PPOAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

  def testCreateAgent(self):
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        check_numerics=True)
    agent.initialize()

  def testComputeAdvantagesNoGae(self):
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=DummyValueNet(self._obs_spec),
        normalize_observations=False,
        use_gae=False)
    rewards = tf.constant([[1.0] * 9, [1.0] * 9])
    discounts = tf.constant([[1.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0],
                             [1.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0]])
    returns = tf.constant([[5.0, 4.0, 3.0, 2.0, 1.0, 3.439, 2.71, 1.9, 1.0],
                           [3.0, 4.0, 7.0, 2.0, -1.0, 5.439, 2.71, -2.9, 1.0]])
    value_preds = tf.constant([
        [3.0] * 10,
        [3.0] * 10,
    ])  # One extra for final time_step.

    expected_advantages = returns - value_preds[:, :-1]
    advantages = agent.compute_advantages(rewards, returns, discounts,
                                          value_preds)
    self.assertAllClose(expected_advantages, advantages)

  def testComputeAdvantagesWithGae(self):
    gae_lambda = 0.95
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec,),
        value_net=DummyValueNet(self._obs_spec),
        normalize_observations=False,
        use_gae=True,
        lambda_value=gae_lambda)
    rewards = tf.constant([[1.0] * 9, [1.0] * 9])
    discounts = tf.constant([[1.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0],
                             [1.0, 1.0, 1.0, 1.0, 0.0, 0.9, 0.9, 0.9, 0.0]])
    returns = tf.constant([[5.0, 4.0, 3.0, 2.0, 1.0, 3.439, 2.71, 1.9, 1.0],
                           [5.0, 4.0, 3.0, 2.0, 1.0, 3.439, 2.71, 1.9, 1.0]])
    value_preds = tf.constant([[3.0] * 10,
                               [3.0] * 10])  # One extra for final time_step.

    gae_vals = tf.constant([[
        2.0808625, 1.13775, 0.145, -0.9, -2.0, 0.56016475, -0.16355, -1.01, -2.0
    ], [
        2.0808625, 1.13775, 0.145, -0.9, -2.0, 0.56016475, -0.16355, -1.01, -2.0
    ]])
    advantages = agent.compute_advantages(rewards, returns, discounts,
                                          value_preds)
    self.assertAllClose(gae_vals, advantages)

  @parameterized.named_parameters([
      ('OneEpoch', 1, True),
      ('FiveEpochs', 5, False),
  ])
  def testTrain(self, num_epochs, use_td_lambda_return):
    # Mock the build_train_op to return an op for incrementing this counter.
    counter = common.create_variable('test_train_counter')
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec,),
        value_net=DummyValueNet(self._obs_spec),
        normalize_observations=False,
        num_epochs=num_epochs,
        use_gae=use_td_lambda_return,
        use_td_lambda_return=use_td_lambda_return,
        train_step_counter=counter)
    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)

    time_steps = ts.TimeStep(
        step_type=tf.constant([[1] * 3] * 2, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]],
                          dtype=tf.float32)

    action_distribution_parameters = {
        'loc': tf.constant([[[0.0]] * 3] * 2, dtype=tf.float32),
        'scale': tf.constant([[[1.0]] * 3] * 2, dtype=tf.float32),
    }

    policy_info = action_distribution_parameters

    experience = trajectory.Trajectory(
        time_steps.step_type, observations, actions, policy_info,
        time_steps.step_type, time_steps.reward, time_steps.discount)

    # Force variable creation.
    agent.policy.variables()

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    # Assert that counter starts out at zero.
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertEqual(0, self.evaluate(counter))
    self.evaluate(loss)
    # Assert that train_op ran increment_counter num_epochs times.
    self.assertEqual(num_epochs, self.evaluate(counter))

  def testGetEpochLoss(self):
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=DummyValueNet(self._obs_spec),
        normalize_observations=False,
        normalize_rewards=False,
        value_pred_loss_coef=1.0,
        policy_l2_reg=1e-4,
        value_function_l2_reg=1e-4,
        entropy_regularization=0.1,
        importance_ratio_clipping=10,
    )
    observations = tf.constant([[1, 2], [3, 4], [1, 2], [3, 4]],
                               dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1], [0], [1]], dtype=tf.float32)
    returns = tf.constant([1.9, 1.0, 1.9, 1.0], dtype=tf.float32)
    sample_action_log_probs = tf.constant([0.9, 0.3, 0.9, 0.3],
                                          dtype=tf.float32)
    advantages = tf.constant([1.9, 1.0, 1.9, 1.0], dtype=tf.float32)
    weights = tf.constant([1.0, 1.0, 0.0, 0.0], dtype=tf.float32)
    sample_action_distribution_parameters = {
        'loc': tf.constant([[9.0], [15.0], [9.0], [15.0]], dtype=tf.float32),
        'scale': tf.constant([[8.0], [12.0], [8.0], [12.0]], dtype=tf.float32),
    }
    train_step = tf.compat.v1.train.get_or_create_global_step()

    loss_info = agent.get_epoch_loss(
        time_steps,
        actions,
        sample_action_log_probs,
        returns,
        advantages,
        sample_action_distribution_parameters,
        weights,
        train_step,
        debug_summaries=False)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    total_loss, extra_loss_info = self.evaluate(loss_info)
    (policy_gradient_loss, value_estimation_loss, l2_regularization_loss,
     entropy_reg_loss, kl_penalty_loss) = extra_loss_info

    # Check loss values are as expected. Factor of 2/4 is because four timesteps
    # were included in the data, but two were masked out. Reduce_means in losses
    # will divide by 4, but computed loss values are for first 2 timesteps.
    expected_pg_loss = -0.0164646133 * 2 / 4
    expected_ve_loss = 123.205 * 2 / 4
    expected_l2_loss = 1e-4 * 12 * 2 / 4
    expected_ent_loss = -0.370111 * 2 / 4
    expected_kl_penalty_loss = 0.0
    self.assertAllClose(
        expected_pg_loss + expected_ve_loss + expected_l2_loss +
        expected_ent_loss + expected_kl_penalty_loss,
        total_loss,
        atol=0.001,
        rtol=0.001)
    self.assertAllClose(expected_pg_loss, policy_gradient_loss)
    self.assertAllClose(expected_ve_loss, value_estimation_loss)
    self.assertAllClose(expected_l2_loss, l2_regularization_loss, atol=0.001,
                        rtol=0.001)
    self.assertAllClose(expected_ent_loss, entropy_reg_loss)
    self.assertAllClose(expected_kl_penalty_loss, kl_penalty_loss)

  @parameterized.named_parameters([
      ('IsZero', 0),
      ('NotZero', 1),
  ])
  def testL2RegularizationLoss(self, not_zero):
    l2_reg = 1e-4 * not_zero
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=DummyValueNet(self._obs_spec),
        normalize_observations=False,
        policy_l2_reg=l2_reg,
        value_function_l2_reg=l2_reg,
    )

    # Call other loss functions to make sure trainable variables are
    #   constructed.
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)
    sample_action_log_probs = tf.constant([[0.9], [0.3]], dtype=tf.float32)
    advantages = tf.constant([1.9, 1.0], dtype=tf.float32)
    current_policy_distribution, unused_network_state = DummyActorNet(
        self._obs_spec, self._action_spec)(time_steps.observation,
                                           time_steps.step_type, ())
    weights = tf.ones_like(advantages)
    agent.policy_gradient_loss(time_steps, actions, sample_action_log_probs,
                               advantages, current_policy_distribution,
                               weights)
    agent.value_estimation_loss(time_steps, returns, weights)

    # Now request L2 regularization loss.
    # Value function weights are [2, 1], actor net weights are [2, 1, 1, 1].
    expected_loss = l2_reg * ((2**2 + 1) + (2**2 + 1 + 1 + 1))
    # Make sure the network is built before we try to get variables.
    agent.policy.action(
        tensor_spec.sample_spec_nest(self._time_step_spec, outer_dims=(2,)))
    loss = agent.l2_regularization_loss()

    self.evaluate(tf.compat.v1.initialize_all_variables())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.named_parameters([
      ('IsZero', 0),
      ('NotZero', 1),
  ])
  def testEntropyRegularizationLoss(self, not_zero):
    ent_reg = 0.1 * not_zero
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=DummyValueNet(self._obs_spec),
        normalize_observations=False,
        entropy_regularization=ent_reg,
    )

    # Call other loss functions to make sure trainable variables are
    #   constructed.
    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)
    sample_action_log_probs = tf.constant([[0.9], [0.3]], dtype=tf.float32)
    advantages = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(advantages)
    current_policy_distribution, unused_network_state = DummyActorNet(
        self._obs_spec, self._action_spec)(time_steps.observation,
                                           time_steps.step_type, ())
    agent.policy_gradient_loss(time_steps, actions, sample_action_log_probs,
                               advantages, current_policy_distribution,
                               weights)
    agent.value_estimation_loss(time_steps, returns, weights)

    # Now request entropy regularization loss.
    # Action stdevs should be ~1.0, and mean entropy ~3.70111.
    expected_loss = -3.70111 * ent_reg
    loss = agent.entropy_regularization_loss(
        time_steps, current_policy_distribution, weights)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testValueEstimationLoss(self):
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=DummyValueNet(self._obs_spec),
        value_pred_loss_coef=1.0,
        normalize_observations=False,
    )

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(returns)

    expected_loss = 123.205
    loss = agent.value_estimation_loss(time_steps, returns, weights)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testPolicyGradientLoss(self):
    actor_net = DummyActorNet(self._obs_spec, self._action_spec)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        normalize_observations=False,
        normalize_rewards=False,
        actor_net=actor_net,
        importance_ratio_clipping=10.0,
    )

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    sample_action_log_probs = tf.constant([0.9, 0.3], dtype=tf.float32)
    advantages = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(advantages)

    current_policy_distribution, unused_network_state = actor_net(
        time_steps.observation, time_steps.step_type, ())

    expected_loss = -0.0164646133
    loss = agent.policy_gradient_loss(time_steps, actions,
                                      sample_action_log_probs, advantages,
                                      current_policy_distribution, weights)

    self.evaluate(tf.compat.v1.initialize_all_variables())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testKlPenaltyLoss(self):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        fc_layer_params=None)
    value_net = value_network.ValueNetwork(
        self._time_step_spec.observation, fc_layer_params=None)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        kl_cutoff_factor=5.0,
        adaptive_kl_target=0.1,
        kl_cutoff_coef=100,
    )

    agent.kl_cutoff_loss = mock.MagicMock(
        return_value=tf.constant(3.0, dtype=tf.float32))
    agent.adaptive_kl_loss = mock.MagicMock(
        return_value=tf.constant(4.0, dtype=tf.float32))

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    action_distribution_parameters = {
        'loc': tf.constant([1.0, 1.0], dtype=tf.float32),
        'scale': tf.constant([1.0, 1.0], dtype=tf.float32),
    }
    current_policy_distribution, unused_network_state = DummyActorNet(
        self._obs_spec, self._action_spec)(time_steps.observation,
                                           time_steps.step_type, ())
    weights = tf.ones_like(time_steps.discount)

    expected_kl_penalty_loss = 7.0

    kl_penalty_loss = agent.kl_penalty_loss(
        time_steps, action_distribution_parameters, current_policy_distribution,
        weights)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    kl_penalty_loss_ = self.evaluate(kl_penalty_loss)
    self.assertEqual(expected_kl_penalty_loss, kl_penalty_loss_)

  @parameterized.named_parameters([
      ('IsZero', 0),
      ('NotZero', 1),
  ])
  def testKlCutoffLoss(self, not_zero):
    kl_cutoff_coef = 30.0 * not_zero
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        fc_layer_params=None)
    value_net = value_network.ValueNetwork(
        self._time_step_spec.observation, fc_layer_params=None)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        kl_cutoff_factor=5.0,
        adaptive_kl_target=0.1,
        kl_cutoff_coef=kl_cutoff_coef,
    )
    kl_divergence = tf.constant([[1.5, -0.5, 6.5, -1.5, -2.3]],
                                dtype=tf.float32)
    expected_kl_cutoff_loss = kl_cutoff_coef * (.24**2)  # (0.74 - 0.5) ^ 2

    loss = agent.kl_cutoff_loss(kl_divergence)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    loss_ = self.evaluate(loss)
    self.assertAllClose([loss_], [expected_kl_cutoff_loss])

  def testAdaptiveKlLoss(self):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        fc_layer_params=None)
    value_net = value_network.ValueNetwork(
        self._time_step_spec.observation, fc_layer_params=None)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        initial_adaptive_kl_beta=1.0,
        adaptive_kl_target=10.0,
        adaptive_kl_tolerance=0.5,
    )

    # Force variable creation
    agent.policy.variables()
    self.evaluate(tf.compat.v1.initialize_all_variables())

    # Loss should not change if data kl is target kl.
    loss_1 = agent.adaptive_kl_loss([10.0])
    loss_2 = agent.adaptive_kl_loss([10.0])
    self.assertEqual(self.evaluate(loss_1), self.evaluate(loss_2))

    # If data kl is low, kl penalty should decrease between calls.
    loss_1 = self.evaluate(agent.adaptive_kl_loss([1.0]))
    adaptive_kl_beta_update_fn = common.function(agent.update_adaptive_kl_beta)
    self.evaluate(adaptive_kl_beta_update_fn([1.0]))
    loss_2 = self.evaluate(agent.adaptive_kl_loss([1.0]))
    self.assertGreater(loss_1, loss_2)

    # # # If data kl is low, kl penalty should increase between calls.
    loss_1 = self.evaluate(agent.adaptive_kl_loss([100.0]))
    self.evaluate(adaptive_kl_beta_update_fn([100.0]))
    loss_2 = self.evaluate(agent.adaptive_kl_loss([100.0]))
    self.assertLess(loss_1, loss_2)

  def testUpdateAdaptiveKlBeta(self):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        fc_layer_params=None)
    value_net = value_network.ValueNetwork(
        self._time_step_spec.observation, fc_layer_params=None)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        initial_adaptive_kl_beta=1.0,
        adaptive_kl_target=10.0,
        adaptive_kl_tolerance=0.5,
    )

    self.evaluate(tf.compat.v1.initialize_all_variables())

    # When KL is target kl, beta should not change.
    update_adaptive_kl_beta_fn = common.function(agent.update_adaptive_kl_beta)
    beta_0 = update_adaptive_kl_beta_fn([10.0])
    expected_beta_0 = 1.0
    self.assertEqual(expected_beta_0, self.evaluate(beta_0))

    # When KL is large, beta should increase.
    beta_1 = update_adaptive_kl_beta_fn([100.0])
    expected_beta_1 = 1.5
    self.assertEqual(expected_beta_1, self.evaluate(beta_1))

    # When KL is small, beta should decrease.
    beta_2 = update_adaptive_kl_beta_fn([1.0])
    expected_beta_2 = 1.0
    self.assertEqual(expected_beta_2, self.evaluate(beta_2))

  def testPolicy(self):
    value_net = value_network.ValueNetwork(
        self._time_step_spec.observation, fc_layer_params=None)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=value_net)
    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=1)
    action_step = agent.policy.action(time_steps)
    actions = action_step.action
    self.assertEqual(actions.shape.as_list(), [1, 1])
    self.evaluate(tf.compat.v1.initialize_all_variables())
    _ = self.evaluate(actions)

  def testNormalizeAdvantages(self):
    advantages = np.array([1.1, 3.2, -1.5, 10.9, 5.6])
    mean = np.sum(advantages) / float(len(advantages))
    variance = np.sum(np.square(advantages - mean)) / float(len(advantages))
    stdev = np.sqrt(variance)
    expected_advantages = (advantages - mean) / stdev
    normalized_advantages = ppo_agent._normalize_advantages(
        tf.constant(advantages, dtype=tf.float32), variance_epsilon=0.0)
    self.assertAllClose(expected_advantages,
                        self.evaluate(normalized_advantages))

  def testAgentDoesNotFailWhenNestedObservationActionAndDebugSummaries(self):
    summary_writer = tf.compat.v2.summary.create_file_writer(FLAGS.test_tmpdir,
                                                             flush_millis=10000)
    summary_writer.set_as_default()

    nested_obs_spec = (self._obs_spec, self._obs_spec, {
        'a': self._obs_spec,
        'b': self._obs_spec,
    })
    nested_time_spec = ts.time_step_spec(nested_obs_spec)

    nested_act_spec = (self._action_spec, {
        'c': self._action_spec,
        'd': self._action_spec
    })

    class NestedActorNet(network.DistributionNetwork):

      def __init__(self, dummy_model):
        output_spec = (dummy_model.output_spec, {
            'c': dummy_model.output_spec,
            'd': dummy_model.output_spec,
        })
        super(NestedActorNet, self).__init__(
            dummy_model.input_tensor_spec, (),
            output_spec=output_spec,
            name='NestedActorNet')
        self.dummy_model = dummy_model

      def call(self, *args, **kwargs):
        dummy_ans, _ = self.dummy_model(*args, **kwargs)
        return (dummy_ans, {'c': dummy_ans, 'd': dummy_ans}), ()

    dummy_model = DummyActorNet(nested_obs_spec, self._action_spec)
    agent = ppo_agent.PPOAgent(
        nested_time_spec,
        nested_act_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=NestedActorNet(dummy_model),
        value_net=DummyValueNet(nested_obs_spec),
        debug_summaries=True)

    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ], dtype=tf.float32)

    observations = (observations, observations, {
        'a': observations,
        'b': observations,
    })

    time_steps = ts.TimeStep(
        step_type=tf.constant([[1] * 3] * 2, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    actions = (actions, {
        'c': actions,
        'd': actions,
    })

    action_distribution_parameters = {
        'loc': tf.constant([[[0.0]] * 3] * 2, dtype=tf.float32),
        'scale': tf.constant([[[1.0]] * 3] * 2, dtype=tf.float32),
    }
    action_distribution_parameters = (action_distribution_parameters, {
        'c': action_distribution_parameters,
        'd': action_distribution_parameters,
    })

    policy_info = action_distribution_parameters

    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)

    agent.train(experience)


if __name__ == '__main__':
  tf.test.main()
