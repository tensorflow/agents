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
"""Tests for TF Agents ppo_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
from absl.testing.absltest import mock

import numpy as np
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import random_tf_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS


class DummyActorNet(network.DistributionNetwork):

  def __init__(self,
               input_spec,
               action_spec,
               preprocessing_layers=None,
               name=None):
    output_spec = self._get_normal_distribution_spec(action_spec)
    super(DummyActorNet, self).__init__(
        input_spec, (), output_spec=output_spec, name='DummyActorNet')
    self._action_spec = action_spec
    self._flat_action_spec = tf.nest.flatten(self._action_spec)[0]

    self._dummy_layers = (preprocessing_layers or []) + [
        tf.keras.layers.Dense(
            self._flat_action_spec.shape.num_elements() * 2,
            kernel_initializer=tf.compat.v1.initializers.constant([[2.0, 1.0],
                                                                   [1.0, 1.0]]),
            bias_initializer=tf.compat.v1.initializers.constant([5.0, 5.0]),
            activation=None,
        )
    ]

  def _get_normal_distribution_spec(self, sample_spec):
    is_multivariate = sample_spec.shape.ndims > 0
    input_param_shapes = tfp.distributions.Normal.param_static_shapes(
        sample_spec.shape)
    input_param_spec = tf.nest.map_structure(
        lambda tensor_shape: tensor_spec.TensorSpec(  # pylint: disable=g-long-lambda
            shape=tensor_shape,
            dtype=sample_spec.dtype),
        input_param_shapes)

    def distribution_builder(*args, **kwargs):
      if is_multivariate:
        # For backwards compatibility, and because MVNDiag does not support
        # `param_static_shapes`, even when using MVNDiag the spec
        # continues to use the terms 'loc' and 'scale'.  Here we have to massage
        # the construction to use 'scale' for kwarg 'scale_diag'.  Since they
        # have the same shape and dtype expectationts, this is okay.
        kwargs = kwargs.copy()
        kwargs['scale_diag'] = kwargs['scale']
        del kwargs['scale']
        return tfp.distributions.MultivariateNormalDiag(*args, **kwargs)
      else:
        return tfp.distributions.Normal(*args, **kwargs)

    return distribution_spec.DistributionSpec(
        distribution_builder, input_param_spec, sample_spec=sample_spec)

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]

    # Calls coming from agent.train() has a time dimension. Direct loss calls
    # may not have a time dimension. It order to make BatchSquash work, we need
    # to specify the outer dimension properly.
    has_time_dim = nest_utils.get_outer_rank(inputs,
                                             self.input_tensor_spec) == 2
    outer_rank = 2 if has_time_dim else 1
    batch_squash = network_utils.BatchSquash(outer_rank)
    hidden_state = batch_squash.flatten(hidden_state)

    for layer in self._dummy_layers:
      hidden_state = layer(hidden_state)

    actions, stdevs = tf.split(hidden_state, 2, axis=1)
    actions = batch_squash.unflatten(actions)
    stdevs = batch_squash.unflatten(stdevs)
    actions = tf.nest.pack_sequence_as(self._action_spec, [actions])
    stdevs = tf.nest.pack_sequence_as(self._action_spec, [stdevs])

    return self.output_spec.build_distribution(
        loc=actions, scale=stdevs), network_state


class DummyValueNet(network.Network):

  def __init__(self,
               observation_spec,
               preprocessing_layers=None,
               name=None,
               outer_rank=1):
    super(DummyValueNet, self).__init__(observation_spec, (), 'DummyValueNet')
    self._outer_rank = outer_rank
    self._dummy_layers = (preprocessing_layers or []) + [
        tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.compat.v1.initializers.constant([2, 1]),
            bias_initializer=tf.compat.v1.initializers.constant([5]))
    ]

  def call(self, inputs, step_type=None, network_state=()):
    del step_type
    hidden_state = tf.cast(tf.nest.flatten(inputs), tf.float32)[0]
    batch_squash = network_utils.BatchSquash(self._outer_rank)
    hidden_state = batch_squash.flatten(hidden_state)
    for layer in self._dummy_layers:
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


def _create_joint_actor_value_networks(observation_spec, action_spec):
  shared_layers = [
      tf.keras.layers.Dense(
          tf.nest.flatten(observation_spec)[0].shape.num_elements(),
          kernel_initializer=tf.compat.v1.initializers.constant([[3.0, 1.0],
                                                                 [1.0, 1.0]]),
          bias_initializer=tf.compat.v1.initializers.constant([5.0, 5.0]),
          activation=None,
      )
  ]
  actor_net = DummyActorNet(observation_spec, action_spec, shared_layers)
  value_net = DummyValueNet(observation_spec, shared_layers)
  return actor_net, value_net


def _default():
  return tf.distribute.get_strategy()


def _one_device():
  return tf.distribute.OneDeviceStrategy('/cpu:0')


def _mirrored():
  return tf.distribute.MirroredStrategy()


class PPOAgentTest(parameterized.TestCase, test_utils.TestCase):

  def setUp(self):
    super(PPOAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

    # Ensure that there are 4 CPU devices available for the mirrored strategy.
    physical_devices = tf.config.list_physical_devices('CPU')
    try:
      tf.config.set_logical_device_configuration(physical_devices[0], [
          tf.config.LogicalDeviceConfiguration(),
          tf.config.LogicalDeviceConfiguration(),
          tf.config.LogicalDeviceConfiguration(),
          tf.config.LogicalDeviceConfiguration(),
      ])
      logical_devices = tf.config.list_logical_devices('CPU')
      assert len(logical_devices) == 4
    except RuntimeError:
      # Cannot modify logical devices once initialized.
      pass

  def testCreateAgent(self):
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=DummyActorNet(self._obs_spec, self._action_spec),
        value_net=DummyValueNet(self._obs_spec),
        check_numerics=True)
    agent.initialize()

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testComputeAdvantagesNoGae(self, strategy_fn):
    with strategy_fn().scope():
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(self._obs_spec, self._action_spec),
          value_net=DummyValueNet(self._obs_spec),
          normalize_observations=False,
          use_gae=False)
      agent.initialize()
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

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testComputeAdvantagesWithGae(self, strategy_fn):
    gae_lambda = 0.95
    with strategy_fn().scope():
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(
              self._obs_spec,
              self._action_spec,
          ),
          value_net=DummyValueNet(self._obs_spec),
          normalize_observations=False,
          use_gae=True,
          lambda_value=gae_lambda)
      agent.initialize()
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

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testSequencePreprocess(self, strategy_fn):
    with strategy_fn().scope():
      counter = common.create_variable('test_train_counter')
      batch_size = 2
      n_time_steps = 3
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(
              self._obs_spec,
              self._action_spec,
          ),
          value_net=DummyValueNet(self._obs_spec),
          normalize_observations=False,
          num_epochs=1,
          use_gae=False,
          use_td_lambda_return=False,
          compute_value_and_advantage_in_train=False,
          train_step_counter=counter)
      agent.initialize()
    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)

    mid_time_step_val = ts.StepType.MID.tolist()
    time_steps = ts.TimeStep(
        step_type=tf.constant(
            [[mid_time_step_val] * n_time_steps] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * n_time_steps] * batch_size, dtype=tf.float32),
        discount=tf.constant(
            [[1] * n_time_steps] * batch_size, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    old_action_distribution_parameters = {
        'loc':
            tf.constant(
                [[[0.0]] * n_time_steps] * batch_size, dtype=tf.float32),
        'scale':
            tf.constant(
                [[[1.0]] * n_time_steps] * batch_size, dtype=tf.float32),
    }

    value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                              dtype=tf.float32)
    policy_info = {
        'dist_params': old_action_distribution_parameters,
        'value_prediction': value_preds,
    }
    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)

    returned_experience = agent.preprocess_sequence(experience)
    self.evaluate(tf.compat.v1.initialize_all_variables())

    self.assertAllClose(observations, returned_experience.observation)
    self.assertAllClose(actions, returned_experience.action)

    expected_value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                                       dtype=tf.float32)
    (_, _, next_time_steps) = trajectory.to_transition(experience)
    expected_returns, expected_advantages = agent.compute_return_and_advantage(
        next_time_steps, expected_value_preds)
    self.assertAllClose(old_action_distribution_parameters,
                        returned_experience.policy_info['dist_params'])
    self.assertEqual((batch_size, n_time_steps),
                     returned_experience.policy_info['return'].shape)
    self.assertAllClose(expected_returns,
                        returned_experience.policy_info['return'][:, :-1])
    self.assertEqual((batch_size, n_time_steps),
                     returned_experience.policy_info['advantage'].shape)
    self.assertAllClose(expected_advantages,
                        returned_experience.policy_info['advantage'][:, :-1])

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testSequencePreprocessNotBatched(self, strategy_fn):
    with strategy_fn().scope():
      counter = common.create_variable('test_train_counter')
      n_time_steps = 3
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(
              self._obs_spec,
              self._action_spec,
          ),
          value_net=DummyValueNet(self._obs_spec),
          normalize_observations=False,
          num_epochs=1,
          use_gae=False,
          use_td_lambda_return=False,
          compute_value_and_advantage_in_train=False,
          train_step_counter=counter)
      agent.initialize()
    observations = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

    mid_time_step_val = ts.StepType.MID.tolist()
    time_steps = ts.TimeStep(
        step_type=tf.constant(
            [mid_time_step_val] * n_time_steps, dtype=tf.int32),
        reward=tf.constant([1] * n_time_steps, dtype=tf.float32),
        discount=tf.constant([1] * n_time_steps, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[0], [1], [1]], dtype=tf.float32)

    old_action_distribution_parameters = {
        'loc': tf.constant([[0.0]] * n_time_steps, dtype=tf.float32),
        'scale': tf.constant([[1.0]] * n_time_steps, dtype=tf.float32),
    }

    value_preds = tf.constant([9., 15., 21.], dtype=tf.float32)
    policy_info = {
        'dist_params': old_action_distribution_parameters,
        'value_prediction': value_preds,
    }
    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)

    returned_experience = agent.preprocess_sequence(experience)
    self.evaluate(tf.compat.v1.initialize_all_variables())

    self.assertAllClose(observations, returned_experience.observation)
    self.assertAllClose(actions, returned_experience.action)

    self.assertAllClose(old_action_distribution_parameters,
                        returned_experience.policy_info['dist_params'])
    self.assertEqual(n_time_steps,
                     returned_experience.policy_info['return'].shape)
    self.assertAllClose([40.4821, 30.79],
                        returned_experience.policy_info['return'][:-1])
    self.assertEqual(n_time_steps,
                     returned_experience.policy_info['advantage'].shape)
    self.assertAllClose([31.482101, 15.790001],
                        returned_experience.policy_info['advantage'][:-1])

  @parameterized.named_parameters(
      ('DefaultOneEpochValueInTrain', _default, 1, True, True),
      ('DefaultFiveEpochsValueInCollect', _default, 5, False, False),
      ('DefaultIncompEpisodesReturnNonZeroLoss', _default, 1, False, True),
      ('OneDeviceOneEpochValueInTrain', _one_device, 1, True, True),
      ('OneDeviceFiveEpochsValueInCollect', _one_device, 5, False, False),
      ('OneDeviceIncompEpisodesReturnNonZeroLoss', _one_device, 1, False, True),
      ('MirroredOneEpochValueInTrain', _mirrored, 1, True, True),
      ('MirroredFiveEpochsValueInCollect', _mirrored, 5, False, False),
      ('MirroredIncompEpisodesReturnNonZeroLoss', _mirrored, 1, False, True))
  def testTrain(self, strategy_fn, num_epochs, use_td_lambda_return,
                compute_value_and_advantage_in_train):
    # Mock the build_train_op to return an op for incrementing this counter.
    with strategy_fn().scope():
      counter = common.create_variable('test_train_counter')
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(
              self._obs_spec,
              self._action_spec,
          ),
          value_net=DummyValueNet(self._obs_spec),
          normalize_observations=False,
          num_epochs=num_epochs,
          use_gae=use_td_lambda_return,
          use_td_lambda_return=use_td_lambda_return,
          compute_value_and_advantage_in_train=compute_value_and_advantage_in_train,
          train_step_counter=counter)
      agent.initialize()
    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)

    mid_time_step_val = ts.StepType.MID.tolist()
    time_steps = ts.TimeStep(
        step_type=tf.constant([[mid_time_step_val] * 3] * 2, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    action_distribution_parameters = {
        'loc': tf.constant([[[0.0]] * 3] * 2, dtype=tf.float32),
        'scale': tf.constant([[[1.0]] * 3] * 2, dtype=tf.float32),
    }
    value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                              dtype=tf.float32)

    policy_info = {
        'dist_params': action_distribution_parameters,
    }
    if not compute_value_and_advantage_in_train:
      policy_info['value_prediction'] = value_preds
    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)
    if not compute_value_and_advantage_in_train:
      experience = agent._preprocess(experience)

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    # Assert that counter starts out at zero.
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertEqual(0, self.evaluate(counter))
    loss_type = self.evaluate(loss)
    loss_numpy = loss_type.loss

    # Assert that loss is not zero as we are training in a non-episodic env.
    self.assertNotEqual(
        loss_numpy,
        0.0,
        msg=('Loss is exactly zero, looks like no training '
             'was performed due to incomplete episodes.'))

    # Assert that train_op ran increment_counter num_epochs times.
    self.assertEqual(num_epochs, self.evaluate(counter))

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testGetEpochLoss(self, strategy_fn):
    with strategy_fn().scope():
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
      agent.initialize()
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

    loss_info = agent.get_loss(
        time_steps,
        actions,
        sample_action_log_probs,
        returns,
        advantages,
        sample_action_distribution_parameters,
        weights,
        train_step,
        debug_summaries=False)

    self.evaluate(tf.compat.v1.global_variables_initializer())
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
    self.assertAllClose(
        expected_l2_loss, l2_regularization_loss, atol=0.001, rtol=0.001)
    self.assertAllClose(expected_ent_loss, entropy_reg_loss)
    self.assertAllClose(expected_kl_penalty_loss, kl_penalty_loss)

  @parameterized.named_parameters(
      ('DefaultIsZero', _default, 0), ('DefaultNotZero', _default, 1),
      ('OneDeviceIsZero', _one_device, 0), ('OneDeviceNotZero', _one_device, 1),
      ('MirroredIsZero', _mirrored, 0), ('MirroredNotZero', _mirrored, 1))
  def testL2RegularizationLoss(self, strategy_fn, not_zero):
    l2_reg = 1e-4 * not_zero
    with strategy_fn().scope():
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
      agent.initialize()

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
                               advantages, current_policy_distribution, weights)
    agent.value_estimation_loss(time_steps, returns, weights)

    # Now request L2 regularization loss.
    # Value function weights are [2, 1], actor net weights are [2, 1, 1, 1].
    expected_loss = l2_reg * ((2**2 + 1) + (2**2 + 1 + 1 + 1))
    # Make sure the network is built before we try to get variables.
    agent.policy.action(
        tensor_spec.sample_spec_nest(self._time_step_spec, outer_dims=(2,)))
    loss = agent.l2_regularization_loss()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.named_parameters(
      ('DefaultIsZero', _default, 0), ('DefaultNotZero', _default, 1),
      ('OneDeviceIsZero', _one_device, 0), ('OneDeviceNotZero', _one_device, 1),
      ('MirroredIsZero', _mirrored, 0), ('MirroredNotZero', _mirrored, 1))
  def testL2RegularizationLossWithSharedVariables(self, strategy_fn, not_zero):
    policy_l2_reg = 4e-4 * not_zero
    value_function_l2_reg = 2e-4 * not_zero
    shared_vars_l2_reg = 1e-4 * not_zero
    with strategy_fn().scope():
      actor_net, value_net = _create_joint_actor_value_networks(
          self._obs_spec, self._action_spec)
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=actor_net,
          value_net=value_net,
          normalize_observations=False,
          policy_l2_reg=policy_l2_reg,
          value_function_l2_reg=value_function_l2_reg,
          shared_vars_l2_reg=shared_vars_l2_reg,
      )
      agent.initialize()

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
                               advantages, current_policy_distribution, weights)
    agent.value_estimation_loss(time_steps, returns, weights)

    # Now request L2 regularization loss.
    # Value function weights are [2, 1], actor net weights are [2, 1, 1, 1],
    # shared weights are [3, 1, 1, 1].
    expected_loss = value_function_l2_reg * (2**2 + 1) + policy_l2_reg * (
        2**2 + 1 + 1 + 1) + shared_vars_l2_reg * (3**2 + 1 + 1 + 1)
    # Make sure the network is built before we try to get variables.
    agent.policy.action(
        tensor_spec.sample_spec_nest(self._time_step_spec, outer_dims=(2,)))
    loss = agent.l2_regularization_loss()

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.named_parameters(
      ('DefaultIsZero', _default, 0), ('DefaultNotZero', _default, 1),
      ('OneDeviceIsZero', _one_device, 0), ('OneDeviceNotZero', _one_device, 1),
      ('MirroredIsZero', _mirrored, 0), ('MirroredNotZero', _mirrored, 1))
  def testEntropyRegularizationLoss(self, strategy_fn, not_zero):
    ent_reg = 0.1 * not_zero
    with strategy_fn().scope():
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(self._obs_spec, self._action_spec),
          value_net=DummyValueNet(self._obs_spec),
          normalize_observations=False,
          entropy_regularization=ent_reg,
      )
      agent.initialize()

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
                               advantages, current_policy_distribution, weights)
    agent.value_estimation_loss(time_steps, returns, weights)

    # Now request entropy regularization loss.
    # Action stdevs should be ~1.0, and mean entropy ~3.70111.
    expected_loss = -3.70111 * ent_reg
    loss = agent.entropy_regularization_loss(time_steps,
                                             current_policy_distribution,
                                             weights)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testValueEstimationLoss(self, strategy_fn):
    with strategy_fn().scope():
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          actor_net=DummyActorNet(self._obs_spec, self._action_spec),
          value_net=DummyValueNet(self._obs_spec),
          value_pred_loss_coef=1.0,
          normalize_observations=False,
      )
      agent.initialize()

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)
    weights = tf.ones_like(returns)

    expected_loss = 123.205
    loss = agent.value_estimation_loss(time_steps, returns, weights)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testPolicyGradientLoss(self, strategy_fn):
    with strategy_fn().scope():
      actor_net = DummyActorNet(self._obs_spec, self._action_spec)
      agent = ppo_agent.PPOAgent(
          self._time_step_spec,
          self._action_spec,
          tf.compat.v1.train.AdamOptimizer(),
          normalize_observations=False,
          normalize_rewards=False,
          actor_net=actor_net,
          value_net=DummyValueNet(self._obs_spec),
          importance_ratio_clipping=10.0,
      )
      agent.initialize()

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

    self.evaluate(tf.compat.v1.global_variables_initializer())
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
        'loc': tf.constant([[1.0], [1.0]], dtype=tf.float32),
        'scale': tf.constant([[1.0], [1.0]], dtype=tf.float32),
    }
    current_policy_distribution, unused_network_state = DummyActorNet(
        self._obs_spec, self._action_spec)(time_steps.observation,
                                           time_steps.step_type, ())
    weights = tf.ones_like(time_steps.discount)

    expected_kl_penalty_loss = 7.0

    kl_penalty_loss = agent.kl_penalty_loss(time_steps,
                                            action_distribution_parameters,
                                            current_policy_distribution,
                                            weights)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    kl_penalty_loss_ = self.evaluate(kl_penalty_loss)
    self.assertEqual(expected_kl_penalty_loss, kl_penalty_loss_)

  @parameterized.named_parameters(
      ('DefaultIsZero', _default, 0), ('DefaultNotZero', _default, 1),
      ('OneDeviceIsZero', _one_device, 0), ('OneDeviceNotZero', _one_device, 1),
      ('MirroredIsZero', _mirrored, 0), ('MirroredNotZero', _mirrored, 1))
  def testKlCutoffLoss(self, strategy_fn, not_zero):
    kl_cutoff_coef = 30.0 * not_zero
    with strategy_fn().scope():
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
      agent.initialize()
    kl_divergence = tf.constant([[1.5, -0.5, 6.5, -1.5, -2.3]],
                                dtype=tf.float32)
    expected_kl_cutoff_loss = kl_cutoff_coef * (.24**2)  # (0.74 - 0.5) ^ 2

    loss = agent.kl_cutoff_loss(kl_divergence)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose([loss_], [expected_kl_cutoff_loss])

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testAdaptiveKlLoss(self, strategy_fn):
    with strategy_fn().scope():
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
      agent.initialize()

    # Initialize variables
    self.evaluate(tf.compat.v1.global_variables_initializer())

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

  @parameterized.named_parameters(('Default', _default),
                                  ('OneDevice', _one_device),
                                  ('Mirrored', _mirrored))
  def testUpdateAdaptiveKlBeta(self, strategy_fn):
    with strategy_fn().scope():
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
      agent.initialize()

    self.evaluate(tf.compat.v1.global_variables_initializer())

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
    self.evaluate(tf.compat.v1.global_variables_initializer())
    _ = self.evaluate(actions)

  def testRNNTrain(self):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        lstm_size=(20,))
    value_net = value_rnn_network.ValueRnnNetwork(
        self._time_step_spec.observation,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        lstm_size=(10,))
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=1,
        train_step_counter=global_step,
    )
    # Use a random env, policy, and replay buffer to collect training data.
    random_env = random_tf_environment.RandomTFEnvironment(
        self._time_step_spec, self._action_spec, batch_size=1)
    collection_policy = random_tf_policy.RandomTFPolicy(
        self._time_step_spec,
        self._action_spec,
        info_spec=agent.collect_policy.info_spec)
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collection_policy.trajectory_spec, batch_size=1, max_length=7)
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        random_env,
        collection_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=1)

    # In graph mode: finish building the graph so the optimizer
    # variables are created.
    if not tf.executing_eagerly():
      _, _ = agent.train(experience=replay_buffer.gather_all())

    # Initialize.
    self.evaluate(agent.initialize())
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Train one step.
    self.assertEqual(0, self.evaluate(global_step))
    self.evaluate(collect_driver.run())
    self.evaluate(agent.train(experience=replay_buffer.gather_all()))
    self.assertEqual(1, self.evaluate(global_step))

  @parameterized.named_parameters([
      ('ValueCalculationInTrain', True),
      ('ValueCalculationInCollect', False),
  ])
  def testStatelessValueNetTrain(self, compute_value_and_advantage_in_train):
    counter = common.create_variable('test_train_counter')
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        self._time_step_spec.observation,
        self._action_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        lstm_size=(20,))
    value_net = value_network.ValueNetwork(
        self._time_step_spec.observation, fc_layer_params=None)
    agent = ppo_agent.PPOAgent(
        self._time_step_spec,
        self._action_spec,
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=1,
        train_step_counter=counter,
        compute_value_and_advantage_in_train=compute_value_and_advantage_in_train
    )
    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)

    mid_time_step_val = ts.StepType.MID.tolist()
    time_steps = ts.TimeStep(
        step_type=tf.constant([[mid_time_step_val] * 3] * 2, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * 2, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]], [[0], [1], [1]]], dtype=tf.float32)

    action_distribution_parameters = {
        'loc': tf.constant([[[0.0]] * 3] * 2, dtype=tf.float32),
        'scale': tf.constant([[[1.0]] * 3] * 2, dtype=tf.float32),
    }
    value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                              dtype=tf.float32)

    policy_info = {
        'dist_params': action_distribution_parameters,
    }
    if not compute_value_and_advantage_in_train:
      policy_info['value_prediction'] = value_preds
    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)
    if not compute_value_and_advantage_in_train:
      experience = agent._preprocess(experience)

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.initialize_all_variables())

    loss_type = self.evaluate(loss)
    loss_numpy = loss_type.loss
    # Assert that loss is not zero as we are training in a non-episodic env.
    self.assertNotEqual(
        loss_numpy,
        0.0,
        msg=('Loss is exactly zero, looks like no training '
             'was performed due to incomplete episodes.'))

  def testAgentDoesNotFailWhenNestedObservationActionAndDebugSummaries(self):
    summary_writer = tf.compat.v2.summary.create_file_writer(
        FLAGS.test_tmpdir, flush_millis=10000)
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

      def call(self, inputs, network_state, *args, **kwargs):
        dummy_ans, _ = self.dummy_model(
            inputs, network_state=network_state, *args, **kwargs)
        return (dummy_ans, {'c': dummy_ans, 'd': dummy_ans}), ()

    dummy_model = DummyActorNet(nested_obs_spec, self._action_spec)
    agent = ppo_agent.PPOAgent(
        nested_time_spec,
        nested_act_spec,
        tf.compat.v1.train.AdamOptimizer(),
        actor_net=NestedActorNet(dummy_model),
        value_net=DummyValueNet(nested_obs_spec),
        compute_value_and_advantage_in_train=False,
        debug_summaries=True)

    observations = tf.constant([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
                               dtype=tf.float32)

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

    value_preds = tf.constant([[9., 15., 21.], [9., 15., 21.]],
                              dtype=tf.float32)
    policy_info = {
        'dist_params': action_distribution_parameters,
        'value_prediction': value_preds,
    }

    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, policy_info,
                                       time_steps.step_type, time_steps.reward,
                                       time_steps.discount)
    experience = agent._preprocess(experience)

    agent.train(experience)


if __name__ == '__main__':
  tf.test.main()
