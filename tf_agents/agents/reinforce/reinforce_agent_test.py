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

"""Tests for TF Agents reinforce_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from absl.testing.absltest import mock

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils


class DummyActorNet(network.Network):

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               unbounded_actions=False,
               stateful=False):
    # When unbounded_actions=True, we skip the final tanh activation and the
    # action shift and scale. This allows us to compute the actor and critic
    # losses by hand more easily.
    # If stateful=True, the network state has the same shape as
    # `input_tensor_spec`. Otherwise it is empty.
    state_spec = (tf.TensorSpec(input_tensor_spec.shape, tf.float32)
                  if stateful else ())
    super(DummyActorNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name='DummyActorNet')
    single_action_spec = tf.nest.flatten(output_tensor_spec)[0]
    activation_fn = None if unbounded_actions else tf.nn.tanh
    self._output_tensor_spec = output_tensor_spec
    self._dummy_layers = [
        tf.keras.layers.Dense(
            single_action_spec.shape.num_elements() * 2,
            activation=activation_fn,
            kernel_initializer=tf.constant_initializer([[2, 1], [1, 1]]),
            bias_initializer=tf.constant_initializer(5),
        ),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    states = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    for layer in self._dummy_layers:
      states = layer(states)

    single_action_spec = tf.nest.flatten(self._output_tensor_spec)[0]
    # action_spec is TensorSpec([1], ...) so make sure there's an outer dim.
    actions = states[..., 0]
    stdevs = states[..., 1]
    actions = tf.reshape(actions, [-1] + single_action_spec.shape.as_list())
    stdevs = tf.reshape(stdevs, [-1] + single_action_spec.shape.as_list())
    actions = tf.nest.pack_sequence_as(self._output_tensor_spec, [actions])
    stdevs = tf.nest.pack_sequence_as(self._output_tensor_spec, [stdevs])

    distribution = nest_utils.map_structure_up_to(
        self._output_tensor_spec,
        tfp.distributions.MultivariateNormalDiag,
        actions,
        stdevs)
    return distribution, network_state


class DummyValueNet(network.Network):

  def __init__(self, observation_spec, name=None, outer_rank=1):
    super(DummyValueNet, self).__init__(observation_spec, (), 'DummyValueNet')
    self._outer_rank = outer_rank
    self._dummy_layers = [
        tf.keras.layers.Dense(
            1,
            kernel_initializer=tf.constant_initializer([2, 1]),
            bias_initializer=tf.constant_initializer([5]))
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


class ReinforceAgentTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ReinforceAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._obs_spec = tensor_spec.TensorSpec([2], tf.float32)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)

  def testCreateAgent(self):
    reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=False),
        optimizer=None,
    )

  def testCreateAgentWithValueNet(self):
    reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=False),
        value_network=DummyValueNet(self._obs_spec),
        value_estimation_loss_coef=0.5,
        optimizer=None,
    )

  def testPolicyGradientLoss(self):
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=None,
    )

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[0], [1]], dtype=tf.float32)
    actions_distribution = agent.collect_policy.distribution(
        time_steps).action
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)

    expected_loss = 10.983667373657227
    loss = agent.policy_gradient_loss(
        actions_distribution, actions, time_steps.is_last(), returns, 1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testPolicyGradientLossMultipleEpisodes(self):
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=None,
    )

    step_type = tf.constant(
        [ts.StepType.FIRST, ts.StepType.LAST, ts.StepType.FIRST,
         ts.StepType.LAST])
    reward = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    discount = tf.constant([1, 1, 1, 1], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=tf.float32)
    time_steps = ts.TimeStep(step_type, reward, discount, observations)

    actions = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)
    actions_distribution = agent.collect_policy.distribution(
        time_steps).action
    returns = tf.constant([1.9, 1.9, 1.0, 1.0], dtype=tf.float32)

    expected_loss = 5.140229225158691
    loss = agent.policy_gradient_loss(
        actions_distribution, actions, time_steps.is_last(), returns, 2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testMaskingRewardSingleEpisodeRewardOnFirst(self):
    # Test that policy_gradient_loss reacts correctly to rewards when there are:
    #   * A single MDP episode
    #   * Returns on the tf.StepType.FIRST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]
    # action      :   [0]       [1]
    # reward      :    3         0
    # ~is_boundary:    1         0
    # is_last     :    1         0
    # valid reward:   3*1       4*0
    #
    # The second action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.
    #
    # The expected_loss is > 0.0 in this case, only LAST should be excluded.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=None,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST])
    reward = tf.constant([3, 4], dtype=tf.float32)
    discount = tf.constant([1, 0], dtype=tf.float32)
    observations = tf.constant([[1, 2], [1, 2]], dtype=tf.float32)
    time_steps = ts.TimeStep(step_type, reward, discount, observations)

    actions = tf.constant([[0], [1]], dtype=tf.float32)
    actions_distribution = agent.collect_policy.distribution(
        time_steps).action
    returns = tf.constant([3.0, 0.0], dtype=tf.float32)

    # Returns on the StepType.FIRST should be counted.
    expected_loss = 10.8935775757
    loss = agent.policy_gradient_loss(
        actions_distribution, actions, time_steps.is_last(), returns, 1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testMaskingReturnSingleEpisodeRewardOnLast(self):
    # Test that policy_gradient_loss reacts correctly to rewards when there are:
    #   * A single MDP episode
    #   * Returns on the tf.StepType.LAST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]
    # action      :   [0]       [1]
    # reward      :    0         3
    # ~is_boundary:    1         0
    # is_last     :    1         0
    # valid reward:   0*1       3*0
    #
    # The second action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.  The first has a 0 reward.
    #
    # The expected_loss is 0.0 in this case.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=None,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST])
    reward = tf.constant([0, 3], dtype=tf.float32)
    discount = tf.constant([1, 0], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2]], dtype=tf.float32)
    time_steps = ts.TimeStep(step_type, reward, discount, observations)

    actions = tf.constant([[0], [1]], dtype=tf.float32)
    actions_distribution = agent.collect_policy.distribution(
        time_steps).action
    returns = tf.constant([0.0, 3.0], dtype=tf.float32)

    # Returns on the StepType.LAST should not be counted.
    expected_loss = 0.0
    loss = agent.policy_gradient_loss(
        actions_distribution, actions, time_steps.is_last(), returns, 1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testMaskingReturnMultipleEpisodesRewardOnFirst(self):
    # Test that policy_gradient_loss reacts correctly to rewards when there are:
    #   * Multiple MDP episodes
    #   * Returns on the tf.StepType.FIRST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F) -> (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]    [1, 2]    [1, 2]
    # action      :   [0]       [1]       [2]       [3]
    # reward      :    3         0         4         0
    # ~is_boundary:    1         0         1         0
    # is_last     :    1         0         1         0
    # valid reward:   3*1       0*0       4*1       0*0
    #
    # The second & fourth action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.
    #
    # The expected_loss is > 0.0 in this case, only LAST should be excluded.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=None,
    )

    step_type = tf.constant(
        [ts.StepType.FIRST, ts.StepType.LAST, ts.StepType.FIRST,
         ts.StepType.LAST])
    reward = tf.constant([3, 0, 4, 0], dtype=tf.float32)
    discount = tf.constant([1, 0, 1, 0], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=tf.float32)
    time_steps = ts.TimeStep(step_type, reward, discount, observations)

    actions = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)
    actions_distribution = agent.collect_policy.distribution(
        time_steps).action
    returns = tf.constant([3.0, 0.0, 4.0, 0.0], dtype=tf.float32)

    # Returns on the StepType.FIRST should be counted.
    expected_loss = 12.2091741562
    loss = agent.policy_gradient_loss(
        actions_distribution, actions, time_steps.is_last(), returns, 2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testMaskingReturnMultipleEpisodesRewardOnLast(self):
    # Test that policy_gradient_loss reacts correctly to returns when there are:
    #   * Multiple MDP episodes
    #   * Returns on the tf.StepType.LAST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F) -> (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]    [1, 2]    [1, 2]
    # action      :   [0]       [1]       [2]       [3]
    # reward      :    0         3         0         4
    # ~is_boundary:    1         0         1         0
    # is_last     :    1         0         1         0
    # valid reward:   0*1       3*0       0*1       4*0
    #
    # The second & fourth action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.
    #
    # The expected_loss is 0.0 in this case.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=None,
    )

    step_type = tf.constant(
        [ts.StepType.FIRST, ts.StepType.LAST, ts.StepType.FIRST,
         ts.StepType.LAST])
    reward = tf.constant([0, 3, 0, 4], dtype=tf.float32)
    discount = tf.constant([1, 0, 1, 0], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=tf.float32)
    time_steps = ts.TimeStep(step_type, reward, discount, observations)

    actions = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)
    actions_distribution = agent.collect_policy.distribution(
        time_steps).action
    returns = tf.constant([0.0, 3.0, 0.0, 4.0], dtype=tf.float32)

    # Returns on the StepType.LAST should not be counted.
    expected_loss = 0.0
    loss = agent.policy_gradient_loss(
        actions_distribution, actions, time_steps.is_last(), returns, 2)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.parameters(
      ([[[0.8, 0.2]]], [1],),
      ([[[0.8, 0.2]], [[0.3, 0.7]]], [0.5, 0.5],),
  )
  def testEntropyLoss(self, probs, weights):
    probs = tf.convert_to_tensor(probs)
    distribution = tfp.distributions.Categorical(probs=probs)
    shape = probs.shape.as_list()
    action_spec = tensor_spec.TensorSpec(shape[2:-1], dtype=tf.int32)
    expected = tf.reduce_mean(
        -tf.reduce_mean(distribution.entropy()) * weights)
    actual = reinforce_agent._entropy_loss(
        distribution, action_spec, weights)
    self.assertAlmostEqual(self.evaluate(actual), self.evaluate(expected),
                           places=4)

  def testValueEstimationLoss(self):
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=False),
        value_network=DummyValueNet(self._obs_spec),
        value_estimation_loss_coef=0.5,
        optimizer=None,
    )

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    returns = tf.constant([1.9, 1.0], dtype=tf.float32)
    value_preds, _ = agent._value_network(time_steps.observation,
                                          time_steps.step_type)

    expected_loss = 123.20500
    loss = agent.value_estimation_loss(
        value_preds, returns, 1)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  def testTrainMaskingRewardSingleBanditEpisode(self):
    # Test that train reacts correctly to experience when there is only a
    # single Bandit episode.  Bandit episodes are encoded differently than
    # MDP episodes.  They have only a single transition with
    # step_type=StepType.FIRST and next_step_type=StepType.LAST.
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L)
    # observation : [1, 2]
    # action      :   [0]
    # reward      :    3
    # ~is_boundary:    0
    # is_last     :    1
    # valid reward:   3*1
    #
    # The single bandit transition is valid and not masked.
    #
    # The expected_loss is > 0.0 in this case, matching the expected_loss of the
    # testMaskingRewardSingleEpisodeRewardOnFirst policy_gradient_loss test.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST])
    next_step_type = tf.constant([ts.StepType.LAST])
    reward = tf.constant([3], dtype=tf.float32)
    discount = tf.constant([0], dtype=tf.float32)
    observations = tf.constant([[1, 2]], dtype=tf.float32)
    actions = tf.constant([[0]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards should be counted.
    expected_loss = 10.8935775757

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)

  def testTrainMaskingRewardMultipleBanditEpisodes(self):
    # Test that train reacts correctly to experience when there are multiple
    # Bandit episodes.  Bandit episodes are encoded differently than
    # MDP episodes.  They (each) have only a single transition with
    # step_type=StepType.FIRST and next_step_type=StepType.LAST.  This test
    # helps ensure that LAST->FIRST->LAST transitions are handled correctly.
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (F, L)
    # observation : [1, 2]    [1, 2]
    # action      :   [0]       [2]
    # reward      :    3         4
    # ~is_boundary:    0         0
    # is_last     :    1         1
    # valid reward:   3*1       4*1
    #
    # All bandit transitions are valid and none are masked.
    #
    # The expected_loss is > 0.0 in this case, matching the expected_loss of the
    # testMaskingRewardMultipleEpisodesRewardOnFirst policy_gradient_loss test.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.FIRST])
    next_step_type = tf.constant([ts.StepType.LAST, ts.StepType.LAST])
    reward = tf.constant([3, 4], dtype=tf.float32)
    discount = tf.constant([0, 0], dtype=tf.float32)
    observations = tf.constant([[1, 2], [1, 2]], dtype=tf.float32)
    actions = tf.constant([[0], [2]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards on the StepType.FIRST should be counted.
    expected_loss = 12.2091741562

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)

  def testTrainMaskingRewardSingleEpisodeRewardOnFirst(self):
    # Test that train reacts correctly to experience when there are:
    #   * A single MDP episode
    #   * Rewards on the tf.StepType.FIRST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]
    # action      :   [0]       [1]
    # reward      :    3         4
    # ~is_boundary:    1         0
    # is_last     :    1         0
    # valid reward:   3*1       4*0
    #
    # The second action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.
    #
    # The expected_loss is > 0.0 in this case, matching the expected_loss of the
    # testMaskingRewardSingleEpisodeRewardOnFirst policy_gradient_loss test.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST])
    next_step_type = tf.constant([ts.StepType.LAST, ts.StepType.FIRST])
    reward = tf.constant([3, 4], dtype=tf.float32)
    discount = tf.constant([1, 0], dtype=tf.float32)
    observations = tf.constant([[1, 2], [1, 2]], dtype=tf.float32)
    actions = tf.constant([[0], [1]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards on the StepType.FIRST should be counted.
    expected_loss = 10.8935775757
    expected_policy_gradient_loss = 10.8935775757
    expected_policy_network_regularization_loss = 0
    expected_entropy_regularization_loss = 0
    expected_value_estimation_loss = 0
    expected_value_network_regularization_loss = 0

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)
    self.assertAllClose(loss_info.extra.policy_gradient_loss,
                        expected_policy_gradient_loss)
    self.assertAllClose(loss_info.extra.policy_network_regularization_loss,
                        expected_policy_network_regularization_loss)
    self.assertAllClose(loss_info.extra.entropy_regularization_loss,
                        expected_entropy_regularization_loss)
    self.assertAllClose(loss_info.extra.value_estimation_loss,
                        expected_value_estimation_loss)
    self.assertAllClose(loss_info.extra.value_network_regularization_loss,
                        expected_value_network_regularization_loss)

  def testTrainMaskingRewardSingleEpisodeRewardOnLast(self):
    # Test that train reacts correctly to experience when there are:
    #   * A single MDP episode
    #   * Rewards on the tf.StepType.LAST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]
    # action      :   [0]       [1]
    # reward      :    0         3
    # ~is_boundary:    1         0
    # is_last     :    1         0
    # valid reward:   0*1       3*0
    #
    # The second action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.  The first has a 0 reward.
    #
    # The expected_loss is = 0.0 in this case.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST])
    next_step_type = tf.constant([ts.StepType.LAST, ts.StepType.FIRST])
    reward = tf.constant([0, 3], dtype=tf.float32)
    discount = tf.constant([1, 0], dtype=tf.float32)
    observations = tf.constant([[1, 2], [1, 2]], dtype=tf.float32)

    actions = tf.constant([[0], [1]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards on the StepType.LAST should not be counted.
    expected_loss = 0.0

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)

  def testTrainMaskingRewardMultipleEpisodesRewardOnFirst(self):
    # Test that train reacts correctly to experience when there are:
    #   * Multiple MDP episodes
    #   * Rewards on the tf.StepType.FIRST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F) -> (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]    [1, 2]    [1, 2]
    # action      :   [0]       [1]       [2]       [3]
    # reward      :    3         0         4         0
    # ~is_boundary:    1         0         1         0
    # is_last     :    1         0         1         0
    # valid reward:   3*1       0*0       4*1       0*0
    #
    # The second & fourth action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.
    #
    # The expected_loss is > 0.0 in this case, matching the expected_loss of the
    # testMaskingRewardMultipleEpisodesRewardOnFirst policy_gradient_loss test.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST,
                             ts.StepType.FIRST, ts.StepType.LAST])
    next_step_type = tf.constant([ts.StepType.LAST, ts.StepType.FIRST,
                                  ts.StepType.LAST, ts.StepType.FIRST])
    reward = tf.constant([3, 0, 4, 0], dtype=tf.float32)
    discount = tf.constant([1, 0, 1, 0], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=tf.float32)
    actions = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards on the StepType.FIRST should be counted.
    expected_loss = 12.2091741562

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)

  def testTrainMaskingPartialEpisodeMultipleEpisodesRewardOnFirst(self):
    # Test that train reacts correctly to experience when there are:
    #   * Multiple MDP episodes
    #   * Rewards on the tf.StepType.FIRST transitions
    #   * Partial episode at end of experience
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F) -> (F, M) -> (M, M)
    # observation : [1, 2]    [1, 2]    [1, 2]    [1, 2]
    # action      :   [0]       [1]       [2]       [3]
    # reward      :    3         0         4         0
    # ~is_boundary:    1         0         1         1
    # is_last     :    1         0         0         0
    # valid reward:   3*1       0*0       4*0       0*0
    #
    # The second action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.  The third & fourth transitions
    # should get masked out for everything due to it being an incomplete episode
    # (notice there is no trailing step_type=(F,L)).
    #
    # The expected_loss is > 0.0 in this case, matching the expected_loss of the
    # testMaskingRewardSingleEpisodeRewardOnFirst policy_gradient_loss test,
    # because the partial second episode should be masked out.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST,
                             ts.StepType.FIRST, ts.StepType.MID])
    next_step_type = tf.constant([ts.StepType.LAST, ts.StepType.FIRST,
                                  ts.StepType.MID, ts.StepType.MID])
    reward = tf.constant([3, 0, 4, 0], dtype=tf.float32)
    discount = tf.constant([1, 0, 1, 0], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=tf.float32)
    actions = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards on the StepType.FIRST should be counted.
    expected_loss = 10.8935775757

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)

  def testTrainMaskingRewardMultipleEpisodesRewardOnLast(self):
    # Test that train reacts correctly to experience when there are:
    #   * Multiple MDP episodes
    #   * Rewards on the tf.StepType.LAST transitions
    #
    # F, L, M = ts.StepType.{FIRST, MID, LAST} in the chart below.
    #
    # Experience looks like this:
    # Trajectories: (F, L) -> (L, F) -> (F, L) -> (L, F)
    # observation : [1, 2]    [1, 2]    [1, 2]    [1, 2]
    # action      :   [0]       [1]       [2]       [3]
    # reward      :    0         3         0         4
    # ~is_boundary:    1         0         1         0
    # is_last     :    1         0         1         0
    # valid reward:   0*1       3*0       0*1       4*0
    #
    # The second & fourth action & reward should be masked out due to being on a
    # boundary (step_type=(L, F)) transition.
    #
    # The expected_loss is = 0.0 in this case.
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=True),
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        use_advantage_loss=False,
        normalize_returns=False,
    )

    step_type = tf.constant([ts.StepType.FIRST, ts.StepType.LAST,
                             ts.StepType.FIRST, ts.StepType.LAST])
    next_step_type = tf.constant([ts.StepType.LAST, ts.StepType.FIRST,
                                  ts.StepType.LAST, ts.StepType.FIRST])
    reward = tf.constant([0, 3, 0, 4], dtype=tf.float32)
    discount = tf.constant([1, 0, 1, 0], dtype=tf.float32)
    observations = tf.constant(
        [[1, 2], [1, 2], [1, 2], [1, 2]], dtype=tf.float32)
    actions = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)

    experience = nest_utils.batch_nested_tensors(trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount))

    # Rewards on the StepType.LAST should be counted.
    expected_loss = 0.0

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_info = self.evaluate(loss)
    self.assertAllClose(loss_info.loss, expected_loss)

  def testPolicy(self):
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=False),
        optimizer=None,
    )
    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = agent.policy.action(time_steps).action
    self.assertEqual(actions.shape.as_list(), [1, 1])
    self.evaluate(tf.compat.v1.global_variables_initializer())
    action_values = self.evaluate(actions)
    tf.nest.map_structure(
        lambda v, s: self.assertAllInRange(v, s.minimum, s.maximum),
        action_values, self._action_spec)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def testGetInitialPolicyState(self, stateful):
    if tf.test.is_built_with_gpu_support():  # b/237573967
      self.skipTest('Test is only applicable on GPU')
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=False,
            stateful=stateful),
        optimizer=None,
    )
    observations = tf.constant([[1, 2]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=3)
    initial_state = reinforce_agent._get_initial_policy_state(
        agent.collect_policy, time_steps)
    if stateful:
      self.assertAllEqual(self.evaluate(initial_state),
                          self.evaluate(tf.zeros((3, 2), dtype=tf.float32)))
    else:
      self.assertEqual(initial_state, ())

  def testTrainWithRnn(self):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        self._obs_spec,
        self._action_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        conv_layer_params=None,
        lstm_size=(40,))

    counter = common.create_variable('test_train_counter')
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        train_step_counter=counter
    )

    batch_size = 5
    observations = tf.constant(
        [[[1, 2], [3, 4], [5, 6]]] * batch_size, dtype=tf.float32)
    time_steps = ts.TimeStep(
        step_type=tf.constant([[1, 1, 2]] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        observation=observations)
    actions = tf.constant([[[0], [1], [1]]] * batch_size, dtype=tf.float32)

    experience = trajectory.Trajectory(
        time_steps.step_type, observations, actions, (),
        time_steps.step_type, time_steps.reward, time_steps.discount)

    # Force variable creation.
    agent.policy.variables()

    if tf.executing_eagerly():
      loss = lambda: agent.train(experience)
    else:
      loss = agent.train(experience)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertEqual(self.evaluate(counter), 0)
    self.evaluate(loss)
    self.assertEqual(self.evaluate(counter), 1)

  def testTrainWithRnnTransitions(self):
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        self._obs_spec,
        self._action_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        conv_layer_params=None,
        lstm_size=(40,))

    counter = common.create_variable('test_train_counter')
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=actor_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        train_step_counter=counter
    )

    batch_size = 5
    observations = tf.constant(
        [[[1, 2], [3, 4], [5, 6]]] * batch_size, dtype=tf.float32)
    time_steps = ts.TimeStep(
        step_type=tf.constant([[1, 1, 1]] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        observation=observations)
    actions = policy_step.PolicyStep(
        tf.constant([[[0], [1], [1]]] * batch_size, dtype=tf.float32))
    next_time_steps = ts.TimeStep(
        step_type=tf.constant([[1, 1, 2]] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        discount=tf.constant([[1] * 3] * batch_size, dtype=tf.float32),
        observation=observations)

    experience = trajectory.Transition(time_steps, actions, next_time_steps)

    agent.initialize()
    agent.train(experience)

  @parameterized.parameters(
      (False,), (True,)
  )
  def testWithAdvantageFn(self, with_value_network):
    advantage_fn = mock.Mock(
        side_effect=lambda returns, _: returns)

    value_network = (DummyValueNet(self._obs_spec) if with_value_network
                     else None)
    agent = reinforce_agent.ReinforceAgent(
        self._time_step_spec,
        self._action_spec,
        actor_network=DummyActorNet(
            self._obs_spec, self._action_spec, unbounded_actions=False),
        value_network=value_network,
        advantage_fn=advantage_fn,
        optimizer=None,
    )

    step_type = tf.constant([[ts.StepType.FIRST, ts.StepType.LAST,
                              ts.StepType.FIRST, ts.StepType.LAST]])
    next_step_type = tf.constant([[ts.StepType.LAST, ts.StepType.FIRST,
                                   ts.StepType.LAST, ts.StepType.FIRST]])
    reward = tf.constant([[0, 0, 0, 0]], dtype=tf.float32)
    discount = tf.constant([[1, 1, 1, 1]], dtype=tf.float32)
    observations = tf.constant(
        [[[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=tf.float32)
    actions = tf.constant([[[0], [1], [2], [3]]], dtype=tf.float32)

    experience = trajectory.Trajectory(
        step_type, observations, actions, (), next_step_type, reward, discount)

    agent.total_loss(experience, reward, None)

    advantage_fn.assert_called_once()


if __name__ == '__main__':
  tf.test.main()
