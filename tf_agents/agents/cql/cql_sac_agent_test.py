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

"""Tests for tf_agents.agents.cql.cql_sac_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.cql import cql_sac_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.utils import test_utils

# TODO(b/237700589): Remove this once the global flag is on.
tf.keras.backend.experimental.enable_tf_random_generator()


class _MockDistribution(object):

  def __init__(self, action):
    self._action = action

  def sample(self, num_samples=None, seed=None):
    del seed
    if not num_samples:
      return self._action

    actions = tf.tile(self._action, tf.constant([num_samples, 1]))
    actions = tf.reshape(
        actions, [num_samples, self._action.shape[0], self._action.shape[1]])
    return actions

  def log_prob(self, sample):
    return tf.constant(10., shape=sample.shape)


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
            kernel_initializer=tf.constant_initializer([[2.0, 1.0], [1.0, 1.0]
                                                       ]),
            bias_initializer=tf.constant_initializer([5.0, 5.0]),
            activation=None,
        )
    ]

  def _get_normal_distribution_spec(self, sample_spec):
    is_multivariate = sample_spec.shape.ndims > 0
    param_properties = tfp.distributions.Normal.parameter_properties()
    input_param_spec = {  # pylint: disable=g-complex-comprehension
        name: tensor_spec.TensorSpec(
            shape=properties.shape_fn(sample_spec.shape),
            dtype=sample_spec.dtype)
        for name, properties in param_properties.items()
    }

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


class CqlSacAgentTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(CqlSacAgentTest, self).setUp()
    self._obs_spec = tensor_spec.BoundedTensorSpec([2],
                                                   tf.float32,
                                                   minimum=0,
                                                   maximum=1)
    self._time_step_spec = ts.time_step_spec(self._obs_spec)
    self._action_spec = tensor_spec.BoundedTensorSpec([1], tf.float32, -1, 1)
    self._random_seed = 0

  @parameterized.parameters((1.0, 10, -2.307371), (10.0, 10, -23.073713))
  def testCqlLoss(self, cql_alpha, num_cql_samples, expected_loss):
    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        cql_alpha=cql_alpha,
        num_cql_samples=num_cql_samples,
        include_critic_entropy_term=False,
        use_lagrange_cql_alpha=False,
        random_seed=self._random_seed,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[5], [6]], dtype=tf.float32)

    loss = agent._cql_loss(
        time_steps, actions, training=False) * agent._get_cql_alpha()

    self.initialize_v1_variables()
    loss_ = self.evaluate(loss)

    self.assertAllClose(loss_, expected_loss)

  def testAgentTrajectoryTrain(self):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._obs_spec,
        self._action_spec,
        fc_layer_params=(10,),
        continuous_projection_net=tanh_normal_projection_network
        .TanhNormalProjectionNetwork)

    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=actor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        cql_alpha=5.0,
        num_cql_samples=1,
        include_critic_entropy_term=False,
        use_lagrange_cql_alpha=False)

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

    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=actor_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
        cql_alpha=5.0,
        num_cql_samples=1,
        include_critic_entropy_term=False,
        use_lagrange_cql_alpha=False)

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

  @parameterized.parameters((False, 0., False, [16.3, 28.1]),
                            (True, 0., True, [7.3, 19.1]),
                            (False, 0.1, False, [16.269377, 28.07928]),
                            (False, 0.1, True, [16.269377, 28.07928]))
  def testCriticLoss(self, include_critic_entropy_term, reward_noise_variance,
                     use_tf_variable, td_targets):
    if use_tf_variable:
      reward_noise_variance = tf.Variable(reward_noise_variance)
    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=None,
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        cql_alpha=1.0,
        num_cql_samples=1,
        include_critic_entropy_term=include_critic_entropy_term,
        use_lagrange_cql_alpha=False,
        reward_noise_variance=reward_noise_variance,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[5], [6]], dtype=tf.float32)

    rewards = tf.constant([10, 20], dtype=tf.float32)
    discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
    next_observations = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    next_time_steps = ts.transition(next_observations, rewards, discounts)

    pred_td_targets = [7., 10.]
    self.evaluate(tf.compat.v1.global_variables_initializer())

    # Expected critic loss has factor of 2, for the two TD3 critics.
    expected_loss = self.evaluate(2 * tf.compat.v1.losses.mean_squared_error(
        tf.constant(td_targets), tf.constant(pred_td_targets)))

    loss = agent._critic_loss_with_optional_entropy_term(
        time_steps,
        actions,
        next_time_steps,
        td_errors_loss_fn=tf.math.squared_difference)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.parameters((0, 6), (1, 13.404237))
  def testActorLoss(self, num_bc_steps, expected_loss):
    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=DummyCriticNet(),
        actor_network=DummyActorNet(self._obs_spec, self._action_spec),
        actor_optimizer=None,
        critic_optimizer=None,
        alpha_optimizer=None,
        cql_alpha=1.0,
        num_cql_samples=1,
        include_critic_entropy_term=False,
        use_lagrange_cql_alpha=False,
        num_bc_steps=num_bc_steps,
        actor_policy_ctor=DummyActorPolicy)

    observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    actions = tf.constant([[5], [6]], dtype=tf.float32)

    loss = agent.actor_loss(time_steps, actions)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    loss_ = self.evaluate(loss)
    self.assertAllClose(loss_, expected_loss)

  @parameterized.parameters(
      (0.0, 10, False, False, 0.850366),
      (1.0, 10, False, True, 4.571599),
      (10.0, 10, False, False, 38.06277),
      (10.0, 10, True, False, 46.153343))
  def testTrainWithRnn(self, cql_alpha, num_cql_samples,
                       include_critic_entropy_term, use_lagrange_cql_alpha,
                       expected_loss):
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

    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=critic_net,
        actor_network=actor_net,
        actor_optimizer=optimizer_fn(1e-3),
        critic_optimizer=optimizer_fn(1e-3),
        alpha_optimizer=optimizer_fn(1e-3),
        cql_alpha=cql_alpha,
        num_cql_samples=num_cql_samples,
        include_critic_entropy_term=include_critic_entropy_term,
        use_lagrange_cql_alpha=use_lagrange_cql_alpha,
        random_seed=self._random_seed,
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

    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, (), time_steps.step_type,
                                       time_steps.reward, time_steps.discount)

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
      loss = self.evaluate(agent.train(experience))
      self.assertAllClose(loss.loss, expected_loss)
      self.assertEqual(self.evaluate(counter), 1)

  @parameterized.parameters(
      (True, False, (-1, 10.0), 5.0, 3.032653, 2.904868, 3.137269),
      (False, False, (-1, 0), 5.0, 5.0, 2.904868, 3.137269),
      (False, True, (-1, 0), 5.0, 6.0, 2.904868, 3.137269))
  def testTrainWithLagrange(self, use_lagrange_cql_alpha,
                            use_variable_for_cql_alpha,
                            log_cql_alpha_clipping,
                            expected_cql_alpha_step_one,
                            expected_cql_alpha_step_two,
                            expected_cql_loss_step_one,
                            expected_cql_loss_step_two):
    if use_variable_for_cql_alpha:
      cql_alpha = tf.Variable(5.0)
      cql_alpha_var = cql_alpha  # Getting around type checking.
    else:
      cql_alpha = 5.0
    cql_alpha_learning_rate = 0.5
    cql_tau = 10
    num_cql_samples = 5

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        self._obs_spec, self._action_spec, fc_layer_params=None)
    critic_net = critic_network.CriticNetwork(
        (self._obs_spec, self._action_spec),
        observation_fc_layer_params=(16,),
        action_fc_layer_params=(16,),
        joint_fc_layer_params=(16,),
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    counter = common.create_variable('test_train_counter')
    optimizer_fn = tf.compat.v1.train.AdamOptimizer
    agent = cql_sac_agent.CqlSacAgent(
        self._time_step_spec,
        self._action_spec,
        critic_network=critic_net,
        actor_network=actor_net,
        actor_optimizer=optimizer_fn(1e-3),
        critic_optimizer=optimizer_fn(1e-3),
        alpha_optimizer=optimizer_fn(1e-3),
        cql_alpha=cql_alpha,
        num_cql_samples=num_cql_samples,
        include_critic_entropy_term=False,
        use_lagrange_cql_alpha=use_lagrange_cql_alpha,
        cql_alpha_learning_rate=cql_alpha_learning_rate,
        cql_tau=cql_tau,
        random_seed=self._random_seed,
        log_cql_alpha_clipping=log_cql_alpha_clipping,
        train_step_counter=counter)

    batch_size = 5
    observations = tf.constant(
        [[[1, 2], [3, 4]]] * batch_size, dtype=tf.float32)
    actions = tf.constant([[[0], [1]]] * batch_size, dtype=tf.float32)
    time_steps = ts.TimeStep(
        step_type=tf.constant([[1] * 2] * batch_size, dtype=tf.int32),
        reward=tf.constant([[1] * 2] * batch_size, dtype=tf.float32),
        discount=tf.constant([[1] * 2] * batch_size, dtype=tf.float32),
        observation=observations)

    experience = trajectory.Trajectory(time_steps.step_type, observations,
                                       actions, (), time_steps.step_type,
                                       time_steps.reward, time_steps.discount)

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
      # Training step one.
      self.assertEqual(self.evaluate(counter), 0)
      loss = self.evaluate(agent.train(experience))
      self.assertEqual(self.evaluate(counter), 1)
      self.assertAllClose(loss.extra.cql_loss, expected_cql_loss_step_one)
      self.assertAllClose(loss.extra.cql_alpha, expected_cql_alpha_step_one)
      if use_lagrange_cql_alpha:
        self.assertGreater(loss.extra.cql_alpha_loss, 0)
      else:
        self.assertEqual(loss.extra.cql_alpha_loss, 0)

      # Training step two.
      if use_variable_for_cql_alpha:
        cql_alpha_var.assign_add(1)
      loss = self.evaluate(agent.train(experience))
      self.assertEqual(self.evaluate(counter), 2)
      self.assertAllClose(loss.extra.cql_loss, expected_cql_loss_step_two)
      # GPU (V100) needs slightly increased to pass.
      if tf.test.is_gpu_available():
        self.assertAllClose(
            loss.extra.cql_alpha,
            expected_cql_alpha_step_two,
            atol=4.5e-5,
            rtol=1.5e-5)
      else:
        self.assertAllClose(
            loss.extra.cql_alpha,
            expected_cql_alpha_step_two,
            rtol=2e-5,
            atol=2e-5)


if __name__ == '__main__':
  tf.test.main()
