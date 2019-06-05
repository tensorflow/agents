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

"""Tests for TF Agents reinforce_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tensorflow.python.util import nest  # pylint:disable=g-direct-tensorflow-import  # TF internal


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
    self._layers = [
        tf.keras.layers.Dense(
            single_action_spec.shape.num_elements() * 2,
            activation=activation_fn,
            kernel_initializer=tf.compat.v1.initializers.constant(
                [[2, 1], [1, 1]]),
            bias_initializer=tf.compat.v1.initializers.constant(5),
        ),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    states = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    for layer in self.layers:
      states = layer(states)

    single_action_spec = tf.nest.flatten(self._output_tensor_spec)[0]
    actions, stdevs = tf.split(states, 2, axis=1)
    actions = tf.reshape(actions, [-1] + single_action_spec.shape.as_list())
    stdevs = tf.reshape(stdevs, [-1] + single_action_spec.shape.as_list())
    actions = tf.nest.pack_sequence_as(self._output_tensor_spec, [actions])
    stdevs = tf.nest.pack_sequence_as(self._output_tensor_spec, [stdevs])

    distribution = nest.map_structure_up_to(
        self._output_tensor_spec, tfp.distributions.Normal, actions, stdevs)
    return distribution, network_state


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

    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertEqual(self.evaluate(counter), 0)
    self.evaluate(loss)
    self.assertEqual(self.evaluate(counter), 1)


if __name__ == '__main__':
  tf.test.main()
