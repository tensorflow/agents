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

"""Tests for greedy_multi_objective_neural_agent.py."""

from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

import collections
from typing import List

import numpy as np
import tensorflow as tf

from tf_agents.bandits.agents import greedy_multi_objective_neural_agent as greedy_multi_objective_agent
from tf_agents.bandits.drivers import driver_utils
from tf_agents.bandits.multi_objective import multi_objective_scalarizer
from tf_agents.bandits.networks import global_and_arm_feature_network
from tf_agents.bandits.networks import heteroscedastic_q_network as heteroscedastic_q_net
from tf_agents.bandits.policies import policy_utilities
from tf_agents.bandits.specs import utils as bandit_spec_utils
from tf_agents.networks import network
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common


class DummyNet(network.Network):

  def __init__(self, observation_spec: types.NestedTensorSpec,
               kernel_weights: np.ndarray, bias: np.ndarray):
    """A simple linear network.

    Args:
      observation_spec: The observation specification.
      kernel_weights: A 2-d numpy array of shape [input_size, output_size].
      bias: A 1-d numpy array of shape [output_size].
    """
    super(DummyNet, self).__init__(observation_spec, (), 'DummyNet')
    assert len(kernel_weights.shape) == 2
    assert len(bias.shape) == 1
    assert kernel_weights.shape[1] == bias.shape[0]

    # Store custom layers that can be serialized through the Checkpointable API.
    self._dummy_layers = [
        tf.keras.layers.Dense(
            kernel_weights.shape[1],
            kernel_initializer=tf.constant_initializer(kernel_weights),
            bias_initializer=tf.constant_initializer(bias))
    ]

  def call(self, inputs: tf.Tensor, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    for layer in self._dummy_layers:
      inputs = layer(inputs)
    return inputs, network_state


class HeteroscedasticDummyNet(heteroscedastic_q_net.HeteroscedasticQNetwork):

  def __init__(self, kernel_weights: np.ndarray, bias: np.ndarray):
    """A simple linear heteroscedastic network.

    Args:
      kernel_weights: A 2-d numpy array of shape [input_size, output_size].
      bias: A 1-d numpy array of shape [output_size].
    """
    assert len(kernel_weights.shape) == 2
    assert len(bias.shape) == 1
    assert kernel_weights.shape[1] == bias.shape[0]

    input_spec = array_spec.ArraySpec([kernel_weights.shape[0]], np.float32)
    action_spec = array_spec.BoundedArraySpec([1], np.float32, 1,
                                              kernel_weights.shape[1])

    input_tensor_spec = tensor_spec.from_spec(input_spec)
    action_tensor_spec = tensor_spec.from_spec(action_spec)

    super(HeteroscedasticDummyNet, self).__init__(input_tensor_spec,
                                                  action_tensor_spec)
    self._value_layer = tf.keras.layers.Dense(
        kernel_weights.shape[1],
        kernel_initializer=tf.constant_initializer(kernel_weights),
        bias_initializer=tf.constant_initializer(bias))

    self._log_variance_layer = tf.keras.layers.Dense(
        kernel_weights.shape[1],
        kernel_initializer=tf.constant_initializer(kernel_weights),
        bias_initializer=tf.constant_initializer(bias))

  def call(self, inputs: tf.Tensor, step_type=None, network_state=()):
    del step_type
    inputs = tf.cast(inputs, tf.float32)
    value = self._value_layer(inputs)
    log_variance = self._log_variance_layer(inputs)
    predictions = collections.namedtuple('QBanditNetworkResult',
                                         ('q_value_logits', 'log_variance'))
    predictions = predictions(value, log_variance)

    return predictions, network_state


def _get_initial_and_final_steps(observations: types.NestedTensorOrArray,
                                 objectives: np.ndarray):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  assert len(objectives.shape) == 2
  assert objectives.shape[0] == batch_size
  assert objectives.shape[1] > 1
  if isinstance(observations, np.ndarray):
    observations = tf.constant(
        observations, dtype=tf.float32, name='observation')
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=objectives.shape, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(objectives, dtype=tf.float32, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      observations)
  return initial_step, final_step


def _get_initial_and_final_steps_with_action_mask(
    observations: types.NestedTensorOrArray, objectives: np.ndarray):
  batch_size = tf.nest.flatten(observations)[0].shape[0]
  assert len(objectives.shape) == 2
  assert objectives.shape[0] == batch_size
  assert objectives.shape[1] > 1
  initial_step = ts.TimeStep(
      tf.constant(
          ts.StepType.FIRST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(0.0, dtype=tf.float32, shape=objectives.shape, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size], name='discount'),
      (observations[0], observations[1]))
  final_step = ts.TimeStep(
      tf.constant(
          ts.StepType.LAST,
          dtype=tf.int32,
          shape=[batch_size],
          name='step_type'),
      tf.constant(objectives, dtype=tf.float32, name='reward'),
      tf.constant(1.0, dtype=tf.float32, shape=[batch_size],
                  name='discount'), (tf.nest.map_structure(
                      lambda x: x + 100., observations[0]), observations[1]))
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


class MultiObjectiveAgentTest(tf.test.TestCase):

  def setUp(self):
    super(MultiObjectiveAgentTest, self).setUp()
    tf.compat.v1.enable_resource_variables()
    self._time_step_spec = ts.time_step_spec(
        observation_spec=tensor_spec.TensorSpec([2], tf.float32),
        reward_spec=tensor_spec.TensorSpec([3], tf.float32))
    self._action_spec = tensor_spec.BoundedTensorSpec(
        dtype=tf.int32, shape=(), minimum=0, maximum=2)
    self._observation_spec = self._time_step_spec.observation
    hv_params = [
        multi_objective_scalarizer.HyperVolumeScalarizer.PARAMS(
            slope=1, offset=0)
    ] * 3
    self._scalarizer = multi_objective_scalarizer.HyperVolumeScalarizer(
        [0, 1, 0.01], hv_params)
    self._bias = np.asarray([-1, -1, -1])
    self._kernel_weights = [
        np.asarray([[1, 2, 3], [4, 5, 6]]),
        np.asarray([[3, 1, 2], [5, 4, 6]]),
        np.asarray([[2, 3, 1], [5, 6, 4]])
    ]

  def _create_objective_networks(self) -> List[DummyNet]:
    return [
        DummyNet(self._observation_spec, weights, self._bias)
        for weights in self._kernel_weights
    ]

  def testCreateAgent(self):
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=None)
    self.assertIsNotNone(agent.policy)
    self.assertEqual(len(agent._variables_to_train()), 6)

  def testCreateAgentWithHeteroscedasticNetworks(self):
    objective_networks = self._create_objective_networks()
    objective_networks[-1] = HeteroscedasticDummyNet(self._kernel_weights[-1],
                                                     self._bias)
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=objective_networks,
        optimizer=None)
    self.assertIsNotNone(agent.policy)
    self.assertEqual(len(agent._variables_to_train()), 8)
    self.assertAllEqual(agent._heteroscedastic, [False, False, True])

  def testCreateAgentWithTooFewObjectiveNetworksRaisesError(self):
    with self.assertRaisesRegexp(
        ValueError,
        'Number of objectives should be at least two, but found to be 1'):
      greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
          self._time_step_spec,
          self._action_spec,
          self._scalarizer,
          objective_networks=[self._create_objective_networks()[0]],
          optimizer=None)

  def testCreateAgentWithWrongActionsRaisesError(self):
    action_spec = tensor_spec.BoundedTensorSpec((5, 6, 7), tf.float32, 0, 2)
    with self.assertRaisesRegexp(ValueError,
                                 'Action spec must be a scalar; got shape'):
      greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
          self._time_step_spec,
          action_spec,
          self._scalarizer,
          objective_networks=self._create_objective_networks(),
          optimizer=None)

  def testInitializeAgent(self):
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=None)
    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))

  def testLoss(self):
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=None)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    objectives = np.array([[8, 12, 11], [25, 18, 32]], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, objectives)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)

    init_op = agent.initialize()
    if not tf.executing_eagerly():
      with self.cached_session() as sess:
        common.initialize_uninitialized_variables(sess)
        self.assertIsNone(sess.run(init_op))
    loss, _ = agent._loss(experience)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss), 0.0)

  def testPolicy(self):
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=None)
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual([2], action_step.action.shape)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [2, 0])

  def testPolicySetScalarizationParameters(self):
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=None)
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    policy.scalarizer.set_parameters(
        direction=tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32),
        transform_params={
            multi_objective_scalarizer.HyperVolumeScalarizer.SLOPE_KEY:
                tf.constant([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1]],
                            dtype=tf.float32),
            multi_objective_scalarizer.HyperVolumeScalarizer.OFFSET_KEY:
                tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
        })
    action_step = policy.action(time_steps)
    # Batch size 2.
    self.assertAllEqual([2], action_step.action.shape)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    actions = self.evaluate(action_step.action)
    self.assertAllEqual(actions, [2, 1])

  def testInitializeRestoreAgent(self):
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=None)
    observations = tf.constant([[1, 2], [2, 1]], dtype=tf.float32)
    time_steps = ts.restart(observations, batch_size=2)
    policy = agent.policy
    action_step = policy.action(time_steps)
    self.evaluate(tf.compat.v1.initialize_all_variables())

    checkpoint = tf.train.Checkpoint(agent=agent)

    latest_checkpoint = tf.train.latest_checkpoint(self.get_temp_dir())
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)

    if tf.executing_eagerly():
      self.evaluate(checkpoint_load_status.initialize_or_restore())
      self.assertAllEqual(self.evaluate(action_step.action), [2, 0])
    else:
      with self.cached_session() as sess:
        checkpoint_load_status.initialize_or_restore(sess)
        self.assertAllEqual(sess.run(action_step.action), [2, 0])

  def testTrainAgent(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=optimizer)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    objectives = np.array([[1, 2, 3], [6, 5, 4]], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, objectives)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss_before), 763.5)
    self.assertLess(self.evaluate(loss_after), 763.5)

  def testTrainAgentWithHeteroscedasticNetworks(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    objective_networks = self._create_objective_networks()
    objective_networks[-1] = HeteroscedasticDummyNet(self._kernel_weights[-1],
                                                     self._bias)
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=objective_networks,
        optimizer=optimizer)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    objectives = np.array([[1, 2, 3], [6, 5, 4]], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, objectives)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss_before), 350.2502672)
    self.assertLess(self.evaluate(loss_after), 350.2502672)

  def testTrainAgentWithWrongNumberOfObjectivesRaisesError(self):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    objective_networks = self._create_objective_networks()
    objective_networks.pop(0)
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        self._time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=objective_networks,
        optimizer=optimizer)
    observations = np.array([[1, 2], [3, 4]], dtype=np.float32)
    actions = np.array([0, 1], dtype=np.int32)
    objectives = np.array([[1, 2, 3], [6, 5, 4]], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, objectives)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    with self.assertRaisesRegexp(
        ValueError,
        'The number of objectives in the objective_values tensor: 3 is '
        'different from the number of objective networks: 2'):
      agent.train(experience, None)

  def testTrainAgentWithMask(self):
    time_step_spec = ts.time_step_spec(
        observation_spec=(tensor_spec.TensorSpec([2], tf.float32),
                          tensor_spec.TensorSpec([3], tf.int32)),
        reward_spec=tensor_spec.TensorSpec([3], tf.float32))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        time_step_spec,
        self._action_spec,
        self._scalarizer,
        objective_networks=self._create_objective_networks(),
        optimizer=optimizer,
        observation_and_action_constraint_splitter=lambda x: (x[0], x[1]))
    observations = (np.array([[1, 2], [3, 4]], dtype=np.float32),
                    np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int32))
    actions = np.array([0, 1], dtype=np.int32)
    objectives = np.array([[1, 2, 3], [6, 5, 4]], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps_with_action_mask(
        observations, objectives)
    action_step = _get_action_step(actions)
    experience = _get_experience(initial_step, action_step, final_step)
    loss_before, _ = agent.train(experience, None)
    loss_after, _ = agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())
    self.assertAllClose(self.evaluate(loss_before), 763.5)
    self.assertLess(self.evaluate(loss_after), 763.5)

  def testTrainPerArmAgent(self):
    obs_spec = bandit_spec_utils.create_per_arm_observation_spec(
        2, 3, 4, add_num_actions_feature=True)
    time_step_spec = ts.time_step_spec(
        observation_spec=obs_spec,
        reward_spec=tensor_spec.TensorSpec([3], tf.float32))
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 3)
    objective_networks = [
        global_and_arm_feature_network.create_feed_forward_common_tower_network(
            obs_spec, (4, 3), (3, 4), (4, 2)) for _ in range(3)
    ]
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
    agent = greedy_multi_objective_agent.GreedyMultiObjectiveNeuralAgent(
        time_step_spec,
        action_spec,
        self._scalarizer,
        objective_networks=objective_networks,
        accepts_per_arm_features=True,
        optimizer=optimizer)
    observations = {
        bandit_spec_utils.GLOBAL_FEATURE_KEY:
            tf.constant([[1, 2], [3, 4]], dtype=tf.float32),
        bandit_spec_utils.PER_ARM_FEATURE_KEY:
            tf.cast(
                tf.reshape(tf.range(24), shape=[2, 4, 3]), dtype=tf.float32),
        bandit_spec_utils.NUM_ACTIONS_FEATURE_KEY:
            tf.ones([2], dtype=tf.int32)
    }
    actions = np.array([0, 3], dtype=np.int32)
    objectives = np.array([[1, 2, 3], [6, 5, 4]], dtype=np.float32)
    initial_step, final_step = _get_initial_and_final_steps(
        observations, objectives)
    action_step = policy_step.PolicyStep(
        action=tf.convert_to_tensor(actions),
        info=policy_utilities.PerArmPolicyInfo(
            chosen_arm_features=np.array([[1, 2, 3], [3, 2, 1]],
                                         dtype=np.float32)))
    experience = _get_experience(initial_step, action_step, final_step)
    agent.train(experience, None)
    self.evaluate(tf.compat.v1.initialize_all_variables())


if __name__ == '__main__':
  tf.test.main()
